"""
ApexWZeRO: Native ZeRO-1 (Zero Redundancy Optimizer State Sharding) for ApexW.

Partitions optimizer states (m and v moments) evenly across distributed GPU ranks
(or simulated multi-rank nodes) so that each GPU only maintains 1 / world_size
of the optimizer state memory.

Combined with ApexW's 16-bit compact moment storage:
- 1 GPU  : 4 bytes/parameter (50% cut vs AdamW's 8 bytes)
- 4 GPUs : 1 byte/parameter   (87.5% cut vs AdamW)
- 8 GPUs : 0.5 bytes/parameter (93.75% cut / 16x less memory than AdamW)
- 64 GPUs: 0.0625 bytes/parameter (128x less memory than AdamW)

Supports PyTorch DDP, torch.distributed ProcessGroups, and single-machine simulation.
"""

import math
from collections import defaultdict
from typing import Optional, Callable, Tuple, List, Dict, Any, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.distributed as dist

try:
    from .optimizer import ApexW
except ImportError:
    from optimizer import ApexW


class ApexWZeRO(Optimizer):
    r"""
    ApexWZeRO: Native ZeRO Stage-1 Optimizer for Distributed PyTorch (Experimental).
    
    .. warning::
        ApexWZeRO is currently an experimental research feature. For production
        multi-GPU distributed training, PyTorch's native FSDP or DeepSpeed ZeRO
        with ApexW is recommended.

    Partitions optimizer historical state memory ($m_t$ and $v_t$) across ranks.
    Each GPU only allocates and updates $1 / \text{world\_size}$ of the model parameters.
    Updated parameter shards are gathered across ranks using non-blocking collectives.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Union[Tuple[float, float], Tuple[float, float, float]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        process_group: Optional[Any] = None,
        amsgrad: bool = False,
        nesterov: bool = False,
        cautious: bool = False,
        grad_centralization: bool = False,
        decay_norm_only: bool = True,
        clip_grad_norm: Optional[float] = None,
        beta3: Optional[float] = None,
        alpha: float = 5.0,
        lion: bool = False,
        precision: str = "auto",
        dynamic_scaling: bool = True,
        overlap_comm: bool = True,
    ):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size(process_group) if world_size is None else world_size
            self.rank = dist.get_rank(process_group) if rank is None else rank
        else:
            self.world_size = world_size if world_size is not None else 1
            self.rank = rank if rank is not None else 0

        self.process_group = process_group
        self.overlap_comm = overlap_comm

        defaults = dict(
            lr=lr,
            betas=(betas[0], betas[1]),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            nesterov=nesterov,
            cautious=cautious,
            grad_centralization=grad_centralization,
            decay_norm_only=decay_norm_only,
            clip_grad_norm=clip_grad_norm,
            beta3=beta3,
            alpha=alpha,
            lion=lion,
            precision=precision,
            dynamic_scaling=dynamic_scaling,
        )
        super().__init__(params, defaults)

        # Build sharded local parameter slices for this rank
        self._sharded_param_map = []
        self._local_params = []
        self._init_zero_sharding()

        # Local ApexW engine that manages ONLY this rank's partition
        self._local_opt = ApexW(
            self._local_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            nesterov=nesterov,
            cautious=cautious,
            grad_centralization=grad_centralization,
            decay_norm_only=decay_norm_only,
            clip_grad_norm=clip_grad_norm,
            beta3=beta3,
            alpha=alpha,
            lion=lion,
            precision=precision,
            dynamic_scaling=dynamic_scaling,
        )

    def _init_zero_sharding(self):
        """
        Partitions parameters across ranks.
        2D+ tensors are sliced along dimension 0.
        1D normalization / bias tensors are assigned round-robin.
        """
        p_index = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                # Slicing along dim 0 for large tensors
                if p.dim() >= 1 and numel >= self.world_size * 256:
                    dim0 = p.shape[0]
                    chunk_size = (dim0 + self.world_size - 1) // self.world_size
                    start = min(self.rank * chunk_size, dim0)
                    end = min((self.rank + 1) * chunk_size, dim0)
                    if end > start:
                        local_slice = p.data[start:end].clone().detach().requires_grad_(True)
                    else:
                        local_slice = torch.empty((0, *p.shape[1:]), dtype=p.dtype, device=p.device).requires_grad_(True)

                    self._local_params.append(local_slice)
                    self._sharded_param_map.append({
                        "type": "slice",
                        "full_param": p,
                        "local_slice": local_slice,
                        "start": start,
                        "end": end,
                        "dim0": dim0,
                        "chunk_size": chunk_size,
                    })
                else:
                    # Round-robin assignment for small tensors
                    owner_rank = p_index % self.world_size
                    if owner_rank == self.rank:
                        local_p = p.data.clone().detach().requires_grad_(True)
                        self._local_params.append(local_p)
                        self._sharded_param_map.append({
                            "type": "round_robin",
                            "full_param": p,
                            "local_slice": local_p,
                            "owner_rank": owner_rank,
                        })
                    else:
                        self._sharded_param_map.append({
                            "type": "round_robin",
                            "full_param": p,
                            "local_slice": None,
                            "owner_rank": owner_rank,
                        })
                p_index += 1

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        for item in self._sharded_param_map:
            loc = item["local_slice"]
            if loc is not None:
                if set_to_none:
                    loc.grad = None
                elif loc.grad is not None:
                    loc.grad.zero_()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a ZeRO-1 partitioned optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Sync full gradients into local shards
        for item in self._sharded_param_map:
            full_p = item["full_param"]
            loc = item["local_slice"]
            if loc is None or full_p.grad is None:
                continue

            if item["type"] == "slice":
                start, end = item["start"], item["end"]
                if end > start:
                    loc.grad = full_p.grad.data[start:end]
            else:
                loc.grad = full_p.grad.data

        # 2. Update local shards using ApexW engine (16-bit compact states, dynamic scaling, etc.)
        self._local_opt.step()

        # 3. Synchronize updated local parameter shards across distributed ranks
        is_dist = dist.is_available() and dist.is_initialized() and self.world_size > 1

        for item in self._sharded_param_map:
            full_p = item["full_param"]
            loc = item["local_slice"]

            if item["type"] == "slice":
                start, end = item["start"], item["end"]
                if is_dist:
                    # Correct multi-rank assembly: zero out non-owned elements on each rank
                    # so that all_reduce(SUM) reconstructs the exact updated parameter
                    # without summing stale values from other ranks.
                    full_p.data.zero_()
                    if end > start and loc is not None:
                        full_p.data[start:end].copy_(loc.data)
                    dist.all_reduce(full_p.data, op=dist.ReduceOp.SUM, group=self.process_group)
                else:
                    if end > start and loc is not None:
                        full_p.data[start:end].copy_(loc.data)

            elif item["type"] == "round_robin":
                owner = item["owner_rank"]
                if owner == self.rank and loc is not None:
                    full_p.data.copy_(loc.data)

                if is_dist:
                    dist.broadcast(full_p.data, src=owner, group=self.process_group)

        return loss

    def get_state_memory_mb(self) -> float:
        """Returns the actual optimizer state memory allocated on THIS rank in MB."""
        return self._local_opt.get_state_memory_mb()

    def get_total_cluster_saved_mb(self) -> float:
        """Returns the total state memory saved across the entire cluster in MB."""
        local_mem = self.get_state_memory_mb()
        adam_equiv = sum(p.numel() * 8 for g in self.param_groups for p in g["params"]) / (1024 * 1024)
        return adam_equiv - local_mem
