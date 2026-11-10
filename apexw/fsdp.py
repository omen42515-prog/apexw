"""
ApexW FSDP (Fully Sharded Data Parallel) & Scale-Out Distributed Module.

Provides native integration for:
1. PyTorch FSDP1: Inspects and extracts 2D matrix slices from flattened 1D FlatParameters.
2. PyTorch FSDP2: DTensor-aware routing for per-parameter sharded models.
3. DistMuon: Communication-efficient distributed Newton-Schulz orthogonalization across GPU ranks
   (inspired by Moonshot AI's Moonlight and Keller Jordan's distributed Muon).
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import warnings

try:
    from .muon import zeropower_via_newtonschulz5, _determine_work_dtype
except ImportError:
    from muon import zeropower_via_newtonschulz5, _determine_work_dtype


def _to_global_rank(group: Optional[Any], group_rank: int) -> int:
    """Translates a process-group relative rank to its global rank for PyTorch collective ops."""
    if group is None or not (dist.is_available() and dist.is_initialized()):
        return group_rank
    if hasattr(dist, "get_global_rank"):
        try:
            return dist.get_global_rank(group, group_rank)
        except Exception:
            return group_rank
    return group_rank


def is_fsdp_flat_param(param: Tensor) -> bool:
    """Returns True if the tensor is an FSDP1 FlatParameter."""
    if getattr(param, "_is_flat_param", False):
        return True
    cls_name = param.__class__.__name__
    return "FlatParameter" in cls_name or hasattr(param, "_param_infos")


def extract_fsdp_param_infos(flat_param: Tensor) -> List[Dict[str, Any]]:
    """
    Extracts underlying parameter metadata (name, shape, offset, numel) from an FSDP1 FlatParameter.

    Supports both real PyTorch FSDP1 FlatParameter instances (which store _shapes, _numels,
    and _fqns directly on the FlatParameter) and custom/mock param info objects.
    """
    infos = []
    param_infos = getattr(flat_param, "_param_infos", None)
    if param_infos is None:
        return infos

    shapes = getattr(flat_param, "_shapes", None)
    numels = getattr(flat_param, "_numels", None)
    fqns = getattr(flat_param, "_fqns", None)

    offset = 0
    for i, info in enumerate(param_infos):
        # 1. Parameter name
        if fqns is not None and i < len(fqns):
            param_name = fqns[i]
        elif hasattr(info, "param_name"):
            param_name = info.param_name
        else:
            param_name = str(i)

        # 2. Shape
        if shapes is not None and i < len(shapes):
            shape = shapes[i]
        elif hasattr(info, "shape"):
            shape = info.shape
        else:
            shape = None

        # 3. Element count
        if numels is not None and i < len(numels):
            numel = numels[i]
        elif hasattr(info, "numel"):
            numel = info.numel
        elif shape is not None:
            numel = 1
            for d in shape:
                numel *= d
        else:
            numel = 0

        dtype = getattr(info, "dtype", getattr(flat_param, "dtype", torch.float32))
        is_2d = False
        if shape is not None and len(shape) >= 2:
            m = shape[0]
            n = numel // m if m > 0 else 0
            is_2d = min(m, n) >= 32

        infos.append({
            "param_name": param_name,
            "shape": shape,
            "numel": numel,
            "dtype": dtype,
            "offset": offset,
            "is_2d": is_2d,
        })
        offset += numel

    return infos


def is_dtensor(param: Any) -> bool:
    """Returns True if param is a PyTorch Distributed Tensor (DTensor / FSDP2)."""
    return hasattr(param, "_local_tensor") or param.__class__.__name__ == "DTensor"


class DistMuon(Optimizer):
    """
    DistMuon: Communication-Optimal Distributed Muon for Multi-GPU Scaling (DDP / FSDP).

    In multi-GPU clusters, computing Newton-Schulz polar decomposition across 8+ GPUs
    is redundant if every rank computes the same matrix operations, and catastrophic
    if matrix slices are all-gathered inefficiently.

    DistMuon partitions 2D parameter updates across data-parallel ranks (rank-cyclic
    assignment). Each rank ONLY computes the Newton-Schulz orthogonalization for its
    assigned subset of layers, and then synchronizes updated weights across ranks via
    non-blocking broadcasts, reducing Newton-Schulz compute overhead by world_size!
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 1e-2,
        ns_steps: int = 4,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        process_group: Optional[Any] = None,
    ):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size(process_group) if world_size is None else world_size
            self.rank = dist.get_rank(process_group) if rank is None else rank
        else:
            self.world_size = world_size or 1
            self.rank = rank or 0

        self.process_group = process_group
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            # Update momentum for all parameters locally
            for p in params:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16 if p.is_cuda else torch.float32)
                state["step"] += 1

                buf = state["momentum"]
                g_fp = p.grad.float()
                buf_fp = buf.float().mul_(momentum).add_(g_fp, alpha=1.0 - momentum)
                buf.copy_(buf_fp)

            # Rank-cyclic distribution: Rank `r` computes Newton-Schulz for `params[r::world_size]`
            handles = []
            for idx, p in enumerate(params):
                owner_rank = idx % self.world_size

                if self.rank == owner_rank:
                    state = self.state[p]
                    buf_fp = state["momentum"].float()
                    g_fp = p.grad.float()
                    update_source = g_fp.lerp(buf_fp, momentum) if nesterov else buf_fp

                    if update_source.ndim >= 2 and min(update_source.shape[0], update_source.numel() // update_source.shape[0]) >= 16:
                        U = zeropower_via_newtonschulz5(update_source, steps=ns_steps)
                        m_dim, n_dim = U.shape[0], U.numel() // U.shape[0]
                        scale = max(1.0, math.sqrt(m_dim / n_dim))
                    else:
                        std = update_source.std() + 1e-7
                        U = update_source / std
                        scale = 0.05

                    eff_lr = lr * scale
                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(U.to(p.dtype), alpha=-eff_lr)

                # Broadcast updated parameters from owner rank to all other ranks
                if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                    src_global = _to_global_rank(self.process_group, owner_rank)
                    handle = dist.broadcast(p.data, src=src_global, group=self.process_group, async_op=True)
                    handles.append(handle)
                elif self.world_size > 1 and not getattr(self, "_warned_unsynced", False):
                    warnings.warn(
                        f"DistMuon: world_size={self.world_size} specified, but torch.distributed is not "
                        f"initialized. Parameters on non-owner ranks will not be synchronized.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._warned_unsynced = True

            # Wait for all cross-rank parameter synchronizations to complete
            for h in handles:
                h.wait()

        return loss
