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

import copy
import warnings
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


class ApexWZeRO(Optimizer):
    r"""
    ApexWZeRO: Native ZeRO Stage-1 Optimizer for Distributed PyTorch.

    Partitions optimizer historical state memory ($m_t$ and $v_t$) across ranks.
    Each GPU only allocates and updates $1 / \text{world\_size}$ of the model parameters.
    Updated parameter shards are gathered across ranks using non-blocking collectives.
    """

    _step_supports_amp_scaling = True

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
        precision: str = "compact",
        dynamic_scaling: bool = True,
        overlap_comm: bool = True,
        **apex_kwargs,
    ):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size(process_group) if world_size is None else world_size
            self.rank = dist.get_rank(process_group) if rank is None else rank
        else:
            self.world_size = world_size if world_size is not None else 1
            self.rank = rank if rank is not None else 0

        self.process_group = process_group
        self.overlap_comm = overlap_comm

        if isinstance(params, nn.Module):
            model = params
            for name, p in model.named_parameters():
                if not hasattr(p, "_param_name"):
                    setattr(p, "_param_name", name)
            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                params = list(model.parameters())

        is_hybrid = apex_kwargs.get("use_muon", False) or (
            isinstance(precision, str)
            and precision.lower().strip() in ("hybrid", "hybrid_apexw", "hybrid-apexw", "apex_hybrid", "hybrid_compact")
        )
        if is_hybrid:
            apex_kwargs["use_muon"] = True
            precision = "compact"

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
            **apex_kwargs,
        )
        super().__init__(params, defaults)

        # Build sharded local parameter slices for this rank
        self._sharded_param_map = []
        self._local_params = []
        self._local_param_groups = []
        self._init_zero_sharding()

        # Local ApexW engine that manages ONLY this rank's partition,
        # preserving all per-group hyperparameters (lr, weight_decay, betas, eps, precision, etc.)
        local_kwargs = dict(apex_kwargs)
        local_kwargs.setdefault("int4_m", False)
        self._local_opt = ApexW(
            self._local_param_groups,
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
            **local_kwargs,
        )

    def _init_zero_sharding(self):
        """
        Partitions parameters across ranks while preserving per-group configuration.
        2D+ tensors are sliced along dimension 0.
        1D normalization / bias tensors are assigned round-robin.
        """
        p_index = 0
        self._local_params = []
        self._local_param_groups = []

        for group_idx, group in enumerate(self.param_groups):
            local_group = {k: v for k, v in group.items() if k not in ("params", "use_muon")}
            local_group["params"] = []

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

                    local_group["params"].append(local_slice)
                    self._local_params.append(local_slice)
                    self._sharded_param_map.append({
                        "type": "slice",
                        "full_param": p,
                        "local_slice": local_slice,
                        "start": start,
                        "end": end,
                        "dim0": dim0,
                        "chunk_size": chunk_size,
                        "group_idx": group_idx,
                    })
                else:
                    # Round-robin assignment for small tensors
                    owner_rank = p_index % self.world_size
                    if owner_rank == self.rank:
                        local_p = p.data.clone().detach().requires_grad_(True)
                        local_group["params"].append(local_p)
                        self._local_params.append(local_p)
                        self._sharded_param_map.append({
                            "type": "round_robin",
                            "full_param": p,
                            "local_slice": local_p,
                            "owner_rank": owner_rank,
                            "group_idx": group_idx,
                        })
                    else:
                        self._sharded_param_map.append({
                            "type": "round_robin",
                            "full_param": p,
                            "local_slice": None,
                            "owner_rank": owner_rank,
                            "group_idx": group_idx,
                        })
                p_index += 1

            self._local_param_groups.append(local_group)

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

        # Seamless PyTorch AMP GradScaler synchronization
        grad_scale = getattr(self, "grad_scale", None)
        found_inf = getattr(self, "found_inf", None)

        if found_inf is not None and bool(found_inf):
            return loss

        if grad_scale is not None:
            self._local_opt.grad_scale = grad_scale
        if found_inf is not None:
            self._local_opt.found_inf = found_inf

        try:
            # 1. Sync per-group hyperparameters (e.g. from learning rate schedulers)
            if len(self.param_groups) == len(self._local_opt.param_groups):
                for outer_g, inner_g in zip(self.param_groups, self._local_opt.param_groups):
                    for k, v in outer_g.items():
                        if k != "params":
                            inner_g[k] = v
            else:
                for outer_g in self.param_groups:
                    for inner_g in self._local_opt.param_groups:
                        for k, v in outer_g.items():
                            if k not in ("params", "lr"):
                                inner_g[k] = v
                        if "lr" in outer_g and "lr" in inner_g:
                            outer_init = outer_g.get("_initial_lr", outer_g["lr"])
                            inner_init = inner_g.get("_initial_lr", inner_g["lr"])
                            ratio = outer_g["lr"] / max(outer_init, 1e-12)
                            inner_g["lr"] = inner_init * ratio

            # 2. Sync full gradients into local shards
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

            # 3. Update local shards using inner engine (16-bit compact states, dynamic scaling, etc.)
            self._local_opt.step()

            # 4. Synchronize updated local parameter shards across distributed ranks
            is_dist = dist.is_available() and dist.is_initialized() and self.world_size > 1
            if self.world_size > 1 and not (dist.is_available() and dist.is_initialized()):
                if not getattr(self, "_warned_unsynced", False):
                    warnings.warn(
                        f"ApexWZeRO: world_size={self.world_size} specified, but torch.distributed is not "
                        f"initialized. Shards on other ranks will not be synchronized across processes.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._warned_unsynced = True

            work_handles = []
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
                        if self.overlap_comm:
                            h = dist.all_reduce(full_p.data, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True)
                            work_handles.append(h)
                        else:
                            dist.all_reduce(full_p.data, op=dist.ReduceOp.SUM, group=self.process_group)
                    else:
                        if end > start and loc is not None:
                            full_p.data[start:end].copy_(loc.data)

                elif item["type"] == "round_robin":
                    owner = item["owner_rank"]
                    if owner == self.rank and loc is not None:
                        full_p.data.copy_(loc.data)

                    if is_dist:
                        src_global = _to_global_rank(self.process_group, owner)
                        if self.overlap_comm:
                            h = dist.broadcast(full_p.data, src=src_global, group=self.process_group, async_op=True)
                            work_handles.append(h)
                        else:
                            dist.broadcast(full_p.data, src=src_global, group=self.process_group)

            if work_handles:
                for h in work_handles:
                    h.wait()
        finally:
            if hasattr(self._local_opt, "grad_scale"):
                self._local_opt.grad_scale = None
            if hasattr(self._local_opt, "found_inf"):
                self._local_opt.found_inf = None
            if hasattr(self, "grad_scale"):
                self.grad_scale = None
            if hasattr(self, "found_inf"):
                self.found_inf = None

        return loss

    def get_state_memory_mb(self) -> float:
        """Returns the actual optimizer state memory allocated on THIS rank in MB."""
        if hasattr(self._local_opt, "get_state_memory_mb"):
            return self._local_opt.get_state_memory_mb()
        total_bytes = 0
        for p_state in self._local_opt.state.values():
            for v in p_state.values():
                if isinstance(v, Tensor):
                    total_bytes += v.numel() * v.element_size()
        return total_bytes / (1024 * 1024)

    def get_rank_saved_mb(self) -> float:
        """Returns the optimizer state memory saved on THIS rank compared to standard AdamW in MB."""
        local_mem = self.get_state_memory_mb()
        adam_equiv = sum(p.numel() * 8 for g in self.param_groups for p in g["params"]) / (1024 * 1024)
        return max(0.0, adam_equiv - local_mem)

    def get_total_cluster_saved_mb(self) -> float:
        """Returns the total state memory saved across the entire cluster in MB."""
        local_mem = self.get_state_memory_mb()
        adam_equiv = sum(p.numel() * 8 for g in self.param_groups for p in g["params"]) / (1024 * 1024)
        w = max(1, self.world_size)
        cluster_adam_equiv = w * adam_equiv
        cluster_zero_mem = w * local_mem
        return max(0.0, cluster_adam_equiv - cluster_zero_mem)

    def print_summary(self) -> None:
        """Prints a human-readable summary of ZeRO state memory usage and cluster savings."""
        rank_mem = self.get_state_memory_mb()
        rank_saved = self.get_rank_saved_mb()
        cluster_saved = self.get_total_cluster_saved_mb()
        total_numel = sum(p.numel() for g in self.param_groups for p in g["params"])
        adam_equiv = (total_numel * 8) / (1024 * 1024)
        pct_saved = (rank_saved / max(1e-6, adam_equiv)) * 100.0

        print("\n" + "=" * 60)
        print(" ApexWZeRO Optimizer Summary")
        print("=" * 60)
        print(f" World Size         : {self.world_size} (Rank {self.rank})")
        print(f" Total Parameters   : {total_numel:,}")
        print(f" Local State Memory : {rank_mem:.2f} MB")
        print(f" AdamW Equivalent   : {adam_equiv:.2f} MB")
        print(f" Memory Saved / Rank: {rank_saved:.2f} MB ({pct_saved:.1f}% reduction)")
        print(f" Cluster Memory Save: {cluster_saved:.2f} MB across {self.world_size} ranks")
        print("=" * 60 + "\n")

    def state_dict(self):
        """Returns checkpoint state including standard PyTorch keys, inner optimizer and ZeRO shard metadata."""
        local_sd = self._local_opt.state_dict()
        is_hybrid = hasattr(self._local_opt, "muon_opt")
        all_local_params = [p for g in self._local_opt.param_groups for p in g["params"]]
        param_to_state_id = {id(p): idx for idx, p in enumerate(all_local_params)}
        param_to_local_gidx = {}
        for g_i, grp in enumerate(self._local_opt.param_groups):
            for p in grp["params"]:
                param_to_local_gidx[id(p)] = g_i

        shard_meta = []
        for e in self._sharded_param_map:
            entry = {"type": e["type"], "group_idx": e.get("group_idx", 0)}
            loc = e.get("local_slice")
            if loc is not None and id(loc) in param_to_state_id:
                entry["local_state_id"] = param_to_state_id[id(loc)]
                entry["local_group_idx"] = param_to_local_gidx[id(loc)]
            elif is_hybrid and e.get("full_param") is not None:
                from .hybrid import _is_muon_eligible
                is_m = _is_muon_eligible(e["full_param"], exclude_embeddings=True, min_muon_dim=32)
                n_m_groups = len(self._local_opt.muon_opt.param_groups) if getattr(self._local_opt, "muon_opt", None) else 1
                entry["local_group_idx"] = 0 if is_m else n_m_groups
                entry["local_state_id"] = None
            else:
                entry["local_group_idx"] = e.get("group_idx", 0)
                entry["local_state_id"] = None

            if e["type"] == "slice":
                entry["start"] = e["start"]
                entry["end"] = e["end"]
                entry["dim0"] = e["dim0"]
            else:
                entry["owner_rank"] = e.get("owner_rank")
            shard_meta.append(entry)

        return {
            "state": local_sd.get("state", {}),
            "param_groups": copy.deepcopy(local_sd.get("param_groups", [])),
            "local_opt": local_sd,
            "zero_meta": {
                "rank": self.rank,
                "world_size": self.world_size,
                "shard_map": shard_meta,
                "is_hybrid": is_hybrid,
            },
        }

    def add_param_group(self, param_group: dict):
        """Adds a new parameter group, creating sharded local slices for this rank."""
        super().add_param_group(param_group)
        if not hasattr(self, "_local_opt"):
            return
        added_group = self.param_groups[-1]
        local_group = {k: v for k, v in added_group.items() if k not in ("params", "use_muon")}
        local_group["params"] = []

        group_idx = len(self.param_groups) - 1
        p_index = len(self._sharded_param_map)
        for p in added_group["params"]:
            numel = p.numel()
            if p.dim() >= 1 and numel >= self.world_size * 256:
                dim0 = p.shape[0]
                chunk_size = (dim0 + self.world_size - 1) // self.world_size
                start = min(self.rank * chunk_size, dim0)
                end = min((self.rank + 1) * chunk_size, dim0)
                if end > start:
                    local_slice = p.data[start:end].clone().detach().requires_grad_(True)
                else:
                    local_slice = torch.empty((0, *p.shape[1:]), dtype=p.dtype, device=p.device).requires_grad_(True)
                local_group["params"].append(local_slice)
                self._local_params.append(local_slice)
                self._sharded_param_map.append({
                    "type": "slice",
                    "full_param": p,
                    "local_slice": local_slice,
                    "start": start,
                    "end": end,
                    "dim0": dim0,
                    "chunk_size": chunk_size,
                    "group_idx": group_idx,
                })
            else:
                owner_rank = p_index % self.world_size
                if owner_rank == self.rank:
                    local_p = p.data.clone().detach().requires_grad_(True)
                    local_group["params"].append(local_p)
                    self._local_params.append(local_p)
                    self._sharded_param_map.append({
                        "type": "round_robin",
                        "full_param": p,
                        "local_slice": local_p,
                        "owner_rank": owner_rank,
                        "group_idx": group_idx,
                    })
                else:
                    self._sharded_param_map.append({
                        "type": "round_robin",
                        "full_param": p,
                        "local_slice": None,
                        "owner_rank": owner_rank,
                        "group_idx": group_idx,
                    })
            p_index += 1

        self._local_param_groups.append(local_group)
        self._local_opt.add_param_group(local_group)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restores inner optimizer state from a checkpoint produced by state_dict() or standard PyTorch dict."""
        if "local_opt" in state_dict:
            self._local_opt.load_state_dict(state_dict["local_opt"])
        elif "state" in state_dict:
            self._local_opt.load_state_dict(state_dict)

        # Reflect restored hyperparameters back to outer param groups
        for outer_g, inner_g in zip(self.param_groups, self._local_opt.param_groups):
            for k, v in inner_g.items():
                if k != "params":
                    outer_g[k] = v

        # Cross-device parameter alignment: ensure loaded states match tensor devices
        for p, p_state in self._local_opt.state.items():
            if isinstance(p, Tensor):
                for k, v in list(p_state.items()):
                    if isinstance(v, Tensor) and v.device != p.device:
                        p_state[k] = v.to(p.device)

        meta = state_dict.get("zero_meta", {})
        saved_rank = meta.get("rank")
        saved_world_size = meta.get("world_size")

        if saved_world_size is not None and saved_world_size != self.world_size:
            import warnings
            warnings.warn(
                f"ApexWZeRO: checkpoint was saved with world_size={saved_world_size}, "
                f"but current world_size={self.world_size}. To reshard optimizer states across "
                f"different GPU counts without mismatch, use reshard_zero_checkpoint().",
                UserWarning,
                stacklevel=2,
            )
        if saved_rank is not None and saved_rank != self.rank:
            import warnings
            warnings.warn(
                f"ApexWZeRO: checkpoint rank={saved_rank} loaded on rank={self.rank}. "
                "This is expected when redistributing checkpoints across different ranks.",
                UserWarning,
                stacklevel=2,
            )


def _assemble_sliced_tensor(
    slices: List[Tuple[int, int, Dict[str, Any]]],
    key: str,
    dim0: int,
    shape_tail: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    dequant_fn: Optional[Callable] = None,
    scales_key: Optional[str] = None,
    block_size: int = 64,
) -> torch.Tensor:
    """Reconstructs a full dimension-0 tensor from distributed rank slices."""
    out = torch.zeros((dim0, *shape_tail), dtype=dtype, device=device)
    for start, end, s_state in slices:
        if end > start and key in s_state:
            val = s_state[key]
            if dequant_fn is not None and scales_key in s_state:
                out[start:end] = dequant_fn(val, s_state[scales_key], block_size=block_size)
            else:
                out[start:end] = val.reshape(end - start, *shape_tail) if val.ndim > 1 else val
    return out


def _slice_moment_state(
    state: Dict[str, Any],
    start: int,
    end: int,
    dim0: int,
    q_info: Dict[str, Any],
    is_quad_v: bool,
    moment_keys: set,
) -> Dict[str, Any]:
    """Partitions a reconstructed parameter moment state into a target rank shard."""
    from .optimizer import _quantize_blockwise_int8, _quantize_blockwise_uint8

    out: Dict[str, Any] = {}
    for k, v in state.items():
        if k == "exp_avg" and "m_bsize" in q_info:
            sl = v[start:end].clone()
            if sl.numel() > 0:
                q_m, q_s = _quantize_blockwise_int8(sl, block_size=q_info["m_bsize"], stochastic=False)
            else:
                q_m = torch.empty(sl.shape, dtype=torch.int8, device=sl.device)
                q_s = torch.empty((0, 1), dtype=torch.float32, device=sl.device)
            out["exp_avg"] = q_m
            out["exp_avg_scales"] = q_s
        elif k == "exp_avg_sq" and "v_bsize" in q_info:
            sl = v[start:end].clone()
            if sl.numel() > 0:
                q_v, q_vs = _quantize_blockwise_uint8(sl, block_size=q_info["v_bsize"], stochastic=False)
            else:
                q_v = torch.empty(sl.shape, dtype=torch.uint8, device=sl.device)
                q_vs = torch.empty((0, 1), dtype=torch.float32, device=sl.device)
            out["exp_avg_sq"] = q_v
            out["exp_avg_sq_scales"] = q_vs
        elif k == "exp_avg_sq" and is_quad_v and torch.is_tensor(v):
            v_2d = v.reshape(dim0, -1)
            out[k] = v_2d[start:end].reshape(-1).clone()
            out["is_quad_v"] = True
        elif k in moment_keys and torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == dim0:
            out[k] = v[start:end].clone()
        else:
            out[k] = v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
    return out


def reshard_zero_checkpoint(
    rank_checkpoints: List[Dict[str, Any]],
    target_world_size: int,
) -> List[Dict[str, Any]]:
    """
    Reshards ZeRO optimizer checkpoints from an N-GPU run to an M-GPU run.

    Allows resuming training seamlessly when changing GPU counts (e.g., saving on 4 GPUs
    and resuming on 8 GPUs, or consolidating to 1 GPU for inference/eval).
    Preserves 16-bit compact moment buffers and dynamic scaling factors without loss.

    Args:
        rank_checkpoints: List of state_dicts saved across source ranks [rank 0, ..., rank N-1].
        target_world_size: New target world size (number of GPUs for resumption).

    Returns:
        List of resharded state_dicts for the target ranks [target rank 0, ..., target rank M-1].
    """
    source_world_size = len(rank_checkpoints)
    if source_world_size == 0:
        raise ValueError("rank_checkpoints list cannot be empty.")
    if target_world_size <= 0:
        raise ValueError(f"target_world_size must be >= 1, got {target_world_size}")

    if source_world_size == target_world_size:
        return [copy.deepcopy(sd) for sd in rank_checkpoints]

    base_sd = rank_checkpoints[0]
    rank_0_local = base_sd.get("local_opt", base_sd)
    rank_0_zero_meta = base_sd.get("zero_meta", {})
    shard_maps = [ckpt.get("zero_meta", {}).get("shard_map", []) for ckpt in rank_checkpoints]

    # Handle synthetic/mock checkpoints without shard maps
    has_shard_maps = bool(shard_maps and shard_maps[0])
    if not has_shard_maps:
        resharded_checkpoints = []
        for target_rank in range(target_world_size):
            source_idx = target_rank % source_world_size
            src_sd = rank_checkpoints[source_idx]
            new_sd = {
                "state": copy.deepcopy(src_sd.get("state", {})),
                "param_groups": copy.deepcopy(src_sd.get("param_groups", rank_0_local.get("param_groups", []))),
                "local_opt": {
                    "state": copy.deepcopy(src_sd.get("local_opt", {}).get("state", src_sd.get("state", {}))),
                    "param_groups": copy.deepcopy(src_sd.get("local_opt", {}).get("param_groups", rank_0_local.get("param_groups", []))),
                },
                "zero_meta": {
                    "rank": target_rank,
                    "world_size": target_world_size,
                    "shard_map": [],
                },
            }
            resharded_checkpoints.append(new_sd)
        return resharded_checkpoints

    num_params = len(shard_maps[0])

    # 1. Map global parameter index to local state index for each source rank
    rank_local_indices: List[Dict[int, int]] = []
    for r in range(source_world_size):
        l_idx = 0
        p_to_l = {}
        for p_idx, entry in enumerate(shard_maps[r]):
            if entry.get("type") == "slice":
                p_to_l[p_idx] = l_idx
                l_idx += 1
            elif entry.get("type") == "round_robin" and entry.get("owner_rank") == r:
                p_to_l[p_idx] = l_idx
                l_idx += 1
        rank_local_indices.append(p_to_l)

    # 2. Reconstruct full parameter moment tensors from source rank checkpoints
    _MOMENT_KEYS = {"exp_avg", "exp_avg_sq", "exp_avg_slow", "max_exp_avg_sq", "momentum", "momentum_slow"}
    _SCALE_KEYS = {"exp_avg_scales", "exp_avg_sq_scales", "momentum_scales", "momentum_slow_scales"}
    default_bsize = int((rank_0_local.get("param_groups") or [{}])[0].get("block_size", 64) or 64)
    quantized_meta: Dict[int, Dict[str, Any]] = {}

    param_to_group: Dict[int, int] = {}
    p_offset = 0
    base_groups = rank_0_local.get("param_groups", base_sd.get("param_groups", []))
    for g_idx, g in enumerate(base_groups):
        for _ in g.get("params", []):
            param_to_group[p_offset] = g_idx
            p_offset += 1

    reconstructed_states: Dict[int, Dict[str, Any]] = {}
    for p_idx in range(num_params):
        entry0 = shard_maps[0][p_idx]
        p_type = entry0.get("type")

        if p_type == "slice":
            dim0 = entry0["dim0"]
            slices = []
            for r in range(source_world_size):
                entry_r = shard_maps[r][p_idx]
                l_idx = entry_r.get("local_state_id")
                if l_idx is None and p_idx in rank_local_indices[r]:
                    l_idx = rank_local_indices[r][p_idx]
                if l_idx is not None:
                    src_state = rank_checkpoints[r].get("local_opt", rank_checkpoints[r]).get("state", {}).get(l_idx, {})
                    start = entry_r.get("start", 0)
                    end = entry_r.get("end", 0)
                    slices.append((start, end, src_state))

            full_state: Dict[str, Any] = {}
            ref_entry = next(((st, en, s_st) for st, en, s_st in slices if en > st and s_st), slices[0] if slices else None)
            if ref_entry is not None and ref_entry[2]:
                ref_start, ref_end, ref_state = ref_entry
                ref_row_len = ref_end - ref_start
                q_info: Dict[str, Any] = {}

                # Check if 8-bit block-quantized moments are present; if so, dequantize per slice
                # before assembling dim0 so cross-block resharding never corrupts scales.
                has_q_m = (
                    "exp_avg" in ref_state and torch.is_tensor(ref_state["exp_avg"])
                    and ref_state["exp_avg"].dtype == torch.int8 and "exp_avg_scales" in ref_state
                )
                has_q_v = (
                    "exp_avg_sq" in ref_state and torch.is_tensor(ref_state["exp_avg_sq"])
                    and ref_state["exp_avg_sq"].dtype == torch.uint8 and "exp_avg_sq_scales" in ref_state
                )
                dequant_m = None
                dequant_v = None
                if has_q_m or has_q_v:
                    from .optimizer import (
                        _dequantize_blockwise_int8,
                        _dequantize_blockwise_uint8,
                    )
                    if has_q_m:
                        q_info["m_bsize"] = default_bsize
                        dequant_m = _dequantize_blockwise_int8
                    if has_q_v:
                        q_info["v_bsize"] = default_bsize
                        dequant_v = _dequantize_blockwise_uint8

                is_quad_v = (
                    bool(ref_state.get("is_quad_v", False))
                    or (
                        "exp_avg" in ref_state
                        and "exp_avg_sq" in ref_state
                        and torch.is_tensor(ref_state["exp_avg"])
                        and torch.is_tensor(ref_state["exp_avg_sq"])
                        and ref_state["exp_avg_sq"].numel() * 4 == ref_state["exp_avg"].numel()
                    )
                )

                for k, v in ref_state.items():
                    if k in _SCALE_KEYS:
                        continue
                    if k == "exp_avg" and has_q_m:
                        full_state[k] = _assemble_sliced_tensor(
                            slices, "exp_avg", dim0, v.shape[1:], torch.float32, v.device,
                            dequant_fn=dequant_m, scales_key="exp_avg_scales", block_size=q_info["m_bsize"]
                        )
                    elif k == "exp_avg_sq" and has_q_v:
                        full_state[k] = _assemble_sliced_tensor(
                            slices, "exp_avg_sq", dim0, v.shape[1:], torch.float32, v.device,
                            dequant_fn=dequant_v, scales_key="exp_avg_sq_scales", block_size=q_info["v_bsize"]
                        )
                    elif k == "exp_avg_sq" and is_quad_v and torch.is_tensor(v):
                        ref_m = ref_state["exp_avg"]
                        cols = ref_m.shape[1] if ref_m.dim() >= 2 else 1
                        assembled_v = _assemble_sliced_tensor(
                            slices, "exp_avg_sq", dim0, (cols // 4 if cols >= 4 else 1,), v.dtype, v.device
                        )
                        full_state[k] = assembled_v.reshape(-1)
                        full_state["is_quad_v"] = True
                    elif k in _MOMENT_KEYS and torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == ref_row_len:
                        full_state[k] = _assemble_sliced_tensor(
                            slices, k, dim0, v.shape[1:], v.dtype, v.device
                        )
                    else:
                        full_state[k] = v.clone() if torch.is_tensor(v) else copy.deepcopy(v)

                if q_info:
                    quantized_meta[p_idx] = q_info
            reconstructed_states[p_idx] = full_state
        else:
            owner = entry0.get("owner_rank", 0)
            if owner < source_world_size:
                entry_owner = shard_maps[owner][p_idx]
                l_idx = entry_owner.get("local_state_id")
                if l_idx is None and p_idx in rank_local_indices[owner]:
                    l_idx = rank_local_indices[owner][p_idx]
                if l_idx is not None:
                    owner_state = rank_checkpoints[owner].get("local_opt", rank_checkpoints[owner]).get("state", {}).get(l_idx, {})
                    reconstructed_states[p_idx] = copy.deepcopy(owner_state)
                else:
                    reconstructed_states[p_idx] = {}
            else:
                reconstructed_states[p_idx] = {}

    # 3. Partition reconstructed states according to target_world_size
    resharded_states: List[Dict[int, Dict[str, Any]]] = [{} for _ in range(target_world_size)]
    resharded_shard_maps: List[List[Dict[str, Any]]] = [[] for _ in range(target_world_size)]
    target_local_indices = [0 for _ in range(target_world_size)]
    target_group_local_indices: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(target_world_size)]

    for p_idx in range(num_params):
        s = reconstructed_states.get(p_idx, {})
        ref_tensor = s.get("exp_avg") if s.get("exp_avg") is not None else s.get("momentum")
        numel = ref_tensor.numel() if ref_tensor is not None else 0
        dim = ref_tensor.dim() if ref_tensor is not None else 0
        is_sliced = (dim >= 1) and (numel >= target_world_size * 256)
        entry0 = shard_maps[0][p_idx]
        g_idx = entry0.get("local_group_idx")
        if g_idx is None:
            if is_hybrid:
                is_muon = ("momentum" in s) or entry0.get("is_muon", False)
                n_m = len(rank_0_local.get("muon_state_dict", {}).get("param_groups", [])) or 1
                g_idx = 0 if is_muon else n_m
            else:
                g_idx = entry0.get("group_idx", param_to_group.get(p_idx, 0))

        if is_sliced:
            dim0 = ref_tensor.shape[0]
            chunk_size = (dim0 + target_world_size - 1) // target_world_size
            q_info = quantized_meta.get(p_idx, {})
            is_quad_v_s = bool(s.get("is_quad_v", False))

            for t_rank in range(target_world_size):
                start = min(t_rank * chunk_size, dim0)
                end = min((t_rank + 1) * chunk_size, dim0)
                slice_state = _slice_moment_state(s, start, end, dim0, q_info, is_quad_v_s, _MOMENT_KEYS)
                l_idx = target_local_indices[t_rank]
                resharded_states[t_rank][l_idx] = slice_state
                target_local_indices[t_rank] += 1
                target_group_local_indices[t_rank][g_idx].append(l_idx)
                resharded_shard_maps[t_rank].append({
                    "type": "slice",
                    "start": start,
                    "end": end,
                    "dim0": dim0,
                    "chunk_size": chunk_size,
                    "group_idx": g_idx,
                })
        else:
            owner = p_idx % target_world_size
            for t_rank in range(target_world_size):
                resharded_shard_maps[t_rank].append({
                    "type": "round_robin",
                    "owner_rank": owner,
                    "group_idx": g_idx,
                })
                if t_rank == owner:
                    l_idx = target_local_indices[t_rank]
                    resharded_states[t_rank][l_idx] = {
                        k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                        for k, v in s.items()
                    }
                    target_local_indices[t_rank] += 1
                    target_group_local_indices[t_rank][g_idx].append(l_idx)

    # 4. Assemble resharded checkpoint structures
    resharded_checkpoints = []
    base_param_groups = rank_0_local.get("param_groups", base_sd.get("param_groups", []))
    is_hybrid = bool(
        rank_0_zero_meta.get("is_hybrid", False)
        or "muon_state_dict" in rank_0_local
        or "adamw_state_dict" in rank_0_local
        or "apexw_state_dict" in rank_0_local
    )

    for target_rank in range(target_world_size):
        local_param_groups = []
        for g_idx, g in enumerate(base_param_groups):
            lg = {k: copy.deepcopy(v) for k, v in g.items() if k != "params"}
            lg["params"] = target_group_local_indices[target_rank][g_idx]
            local_param_groups.append(lg)

        new_local_opt: Dict[str, Any] = {
            "state": resharded_states[target_rank],
            "param_groups": local_param_groups,
        }

        if is_hybrid:
            muon_sd_0 = rank_0_local.get("muon_state_dict", {})
            n_muon_groups = len(muon_sd_0.get("param_groups", [])) if muon_sd_0 else 0
            if n_muon_groups == 0:
                n_muon_groups = max(1, len(local_param_groups) // 2) if len(local_param_groups) > 1 else 1

            local_muon_groups = local_param_groups[:n_muon_groups]
            local_adamw_groups = local_param_groups[n_muon_groups:]

            m_pids = [pid for g in local_muon_groups for pid in g["params"]]
            a_pids = [pid for g in local_adamw_groups for pid in g["params"]]
            m_pid_map = {old: new for new, old in enumerate(m_pids)}
            a_pid_map = {old: new for new, old in enumerate(a_pids)}

            muon_sub_groups = []
            for g in local_muon_groups:
                g_sub = {k: copy.deepcopy(v) for k, v in g.items() if k != "params"}
                g_sub["params"] = [m_pid_map[p] for p in g["params"]]
                muon_sub_groups.append(g_sub)
            muon_sub_state = {m_pid_map[k]: copy.deepcopy(resharded_states[target_rank][k]) for k in m_pids if k in resharded_states[target_rank]}

            adamw_sub_groups = []
            for g in local_adamw_groups:
                g_sub = {k: copy.deepcopy(v) for k, v in g.items() if k != "params"}
                g_sub["params"] = [a_pid_map[p] for p in g["params"]]
                adamw_sub_groups.append(g_sub)
            adamw_sub_state = {a_pid_map[k]: copy.deepcopy(resharded_states[target_rank][k]) for k in a_pids if k in resharded_states[target_rank]}

            new_local_opt["muon_state_dict"] = {"state": muon_sub_state, "param_groups": muon_sub_groups}
            new_local_opt["adamw_state_dict"] = {"state": adamw_sub_state, "param_groups": adamw_sub_groups}
            new_local_opt["apexw_state_dict"] = {"state": adamw_sub_state, "param_groups": adamw_sub_groups}

        new_sd = {
            "state": resharded_states[target_rank],
            "param_groups": copy.deepcopy(local_param_groups),
            "local_opt": new_local_opt,
            "zero_meta": {
                "rank": target_rank,
                "world_size": target_world_size,
                "shard_map": resharded_shard_maps[target_rank],
                "is_hybrid": is_hybrid,
            },
        }
        resharded_checkpoints.append(new_sd)

    return resharded_checkpoints


class ZeROStateAdapter:
    """
    Adapter enabling DeepSpeed ZeRO-1/2/3 to shard and reconstruct ApexW's
    16-bit packed states without corrupting dynamic scaling factors (scale_v).
    Supports both static utility methods and instance-based wrapping.
    """

    def __init__(self, optimizer: Optional[Any] = None):
        self.optimizer = optimizer

    def get_local_state_dict(self) -> Dict[str, Any]:
        if self.optimizer is None:
            raise ValueError("ZeROStateAdapter: no optimizer provided.")
        sd = self.optimizer.state_dict()
        scale_v = 1.0
        if hasattr(self.optimizer, "state"):
            for st in self.optimizer.state.values():
                if isinstance(st, dict) and "scale_v" in st:
                    scale_v = st["scale_v"]
                    break
        return self.pack_scaling_metadata(sd, scale_v=scale_v)

    def load_local_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.optimizer is None:
            raise ValueError("ZeROStateAdapter: no optimizer provided.")
        scale_v = self.unpack_scaling_metadata(state_dict)
        self.optimizer.load_state_dict(state_dict)
        if hasattr(self.optimizer, "state"):
            for st in self.optimizer.state.values():
                if isinstance(st, dict) and "scale_v" in st:
                    st["scale_v"] = scale_v

    @staticmethod
    def pack_scaling_metadata(state_dict: Dict[str, Any], scale_v: float = 1.0) -> Dict[str, Any]:
        """Attaches dynamic scaling metadata to the state dict before ZeRO partitioning."""
        state_dict["_apexw_scale_v"] = scale_v
        return state_dict

    @staticmethod
    def unpack_scaling_metadata(state_dict: Dict[str, Any]) -> float:
        """Extracts dynamic scaling factor after ZeRO reconstruction."""
        return float(state_dict.get("_apexw_scale_v", 1.0))
