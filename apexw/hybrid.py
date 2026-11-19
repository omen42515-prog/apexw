"""
HybridApexW: Unified Next-Gen Optimizer combining ApexMuon and ApexW.

Automatically partitions model parameters:
- 2D Hidden Weight Matrices -> ApexMuon (Orthogonal Newton-Schulz, 2 bytes/param)
- 1D Norms, Biases & Embeddings -> ApexW (Compact 16-bit 2-moment AdamW, 4 bytes/param)

Accepts the same ``params`` argument as ``torch.optim.AdamW``:
- A plain ``nn.Module`` (uses named_parameters for name-aware routing).
- A flat iterable of ``Tensor`` parameters.
- A list of param-group dicts ``[{'params': [...], 'lr': ..., 'weight_decay': ...}, ...]``.
  Per-group overrides (``lr``, ``weight_decay``, etc.) are preserved on each
  Muon and AdamW sub-group respectively.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any, Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .optimizer import ApexW, _normalize_precision
from .muon import ApexMuon
from .fsdp import is_fsdp_flat_param


class CombinedOptimizerState(defaultdict):
    """Transparent proxy mapping parameter state lookups dynamically across both sub-optimizers."""

    def __init__(self, muon_opt, apexw_opt=None, adamw_opt=None):
        super().__init__(dict)
        self._muon_opt = muon_opt
        opt = apexw_opt if apexw_opt is not None else adamw_opt
        self._apexw_opt = opt
        self._adamw_opt = opt

    @property
    def muon_state(self):
        return self._muon_opt.state if self._muon_opt is not None else {}

    @property
    def adamw_state(self):
        return self._adamw_opt.state if self._adamw_opt is not None else {}

    @property
    def apexw_state(self):
        return self.adamw_state

    def __getitem__(self, p):
        return self.muon_state[p] if p in self.muon_state else self.adamw_state[p]

    def __setitem__(self, p, val):
        if p in self.muon_state:
            self.muon_state[p] = val
        else:
            self.adamw_state[p] = val

    def __contains__(self, p):
        return p in self.muon_state or p in self.adamw_state

    def __len__(self):
        return len(self.muon_state) + len(self.adamw_state)

    def values(self):
        return list(self.muon_state.values()) + list(self.adamw_state.values())

    def items(self):
        return list(self.muon_state.items()) + list(self.adamw_state.items())


def _is_muon_eligible(p: Tensor, exclude_embeddings: bool, min_muon_dim: int) -> bool:
    """Returns True if parameter p should be routed to the Muon engine."""
    if not p.requires_grad or p.ndim < 2:
        return False
    if is_fsdp_flat_param(p):
        return False
    m = p.shape[0]
    n = p.numel() // m
    if min(m, n) < min_muon_dim:
        return False
    # Heuristic: very wide/tall matrices with extreme aspect ratio are likely token embeddings.
    if exclude_embeddings and max(m, n) >= 1000 and max(m / n, n / m) >= 4.0:
        return False
    return True


def partition_parameters_for_hybrid(
    params_or_model: Union[nn.Module, Iterable[Tensor], List[Dict[str, Any]]],
    min_muon_dim: int = 32,
    exclude_embeddings: bool = True,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Partitions parameters into 2D hidden matrices (Muon) and 1D/embeddings (AdamW).

    Returns flat (unannotated) lists. Per-group metadata from param-group dicts is
    discarded; use ``_partition_grouped_params`` when group overrides must be preserved.
    """
    muon_params: List[Tensor] = []
    adamw_params: List[Tensor] = []

    if isinstance(params_or_model, nn.Module):
        for name, p in params_or_model.named_parameters():
            if not p.requires_grad:
                continue
            name_lower = name.lower()
            is_1d_or_bias = p.ndim < 2 or name_lower.endswith(".bias") or "norm" in name_lower
            is_embed_or_head = exclude_embeddings and (
                any(k in name_lower for k in ("embed", "wte", "wpe", "shared", "lm_head", "cls_head", "score_head"))
                or name_lower == "head.weight"
                or name_lower.endswith(".head.weight")
            )
            is_fsdp = is_fsdp_flat_param(p)

            if is_1d_or_bias or is_embed_or_head or is_fsdp:
                adamw_params.append(p)
            else:
                m = p.shape[0]
                n = p.numel() // m
                is_extreme_aspect = exclude_embeddings and max(m, n) >= 1000 and max(m / n, n / m) >= 4.0
                if min(m, n) >= min_muon_dim and not is_extreme_aspect:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
    else:
        # Collect raw tensors from iterable or param groups
        raw_params = []
        if isinstance(params_or_model, (list, tuple)) and len(params_or_model) > 0 and isinstance(params_or_model[0], dict):
            for group in params_or_model:
                raw_params.extend(group["params"])
        else:
            raw_params = list(params_or_model)

        for p in raw_params:
            if _is_muon_eligible(p, exclude_embeddings, min_muon_dim):
                muon_params.append(p)
            else:
                if p.requires_grad:
                    adamw_params.append(p)

    return muon_params, adamw_params


def _partition_grouped_params(
    param_groups: List[Dict[str, Any]],
    min_muon_dim: int,
    exclude_embeddings: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits a list of param groups into Muon groups and AdamW groups, preserving all
    per-group metadata (lr, weight_decay, betas, etc.).

    Each input group may produce zero, one, or two output groups (one Muon, one AdamW)
    depending on which parameters it contains. Per-group overrides are preserved on each
    output group so the user's intended lr/wd settings are honoured exactly.
    """
    muon_groups: List[Dict[str, Any]] = []
    adamw_groups: List[Dict[str, Any]] = []

    for group in param_groups:
        muon_ps = []
        adamw_ps = []
        is_muon_group = group.get("use_muon", None)

        for p in group["params"]:
            if not p.requires_grad:
                continue
            if is_muon_group is True and p.ndim >= 2:
                muon_ps.append(p)
            elif is_muon_group is False or p.ndim < 2:
                adamw_ps.append(p)
            else:
                if _is_muon_eligible(p, exclude_embeddings, min_muon_dim):
                    muon_ps.append(p)
                else:
                    adamw_ps.append(p)

        # Strip 'params' and 'use_muon' keys and carry every other override (lr, weight_decay, betas, …)
        group_meta = {k: v for k, v in group.items() if k not in ("params", "use_muon")}

        if muon_ps:
            muon_groups.append({"params": muon_ps, **group_meta})
        if adamw_ps:
            adamw_groups.append({"params": adamw_ps, **group_meta})

    return muon_groups, adamw_groups


class HybridApexW(Optimizer):
    """HybridApexW: Unified Dual-Engine Optimizer for Modern Deep Networks.

    Routes 2D weight matrices to ApexMuon and 1D vectors/embeddings to ApexW.

    Accepts the same ``params`` argument as ``torch.optim.AdamW``:
    - A plain ``nn.Module`` (uses named_parameters for name-aware partitioning).
    - A flat iterable of ``Tensor`` parameters.
    - A list of param-group dicts ``[{'params': [...], 'lr': ..., 'weight_decay': ...}, ...]``.
      Per-group overrides (``lr``, ``weight_decay``, etc.) are preserved on each
      Muon and AdamW sub-group respectively.
      Explicit ``use_muon: True`` or ``use_muon: False`` per group are strictly respected.

    Args:
        params: Model, flat param iterable, or list of param-group dicts.
        lr: Learning rate for AdamW components (default: 1e-3).
        muon_lr: Learning rate for Muon components (default: None, inherits ``lr``).
        muon_momentum: Momentum for Muon (default: 0.95).
        adamw_betas: Betas for AdamW components (default: None -> falls back to ``betas`` kwarg or (0.9, 0.99)).
        weight_decay: Decoupled weight decay (default: 1e-2).
        ns_steps: Number of Newton-Schulz iterations (default: 5; matches standalone Muon convergence).
        scale_mode: Matrix scaling method: 'rms_matched' (default, 0.2*sqrt(max(m,n)) matching AdamW RMS),
            'none', 'spectral', 'rms', or 'aspect_ratio'.
        ns_method: Newton-Schulz algorithm: 'newtonschulz5' (default) or 'polar_express'.
        cautious: Enable cautious updates across both engines (default: False).
        precision: State storage precision: 'compact', 'auto', 'ultra', or 'full' (default: 'compact').
        fused: Enable fused kernel execution (default: None; auto-optimizes both engines).
        capturable: Whether this optimizer is safe to capture in a CUDA graph (default: False).
        compiled: Whether to accelerate Newton-Schulz and Polar Express iterations with
            torch.compile using mode='reduce-overhead' for CUDA Graph kernel fusion (default: True).
    """

    _step_supports_amp_scaling = True

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        muon_lr: Optional[float] = None,
        muon_momentum: float = 0.95,
        adamw_betas: Optional[Tuple[float, float]] = None,
        apexw_betas: Optional[Tuple[float, float]] = None,
        weight_decay: float = 1e-2,
        muon_weight_decay: Optional[float] = None,
        ns_steps: int = 5,  # 5 steps matches validated standalone Muon convergence
        scale_mode: str = "rms_matched",  # 0.2 * sqrt(max(m, n)) matches AdamW update RMS at drop-in lr
        ns_method: str = "newtonschulz5",
        beta3: Optional[float] = None,
        alpha: float = 5.0,
        cautious: bool = False,
        min_muon_dim: int = 32,
        exclude_embeddings: bool = True,
        precision: Union[str, torch.dtype] = "compact",
        batch_bmm: Optional[bool] = None,
        nesterov: bool = False,
        fused: Optional[bool] = None,
        capturable: bool = False,
        compiled: bool = False,
        muon_compiled: Optional[bool] = None,
        **apex_kwargs,
    ):
        precision = _normalize_precision(precision)
        if precision == "hybrid":
            precision = "compact"
        # Pop 'betas' from apex_kwargs so ApexW(..., use_muon=True, betas=(...)) never
        # raises TypeError: got multiple values for keyword argument 'betas'.
        extra_betas = apex_kwargs.pop("betas", None)
        if extra_betas is not None:
            if len(extra_betas) >= 3 and beta3 is None:
                beta3 = extra_betas[2]
            if adamw_betas is None:
                adamw_betas = (float(extra_betas[0]), float(extra_betas[1]))
        if apexw_betas is not None and adamw_betas is None:
            adamw_betas = apexw_betas
        if adamw_betas is None:
            adamw_betas = (0.9, 0.999) if precision in ("8bit", "full") else (0.9, 0.99)

        if muon_lr is None:
            muon_lr = (lr * 5.0) if scale_mode in ("rms_matched", "spectral") else lr
        if muon_weight_decay is None:
            muon_weight_decay = weight_decay
        if muon_compiled is not None:
            compiled = muon_compiled

        self._min_muon_dim = min_muon_dim
        self._exclude_embeddings = exclude_embeddings
        self._muon_init_kwargs = dict(
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=nesterov,
            weight_decay=muon_weight_decay,
            ns_steps=ns_steps,
            beta3=beta3,
            alpha=alpha,
            cautious=cautious,
            precision=precision,
            batch_bmm=batch_bmm,
            fused=fused,
            capturable=capturable,
            scale_mode=scale_mode,
            ns_method=ns_method,
            compiled=compiled,
        )
        self._adamw_init_kwargs = dict(
            lr=lr,
            betas=adamw_betas,
            weight_decay=weight_decay,
            beta3=beta3,
            alpha=alpha,
            cautious=cautious,
            precision=precision,
            fused=fused,
            capturable=capturable,
            **apex_kwargs,
        )

        # ── Detect whether the user passed grouped param dicts ───────────────
        # Grouped params carry per-group overrides (lr, weight_decay, etc.) that
        # must be preserved on each sub-optimizer group.  Plain iterables and
        # nn.Module are handled by the original flat-partition path.
        is_grouped = (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        )

        if is_grouped:
            # Split each group into a Muon portion and an AdamW portion,
            # carrying all per-group metadata (lr, weight_decay, betas, …).
            muon_groups, adamw_groups = _partition_grouped_params(
                params,
                min_muon_dim=min_muon_dim,
                exclude_embeddings=exclude_embeddings,
            )
            self.muon_params  = [p for g in muon_groups  for p in g["params"]]
            self.adamw_params = [p for g in adamw_groups for p in g["params"]]
            self.apexw_params = self.adamw_params

            self.muon_opt = ApexMuon(muon_groups, **self._muon_init_kwargs) if self.muon_params else None
            self.adamw_opt = ApexW(adamw_groups, **self._adamw_init_kwargs) if self.adamw_params else None
            self.apexw_opt = self.adamw_opt

        else:
            self.muon_params, self.adamw_params = partition_parameters_for_hybrid(
                params,
                min_muon_dim=min_muon_dim,
                exclude_embeddings=exclude_embeddings,
            )
            self.apexw_params = self.adamw_params

            self.muon_opt = ApexMuon(self.muon_params, **self._muon_init_kwargs) if len(self.muon_params) > 0 else None
            self.adamw_opt = ApexW(self.adamw_params, **self._adamw_init_kwargs) if len(self.adamw_params) > 0 else None
            self.apexw_opt = self.adamw_opt

        all_groups = []
        if self.muon_opt is not None:
            all_groups.extend(self.muon_opt.param_groups)
        if self.adamw_opt is not None:
            all_groups.extend(self.adamw_opt.param_groups)

        defaults = dict(lr=lr, muon_lr=muon_lr, weight_decay=weight_decay)
        super().__init__(self.muon_params + self.adamw_params, defaults)
        self.param_groups = all_groups
        self.state = CombinedOptimizerState(self.muon_opt, self.adamw_opt)
        self._initialized = True

    def zero_grad(self, set_to_none: bool = True):
        if self.muon_opt is not None:
            self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.adamw_opt is not None:
            self.adamw_opt.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict):
        """Add a param group to the :class:`HybridApexW`'s `param_groups`.

        Automatically partitions the new parameters between the internal ApexMuon
        and ApexW engines, respecting any explicit 'use_muon' group tags.
        """
        super().add_param_group(param_group)
        if not getattr(self, "_initialized", False):
            return
        muon_groups, adamw_groups = _partition_grouped_params(
            [param_group],
            min_muon_dim=self._min_muon_dim,
            exclude_embeddings=self._exclude_embeddings,
        )
        for g in muon_groups:
            self.muon_params.extend(g["params"])
            if self.muon_opt is not None:
                self.muon_opt.add_param_group(g)
            else:
                self.muon_opt = ApexMuon([g], **self._muon_init_kwargs)
        for g in adamw_groups:
            self.adamw_params.extend(g["params"])
            if self.adamw_opt is not None:
                self.adamw_opt.add_param_group(g)
            else:
                self.adamw_opt = ApexW([g], **self._adamw_init_kwargs)
        self.apexw_params = self.adamw_params
        self.apexw_opt = self.adamw_opt

        all_groups = []
        if self.muon_opt is not None:
            all_groups.extend(self.muon_opt.param_groups)
        if self.adamw_opt is not None:
            all_groups.extend(self.adamw_opt.param_groups)
        self.param_groups = all_groups
        self.state = CombinedOptimizerState(self.muon_opt, self.adamw_opt)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Seamless PyTorch AMP GradScaler synchronization
        grad_scale = getattr(self, "grad_scale", None)
        found_inf = getattr(self, "found_inf", None)

        try:
            if self.muon_opt is not None:
                if grad_scale is not None:
                    self.muon_opt.grad_scale = grad_scale
                if found_inf is not None:
                    self.muon_opt.found_inf = found_inf
                self.muon_opt.step()

            if self.adamw_opt is not None:
                if grad_scale is not None:
                    self.adamw_opt.grad_scale = grad_scale
                if found_inf is not None:
                    self.adamw_opt.found_inf = found_inf
                self.adamw_opt.step()
        finally:
            for sub in (self.muon_opt, self.adamw_opt):
                if sub is not None:
                    if hasattr(sub, "grad_scale"):
                        sub.grad_scale = None
                    if hasattr(sub, "found_inf"):
                        sub.found_inf = None
            if hasattr(self, "grad_scale"):
                self.grad_scale = None
            if hasattr(self, "found_inf"):
                self.found_inf = None

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """Returns standard PyTorch-compatible optimizer state_dict without duplicating state tensors."""
        muon_sd = self.muon_opt.state_dict() if self.muon_opt else None
        adamw_sd = self.adamw_opt.state_dict() if self.adamw_opt else None

        combined_state: Dict[int, Any] = {}
        combined_groups: List[Dict[str, Any]] = []
        offset = 0

        if muon_sd is not None:
            max_id = -1
            for g in muon_sd.get("param_groups", []):
                g_copy = dict(g)
                ids = [pid + offset for pid in g.get("params", [])]
                g_copy["params"] = ids
                if ids:
                    max_id = max(max_id, max(ids))
                combined_groups.append(g_copy)
            for pid, st in muon_sd.get("state", {}).items():
                combined_state[pid + offset] = st
            if combined_state or max_id >= 0:
                offset = max(max_id + 1, max((k + 1 for k in combined_state), default=0))

        if adamw_sd is not None:
            for g in adamw_sd.get("param_groups", []):
                g_copy = dict(g)
                g_copy["params"] = [pid + offset for pid in g.get("params", [])]
                combined_groups.append(g_copy)
            for pid, st in adamw_sd.get("state", {}).items():
                combined_state[pid + offset] = st

        return {
            "state": combined_state,
            "param_groups": combined_groups,
            "muon_state_dict": muon_sd,
            "adamw_state_dict": adamw_sd,
            "apexw_state_dict": adamw_sd,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restores optimizer state, preserving sub-optimizer dtypes and quantized buffers."""
        apexw_sd = state_dict.get("apexw_state_dict") or state_dict.get("adamw_state_dict")
        if state_dict.get("muon_state_dict") or apexw_sd:
            if self.muon_opt and state_dict.get("muon_state_dict"):
                self.muon_opt.load_state_dict(state_dict["muon_state_dict"])
            if self.adamw_opt and apexw_sd:
                self.adamw_opt.load_state_dict(apexw_sd)
        elif "state" in state_dict and "param_groups" in state_dict:
            n_muon_groups = len(self.muon_opt.param_groups) if self.muon_opt else 0
            sg_list = state_dict["param_groups"]
            flat_state = state_dict["state"]
            if self.muon_opt is not None:
                m_groups = sg_list[:n_muon_groups]
                m_state = {pid: flat_state[pid] for g in m_groups for pid in g.get("params", []) if pid in flat_state}
                self.muon_opt.load_state_dict({"state": m_state, "param_groups": m_groups})
            if self.adamw_opt is not None:
                a_groups = sg_list[n_muon_groups:]
                a_state = {pid: flat_state[pid] for g in a_groups for pid in g.get("params", []) if pid in flat_state}
                self.adamw_opt.load_state_dict({"state": a_state, "param_groups": a_groups})

        all_groups = []
        if self.muon_opt is not None:
            all_groups.extend(self.muon_opt.param_groups)
        if self.adamw_opt is not None:
            all_groups.extend(self.adamw_opt.param_groups)
        self.param_groups = all_groups

        # Ensure all state tensors automatically match their target parameter device
        for p, p_state in self.state.items():
            if isinstance(p, Tensor):
                for k, v in list(p_state.items()):
                    if isinstance(v, Tensor) and v.device != p.device:
                        p_state[k] = v.to(p.device)

    def get_state_memory_mb(self) -> float:
        """Returns the total allocated or estimated optimizer state memory across both engines in MB."""
        muon_alloc = self._allocated_state_bytes(self.muon_opt)
        adamw_alloc = self._allocated_state_bytes(self.adamw_opt)
        muon_bytes = muon_alloc if muon_alloc > 0 else self._estimate_muon_bytes()
        adamw_bytes = adamw_alloc if adamw_alloc > 0 else self._estimate_apexw_bytes()
        return (muon_bytes + adamw_bytes) / (1024 * 1024)

    def _estimate_apexw_bytes(self) -> int:
        return self._estimate_adamw_bytes()

    @staticmethod
    def _allocated_state_bytes(sub_opt: Optional[Optimizer]) -> int:
        if sub_opt is None:
            return 0
        total = 0
        for p_state in sub_opt.state.values():
            for v in p_state.values():
                if isinstance(v, Tensor):
                    total += v.numel() * v.element_size()
        return total

    def _estimate_muon_bytes(self) -> int:
        if self.muon_opt is None:
            return 0
        total = 0
        for g in self.muon_opt.param_groups:
            prec = g.get("precision", "compact")
            n_bufs = 2 if g.get("beta3") is not None else 1
            for p in g["params"]:
                n = p.numel()
                if prec in ("8bit", "ultra"):
                    n_blocks = (n + 255) // 256
                    if n >= 524288:
                        buf_bytes = (((n + 255) // 256) * 256 // 4) * 3 + n_blocks * 4
                    else:
                        buf_bytes = n + n_blocks * 4
                    total += buf_bytes * n_bufs
                else:
                    total += n * 4 * n_bufs
        return total

    def _estimate_adamw_bytes(self) -> int:
        if self.adamw_opt is None:
            return 0
        total = 0
        for g in self.adamw_opt.param_groups:
            prec = g.get("precision", "compact")
            betas = g.get("betas", (0.9, 0.99))
            has_slow = g.get("beta3") is not None
            for p in g["params"]:
                n = p.numel()
                rows = p.shape[0] if p.ndim >= 2 else 0
                cols = (n // rows) if rows > 0 else 0
                if prec == "ultra":
                    bpp = 1.5 + (2.0 if has_slow else 0.0)
                    total += int(n * bpp)
                elif prec == "8bit":
                    is_fact = p.ndim >= 2 and n >= 32768 and rows >= 64 and cols >= 64
                    if is_fact:
                        n_blocks = (n + 255) // 256
                        m_b = ((((n + 255) // 256) * 256 // 4) * 3 if n >= 524288 else n) + n_blocks * 4
                        v_b = (rows + cols) * 4
                        total += m_b + v_b + (n if has_slow else 0)
                    else:
                        bpp = 2.0 + (1.0 if has_slow else 0.0)
                        total += int(n * bpp)
                elif prec == "full":
                    bpp = 8.0 + (4.0 if has_slow else 0.0)
                    total += int(n * bpp)
                else:  # compact / auto
                    is_fact = p.ndim >= 2 and n >= 524288 and rows >= 128 and cols >= 128
                    m_bytes = 2.0 if p.dtype in (torch.float16, torch.bfloat16) else 4.0
                    if is_fact:
                        total += int(n * m_bytes) + (rows + cols) * 4 + (int(n * m_bytes) if has_slow else 0)
                    else:
                        v_bytes = 4.0 if (len(betas) >= 2 and betas[1] >= 0.99) else 2.0
                        bpp = m_bytes + v_bytes + (m_bytes if has_slow else 0.0)
                        total += int(n * bpp)
        return total

    def print_summary(self):
        """Prints a human-readable summary of parameter breakdown and real state memory usage."""
        muon_numel = sum(p.numel() for p in self.muon_params)
        adamw_numel = sum(p.numel() for p in self.adamw_params)
        total_numel = muon_numel + adamw_numel

        muon_alloc = self._allocated_state_bytes(self.muon_opt)
        adamw_alloc = self._allocated_state_bytes(self.adamw_opt)
        is_estimated = (muon_numel > 0 and muon_alloc == 0) or (adamw_numel > 0 and adamw_alloc == 0)

        muon_bytes = muon_alloc if muon_alloc > 0 else self._estimate_muon_bytes()
        adamw_bytes = adamw_alloc if adamw_alloc > 0 else self._estimate_adamw_bytes()

        muon_bpp = (muon_bytes / muon_numel) if muon_numel > 0 else 0.0
        adamw_bpp = (adamw_bytes / adamw_numel) if adamw_numel > 0 else 0.0

        muon_mb = muon_bytes / (1024 * 1024)
        adamw_mb = adamw_bytes / (1024 * 1024)
        total_mb = muon_mb + adamw_mb
        baseline_mb = (total_numel * 8) / (1024 * 1024)
        saved_mb = baseline_mb - total_mb
        saved_pct = (saved_mb / baseline_mb * 100.0) if baseline_mb > 0 else 0.0
        est_tag = " est." if is_estimated else ""

        print("\n" + "=" * 60)
        print(" ApexW Hybrid Optimizer Summary")
        print("=" * 60)
        print(f" Total Parameters   : {total_numel:,}")
        print(f"  - ApexMuon (2D)   : {muon_numel:,} ({muon_numel/max(1, total_numel)*100:.1f}%) [{muon_bpp:.1f} bytes/param{est_tag}]")
        print(f"  - ApexW (1D/Embed): {adamw_numel:,} ({adamw_numel/max(1, total_numel)*100:.1f}%) [{adamw_bpp:.1f} bytes/param{est_tag}]")
        print("-" * 60)
        print(f" State Memory (VRAM):")
        print(f"  - Hybrid Total    : {total_mb:.2f} MB")
        print(f"  - Standard AdamW  : {baseline_mb:.2f} MB")
        print(f"  - Memory Saved    : {saved_mb:.2f} MB ({saved_pct:.1f}% reduction)")
        print("=" * 60 + "\n")
