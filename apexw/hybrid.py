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

    def __init__(self, muon_opt, adamw_opt):
        super().__init__(dict)
        self._muon_opt = muon_opt
        self._adamw_opt = adamw_opt

    @property
    def muon_state(self):
        return self._muon_opt.state if self._muon_opt is not None else {}

    @property
    def adamw_state(self):
        return self._adamw_opt.state if self._adamw_opt is not None else {}

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
            is_embed = exclude_embeddings and any(k in name_lower for k in ("embed", "wte", "shared"))
            is_fsdp = is_fsdp_flat_param(p)

            if is_1d_or_bias or is_embed or is_fsdp:
                adamw_params.append(p)
            else:
                m = p.shape[0]
                n = p.numel() // m
                if min(m, n) >= min_muon_dim:
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
            if is_muon_group is True:
                muon_ps.append(p)
            elif is_muon_group is False:
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
        adamw_betas: Betas for AdamW components (default: (0.9, 0.99)).
        weight_decay: Decoupled weight decay (default: 1e-2).
        ns_steps: Number of Newton-Schulz iterations (default: 5; matches standalone Muon convergence).
        scale_mode: Matrix scaling method: 'none' (default, matches AdamWMuon 1:1), 'spectral', or 'aspect_ratio'.
        ns_method: Newton-Schulz algorithm: 'newtonschulz5' (default) or 'polar_express'.
        cautious: Enable cautious updates across both engines (default: False).
        precision: State storage precision: 'auto', 'compact', or 'full' (default: 'auto').
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
        adamw_betas: Tuple[float, float] = (0.9, 0.99),  # Synced with ApexW default for consistent 1D norm updates
        weight_decay: float = 1e-2,
        muon_weight_decay: Optional[float] = None,
        ns_steps: int = 3,  # 5 steps matches validated standalone Muon convergence
        scale_mode: str = "none",  # 'none' gives exact 1:1 update step scaling with user's lr
        ns_method: str = "newtonschulz5",
        beta3: Optional[float] = None,
        alpha: float = 5.0,
        cautious: bool = False,
        min_muon_dim: int = 32,
        exclude_embeddings: bool = True,
        precision: Union[str, torch.dtype] = "auto",
        batch_bmm: bool = False,
        nesterov: bool = False,
        fused: Optional[bool] = None,
        capturable: bool = False,
        compiled: bool = False,
        muon_compiled: Optional[bool] = None,
        **apex_kwargs,
    ):
        precision = _normalize_precision(precision)
        if muon_lr is None:
            muon_lr = lr
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

            # Build sub-optimizers with their correctly-scoped param groups.
            # The global muon_lr / weight_decay act as fallback defaults; each group
            # dict already carries its own overrides where the user specified them.
            self.muon_opt = ApexMuon(
                muon_groups,
                lr=muon_lr,
                momentum=muon_momentum,
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
            ) if self.muon_params else None

            self.adamw_opt = ApexW(
                adamw_groups,
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
            ) if self.adamw_params else None

        else:
            # Flat param list or nn.Module — original path.
            self.muon_params, self.adamw_params = partition_parameters_for_hybrid(
                params,
                min_muon_dim=min_muon_dim,
                exclude_embeddings=exclude_embeddings,
            )

            self.muon_opt = ApexMuon(
                self.muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
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
            ) if len(self.muon_params) > 0 else None

            self.adamw_opt = ApexW(
                self.adamw_params,
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
            ) if len(self.adamw_params) > 0 else None

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

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """Returns standard PyTorch-compatible optimizer state_dict."""
        sd = super().state_dict()
        sd["muon_state_dict"] = self.muon_opt.state_dict() if self.muon_opt else None
        sd["adamw_state_dict"] = self.adamw_opt.state_dict() if self.adamw_opt else None
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restores optimizer state, automatically handling cross-device and precision casting."""
        if self.muon_opt and state_dict.get("muon_state_dict"):
            self.muon_opt.load_state_dict(state_dict["muon_state_dict"])
        if self.adamw_opt and state_dict.get("adamw_state_dict"):
            self.adamw_opt.load_state_dict(state_dict["adamw_state_dict"])

        if not state_dict.get("muon_state_dict") and not state_dict.get("adamw_state_dict") and "state" in state_dict:
            try:
                super().load_state_dict(state_dict)
            except Exception:
                pass

        # Ensure all state tensors automatically match their target parameter device
        for p, p_state in self.state.items():
            if isinstance(p, Tensor):
                for k, v in list(p_state.items()):
                    if isinstance(v, Tensor) and v.device != p.device:
                        p_state[k] = v.to(p.device)

    def print_summary(self):
        """Prints a human-readable summary of parameter breakdown and memory savings."""
        muon_numel = sum(p.numel() for p in self.muon_params)
        adamw_numel = sum(p.numel() for p in self.adamw_params)
        total_numel = muon_numel + adamw_numel

        muon_mb = (muon_numel * 2) / (1024 * 1024)
        adamw_mb = (adamw_numel * 4) / (1024 * 1024)
        total_mb = muon_mb + adamw_mb
        baseline_mb = (total_numel * 8) / (1024 * 1024)
        saved_mb = baseline_mb - total_mb
        saved_pct = (saved_mb / baseline_mb * 100.0) if baseline_mb > 0 else 0.0

        print("\n" + "=" * 60)
        print(" ApexW Hybrid Optimizer Summary")
        print("=" * 60)
        print(f" Total Parameters   : {total_numel:,}")
        print(f"  - ApexMuon (2D)   : {muon_numel:,} ({muon_numel/max(1, total_numel)*100:.1f}%) [2 bytes/param]")
        print(f"  - ApexW (1D/Embed): {adamw_numel:,} ({adamw_numel/max(1, total_numel)*100:.1f}%) [4 bytes/param]")
        print("-" * 60)
        print(f" State Memory (VRAM):")
        print(f"  - Hybrid Total    : {total_mb:.2f} MB")
        print(f"  - Standard AdamW  : {baseline_mb:.2f} MB")
        print(f"  - Memory Saved    : {saved_mb:.2f} MB ({saved_pct:.1f}% reduction)")
        print("=" * 60 + "\n")
