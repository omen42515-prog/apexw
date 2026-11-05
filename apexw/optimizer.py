import math
import warnings
from collections import defaultdict
from collections.abc import MutableMapping
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable, Tuple, List, Dict, Any, Union

try:
    from .cuda_extension import (
        is_cuda_kernel_available,
        fused_adamw_cuda_step,
        fused_multi_tensor_adamw_cuda_step,
        step_fused,
    )
except ImportError:
    try:
        from cuda_extension import (
            is_cuda_kernel_available,
            fused_adamw_cuda_step,
            fused_multi_tensor_adamw_cuda_step,
            step_fused,
        )
    except ImportError:
        def is_cuda_kernel_available() -> bool:
            return False
        def fused_adamw_cuda_step(*args, **kwargs) -> bool:
            return False
        def fused_multi_tensor_adamw_cuda_step(*args, **kwargs) -> bool:
            return False
        def step_fused(*args, **kwargs) -> bool:
            return False

try:
    from .triton_kernel import is_triton_available, triton_multi_tensor_adamw_step, is_rocm
except ImportError:
    try:
        from triton_kernel import is_triton_available, triton_multi_tensor_adamw_step, is_rocm
    except ImportError:
        def is_triton_available() -> bool:
            return False
        def triton_multi_tensor_adamw_step(*args, **kwargs) -> bool:
            return False
        def is_rocm() -> bool:
            return False

try:
    from .xla_support import is_xla_available, is_xla_tensor, is_xla_device
except ImportError:
    try:
        from xla_support import is_xla_available, is_xla_tensor, is_xla_device
    except ImportError:
        def is_xla_available() -> bool:
            return False
        def is_xla_tensor(t) -> bool:
            return getattr(t, "device", None) is not None and t.device.type == "xla"
        def is_xla_device(d) -> bool:
            return False

try:
    from torch.compiler import is_compiling
except ImportError:
    try:
        from torch._dynamo import is_compiling
    except ImportError:
        def is_compiling():
            return False

CAPTURABLE_SUPPORTED_DEVICES = ("cuda", "xpu", "hpu", "privateuseone")


def _mps_supports_bfloat16() -> bool:
    """Checks whether Apple Silicon MPS backend supports bfloat16 on this host."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    if hasattr(_mps_supports_bfloat16, "_cached"):
        return _mps_supports_bfloat16._cached
    try:
        t = torch.zeros(1, dtype=torch.bfloat16, device="mps")
        _ = t + 1
        _mps_supports_bfloat16._cached = True
        return True
    except Exception:
        _mps_supports_bfloat16._cached = False
        return False


def _normalize_precision(precision: Union[str, torch.dtype]) -> str:
    """Normalize user-provided precision mode into 'auto', 'compact', 'ultra', or 'full'.

    Accepts:
    - Standard strings: 'auto', 'compact', 'ultra', 'full'
    - Precision aliases: 'fp32', 'float32', 'float', 'fp64', 'float64', 'double' -> 'full'
    - 16-bit aliases: 'bf16', 'bfloat16', 'fp16', 'float16', 'half', '16', '16-bit' -> 'compact'
    - 8-bit aliases: 'ultra', 'int8', '8bit', '8-bit' -> 'ultra' (block-wise INT8 m + BF16 v)
    - PyTorch dtypes: torch.float32, torch.float64 -> 'full'
    - PyTorch 16-bit dtypes: torch.bfloat16, torch.float16 -> 'compact'
    """
    if isinstance(precision, torch.dtype):
        if precision in (torch.float32, torch.float64):
            return "full"
        if precision in (torch.bfloat16, torch.float16):
            return "compact"
        if precision == torch.int8:
            return "ultra"
        return "full"
    if isinstance(precision, str):
        p_clean = precision.lower().strip()
        if p_clean in ("full", "fp32", "float32", "float", "fp64", "float64", "double"):
            return "full"
        if p_clean in ("compact", "bf16", "bfloat16", "fp16", "float16", "half", "16", "16-bit"):
            return "compact"
        if p_clean in ("ultra", "int8", "8bit", "8-bit"):
            return "ultra"
        if p_clean in ("auto", "smart", "adaptive"):
            return "auto"
    raise ValueError(
        f"Invalid precision mode '{precision}'. Must be 'auto', 'compact', 'ultra', 'full', "
        f"or a valid torch dtype (e.g. torch.float32, torch.bfloat16, torch.float16, torch.int8)."
    )


def _get_compact_dtype(p: Tensor) -> torch.dtype:
    """Determine optimal 16-bit storage dtype based on hardware and tensor device.

    Complex tensors and double-precision (float64) tensors must retain their
    native dtype to prevent numerical precision loss or PyTorch op incompatibilities.
    On Apple Silicon (MPS), checks if Metal supports bfloat16, falling back
    safely to float16 with dynamic scaling to prevent hardware errors.
    On CUDA/ROCm/CPU, uses bfloat16 for full dynamic range protection.
    """
    if torch.is_complex(p) or p.dtype == torch.float64:
        return p.dtype
    if p.dtype == torch.float16:
        return torch.float16
    if p.device.type == "mps":
        if _mps_supports_bfloat16():
            return torch.bfloat16
        return torch.float16
    if hasattr(torch, "bfloat16"):
        return torch.bfloat16
    return torch.float32


# ── Beta2 precision thresholds ───────────────────────────────────────────────
# The bf16 steady-state EMA update margin is (1 - beta2) * 128. When this
# falls below 1.0, the per-step gradient² contribution is smaller than the
# bf16 quantisation step of v_t, causing the EMA to silently stagnate.
#
# Threshold  Margin  Effect
# beta2<0.99  >1.28x  Negligible (LLM recipes at 0.95/0.98 are safe)
# 0.99..0.995  ~0.64x  Mild – occasional updates dropped; dynamic scaling helps
# ≥0.995      <0.64x  Severe – v_t effectively freezes; force v to fp32
#
# Many LLM recipes use beta2=0.98/0.99; default AdamW uses beta2=0.999 (severe risk).
_BETA2_WARN_THRESHOLD = 0.995  # Issue a warning for beta2 approaching fp32 threshold
_BETA2_V_FP32_THRESHOLD = 0.995  # Force second-moment state to fp32


def _get_v_dtype_for_beta2(compact_dtype: torch.dtype, beta2: float) -> torch.dtype:
    """Return the dtype to use for the second-moment (variance) buffer.

    When ``beta2 >= _BETA2_V_FP32_THRESHOLD``, the per-step EMA contribution
    ``(1 - beta2) * g²`` falls below the bf16/fp16 quantisation step of the
    accumulated ``v_t``.  This causes ``v_t`` to stagnate — the EMA stops
    tracking gradient statistics — inflating the effective learning rate and
    destabilising long training runs.

    The fix is to store ``v`` (``exp_avg_sq``) in ``float32`` while keeping
    ``m`` (``exp_avg``) in the compact dtype.  This costs 2 extra bytes per
    parameter (6 vs 4 total) but remains 25 % cheaper than full AdamW (8 bytes).
    """
    if compact_dtype in (torch.bfloat16, torch.float16) and beta2 >= _BETA2_V_FP32_THRESHOLD:
        return torch.float32
    return compact_dtype


def _get_scalar_dtype(is_fused: bool = False) -> torch.dtype:
    """Match PyTorch's step tensor dtype policy."""
    if is_fused:
        return torch.float32
    return torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32


def _get_value(x):
    """Extract Python scalar from a 0-dim tensor or return scalar as-is."""
    if isinstance(x, Tensor):
        return x.item()
    return x


class PackedStateProxy(MutableMapping):
    """Provides transparent, sliced access into the flat optimizer state for individual packed parameters."""

    def __init__(self, flat_state: dict, offset: int, numel: int, shape: torch.Size):
        self._flat_state = flat_state
        self._offset = offset
        self._numel = numel
        self._shape = shape

    def __getitem__(self, key):
        val = self._flat_state[key]
        if isinstance(val, Tensor) and val.numel() > 1:
            if val.dim() == 2 and val.shape[-1] == 1:
                return val[self._offset : self._offset + self._numel, 0].view(self._shape)
            return val[self._offset : self._offset + self._numel].view(self._shape)
        return val

    def __setitem__(self, key, value):
        self._flat_state[key] = value

    def __delitem__(self, key):
        del self._flat_state[key]

    def __iter__(self):
        return iter(self._flat_state)

    def __len__(self):
        return len(self._flat_state)

    def __contains__(self, key):
        return key in self._flat_state


class StateDictWrapper(defaultdict):
    """Wraps optimizer.state to transparently support state lookup for auto-packed parameters."""

    def __init__(self, opt, initial_state=None):
        super().__init__(dict)
        self._opt = opt
        if initial_state:
            self.update(initial_state)

    def __getitem__(self, p):
        if hasattr(self._opt, "_param_to_packed") and id(p) in self._opt._param_to_packed:
            flat_param, offset, numel, shape = self._opt._param_to_packed[id(p)]
            return PackedStateProxy(super().__getitem__(flat_param), offset, numel, shape)
        return super().__getitem__(p)

    def __contains__(self, p):
        if hasattr(self._opt, "_param_to_packed") and id(p) in self._opt._param_to_packed:
            flat_param, _, _, _ = self._opt._param_to_packed[id(p)]
            return flat_param in self
        return super().__contains__(p)


class FastExecutionPlan:
    """
    Cached execution plan for an optimizer parameter group.
    Bypasses per-step Python introspection, list building, and dictionary lookups,
    dropping step dispatch latency from ~0.35ms to < 0.03ms per step.
    """
    __slots__ = (
        "path_type",
        "params",
        "exp_avgs",
        "exp_avg_sqs",
        "state_steps",
        "exp_avgs_slow",
        "max_exp_avg_sqs",
        "decay_norm_only",
        "extra",
        "param_ids",
        "decay_params",
        "decay_m",
        "decay_v",
        "decay_max_v",
        "decay_steps",
        "no_decay_params",
        "no_decay_m",
        "no_decay_v",
        "no_decay_max_v",
        "no_decay_steps",
        # Triton path: pre-cached list of (scale_v, sqrt_scale_v, apply_wd) per param —
        # avoids a per-step dict comprehension over all params.
        "scales_v_list",
        # Triton decay flags: pre-computed apply_wd bool per param
        "apply_wd_list",
    )

    def __init__(
        self,
        path_type: str,
        params: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: Optional[List[Tensor]],
        state_steps: List[Tensor],
        exp_avgs_slow: Optional[List[Tensor]] = None,
        max_exp_avg_sqs: Optional[List[Tensor]] = None,
        decay_norm_only: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.path_type = path_type
        self.params = list(params)
        self.exp_avgs = list(exp_avgs)
        self.exp_avg_sqs = list(exp_avg_sqs) if exp_avg_sqs is not None else None
        self.state_steps = list(state_steps)
        self.exp_avgs_slow = list(exp_avgs_slow) if exp_avgs_slow is not None else None
        self.max_exp_avg_sqs = list(max_exp_avg_sqs) if max_exp_avg_sqs is not None else None
        self.decay_norm_only = decay_norm_only
        self.extra = extra or {}
        self.param_ids = tuple(id(p) for p in self.params)

        if path_type == "native_fused":
            decay_indices = [i for i, p in enumerate(self.params) if p.dim() >= 2 or not decay_norm_only]
            no_decay_indices = [i for i, p in enumerate(self.params) if p.dim() < 2 and decay_norm_only]

            self.decay_params = [self.params[i] for i in decay_indices]
            self.decay_m = [self.exp_avgs[i] for i in decay_indices]
            self.decay_v = [self.exp_avg_sqs[i] for i in decay_indices] if self.exp_avg_sqs else []
            self.decay_max_v = [self.max_exp_avg_sqs[i] for i in decay_indices] if self.max_exp_avg_sqs else []
            self.decay_steps = [self.state_steps[i] for i in decay_indices]

            self.no_decay_params = [self.params[i] for i in no_decay_indices]
            self.no_decay_m = [self.exp_avgs[i] for i in no_decay_indices]
            self.no_decay_v = [self.exp_avg_sqs[i] for i in no_decay_indices] if self.exp_avg_sqs else []
            self.no_decay_max_v = [self.max_exp_avg_sqs[i] for i in no_decay_indices] if self.max_exp_avg_sqs else []
            self.no_decay_steps = [self.state_steps[i] for i in no_decay_indices]
        else:
            self.decay_params = None
            self.decay_m = None
            self.decay_v = None
            self.decay_max_v = None
            self.decay_steps = None
            self.no_decay_params = None
            self.no_decay_m = None
            self.no_decay_v = None
            self.no_decay_max_v = None
            self.no_decay_steps = None

        # Triton path caches: populated lazily on first execute_triton() call
        self.scales_v_list = None   # List[float] — scale_v per param (avoids per-step dict build)
        self.apply_wd_list = None   # List[bool]  — apply_wd per param (avoids per-step .dim() check)

    def is_valid_for_step(self, group: dict) -> bool:
        if len(group["params"]) != len(self.params):
            return False
        for i, p in enumerate(self.params):
            if p.grad is None or id(p) != self.param_ids[i]:
                return False
        if group.get("amsgrad", False) != bool(self.max_exp_avg_sqs):
            return False
        if bool(group.get("beta3", None)) != bool(self.exp_avgs_slow):
            return False
        if group.get("clip_grad_norm") is not None or group.get("grad_centralization", False):
            return False
        # Guard against mid-training mode changes that alter the execution path.
        if self.path_type in ("native_fused", "cuda"):
            if group.get("nesterov", False) or group.get("lion", False) or group.get("cautious", False):
                return False
        elif self.path_type == "triton":
            if group.get("nesterov", False) or group.get("lion", False) or group.get("amsgrad", False):
                return False
        return True

    def execute(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        if self.path_type == "native_fused":
            self.execute_native_fused(group, grad_scale, found_inf)
        elif self.path_type == "triton":
            self.execute_triton(opt, group, grad_scale, found_inf)
        elif self.path_type == "cuda":
            self.execute_cuda(opt, group, grad_scale, found_inf)
        elif self.path_type == "foreach":
            self.execute_foreach(opt, group, grad_scale, found_inf)
        elif self.path_type == "single":
            self.execute_single(opt, group, grad_scale, found_inf)

    def execute_triton(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        step_val = int(_get_value(self.state_steps[0])) + 1
        torch._foreach_add_(self.state_steps, 1)

        # In-register gradient unscaling optimization:
        # If grad_scale is a CPU tensor (standard PyTorch GradScaler), reading float(grad_scale)
        # has ZERO GPU synchronization latency! Passing inv_scale to Triton tells the kernel
        # to unscale directly in SRAM registers, eliminating the extra _foreach_mul_ memory pass.
        # If grad_scale is on CUDA, unscale asynchronously via _foreach_mul_ to avoid CPU-GPU sync.
        grads = [p.grad for p in self.params]
        inv_scale = 1.0
        if grad_scale is not None:
            if not getattr(grad_scale, "is_cuda", False):
                inv_scale = 1.0 / float(grad_scale)
            else:
                is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
                if not is_capturing and not is_compiling():
                    torch._foreach_mul_(grads, grad_scale.reciprocal())

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        beta3 = group.get("beta3", None)
        alpha = group.get("alpha", 5.0)
        wd = group["weight_decay"]
        eps = group["eps"]
        cautious = group.get("cautious", False)

        dynamic_scaling = group.get("dynamic_scaling", True)
        has_16bit_states = bool(
            self.exp_avg_sqs and self.exp_avg_sqs[0].dtype in (torch.bfloat16, torch.float16)
        )

        # FIX 2: Cache scale_v and apply_wd as aligned lists in plan slots.
        # On warm steps, skip the per-step dict comprehension + dim() scan entirely.
        # Only rebuild on the rare 100-step dynamic-scaling recalibration boundary.
        needs_cache_rebuild = self.scales_v_list is None or self.apply_wd_list is None

        if dynamic_scaling and has_16bit_states and step_val > 1 and step_val % 100 == 0:
            # Rare recalibration: adjust scale_v across all v buffers, then force cache rebuild
            v_max = max(v.max().item() for v in self.exp_avg_sqs) if self.exp_avg_sqs else 1.0
            first_scale = opt.state[self.params[0]].get("scale_v", 1.0)
            factor = 1.0
            if v_max < 0.25 and first_scale < 65536.0:
                factor = 4.0
            elif v_max > 16384.0 and first_scale > 1.0:
                factor = 0.25
            if factor != 1.0:
                torch._foreach_mul_(self.exp_avg_sqs, factor)
                for p in self.params:
                    s = opt.state[p].get("scale_v", 1.0) * factor
                    opt.state[p]["scale_v"] = s
                    opt.state[p]["sqrt_scale_v"] = math.sqrt(s)
            needs_cache_rebuild = True

        if needs_cache_rebuild:
            # Build once and store aligned lists — no dicts, no per-step lookups
            if dynamic_scaling and has_16bit_states:
                self.scales_v_list = [
                    opt.state[p].get("scale_v", 1.0) if p in opt.state else 1.0
                    for p in self.params
                ]
            else:
                self.scales_v_list = None  # signals "all scale_v = 1.0" to the loop

            if wd != 0.0 and self.decay_norm_only:
                self.apply_wd_list = [p.dim() >= 2 for p in self.params]
            else:
                self.apply_wd_list = None  # signals "apply_wd same for all params"

        triton_multi_tensor_adamw_step(
            self.params,
            grads,
            self.exp_avgs,
            self.exp_avg_sqs,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=wd,
            step=step_val,
            decay_norm_only=self.decay_norm_only,
            scales_v=self.scales_v_list,
            inv_scale=inv_scale,
            exp_avgs_slow=self.exp_avgs_slow if beta3 is not None else None,
            beta3=beta3,
            alpha=alpha,
            cautious=cautious,
            skip_preflight=True,
        )

    def execute_cuda(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        step_val = int(_get_value(self.state_steps[0])) + 1
        torch._foreach_add_(self.state_steps, 1)

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        eps = group["eps"]

        grads = [p.grad for p in self.params]
        step_fused(
            self.params,
            grads,
            self.exp_avgs,
            self.exp_avg_sqs,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=wd,
            step=step_val,
            decay_norm_only=self.decay_norm_only,
        )

    def execute_single(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        grads = [p.grad for p in self.params]
        opt._single_tensor_step(
            self.params,
            grads,
            self.exp_avgs,
            self.exp_avgs_slow if self.exp_avgs_slow else [],
            self.exp_avg_sqs if self.exp_avg_sqs else [],
            self.max_exp_avg_sqs if self.max_exp_avg_sqs else [],
            self.state_steps,
            lr=group["lr"],
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            beta3=group.get("beta3", None),
            alpha=group.get("alpha", 5.0),
            lion=group.get("lion", False),
            eps=group["eps"],
            weight_decay=group["weight_decay"],
            decay_norm_only=self.decay_norm_only,
            nesterov=group["nesterov"],
            cautious=group["cautious"],
            amsgrad=group["amsgrad"],
            maximize=group["maximize"],
            capturable=group["capturable"],
            precision=group["precision"],
            dynamic_scaling=group.get("dynamic_scaling", True),
            has_complex=self.extra.get("has_complex", False),
            grad_scale=grad_scale,
            found_inf=found_inf,
        )

    def execute_native_fused(
        self,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        device = self.params[0].device
        device_grad_scale = (
            grad_scale.to(device, non_blocking=True) if grad_scale is not None else None
        )
        device_found_inf = (
            found_inf.to(device, non_blocking=True) if found_inf is not None else None
        )
        torch._foreach_add_(self.state_steps, 1)

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        eps = group["eps"]
        amsgrad = group["amsgrad"]
        maximize = group["maximize"]

        # Fast path: single fused kernel launch when norm protection is not needed or wd is 0
        if not self.decay_norm_only or wd == 0.0 or not self.no_decay_params:
            if self.params:
                torch._fused_adamw_(
                    self.params,
                    [p.grad for p in self.params],
                    self.exp_avgs,
                    self.exp_avg_sqs,
                    self.max_exp_avg_sqs if self.max_exp_avg_sqs else [],
                    self.state_steps,
                    amsgrad=amsgrad,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=wd,
                    eps=eps,
                    maximize=maximize,
                    grad_scale=device_grad_scale,
                    found_inf=device_found_inf,
                )
        else:
            if self.decay_params:
                torch._fused_adamw_(
                    self.decay_params,
                    [p.grad for p in self.decay_params],
                    self.decay_m,
                    self.decay_v,
                    self.decay_max_v,
                    self.decay_steps,
                    amsgrad=amsgrad,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=wd,
                    eps=eps,
                    maximize=maximize,
                    grad_scale=device_grad_scale,
                    found_inf=device_found_inf,
                )
            if self.no_decay_params:
                torch._fused_adamw_(
                    self.no_decay_params,
                    [p.grad for p in self.no_decay_params],
                    self.no_decay_m,
                    self.no_decay_v,
                    self.no_decay_max_v,
                    self.no_decay_steps,
                    amsgrad=amsgrad,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.0,
                    eps=eps,
                    maximize=maximize,
                    grad_scale=device_grad_scale,
                    found_inf=device_found_inf,
                )
        if device_found_inf is not None:
            torch._foreach_sub_(self.state_steps, [device_found_inf] * len(self.state_steps))

    def execute_foreach(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        grads = [p.grad for p in self.params]
        opt._multi_tensor_step(
            self.params,
            grads,
            self.exp_avgs,
            self.exp_avgs_slow if self.exp_avgs_slow else [],
            self.exp_avg_sqs if self.exp_avg_sqs else [],
            self.max_exp_avg_sqs if self.max_exp_avg_sqs else [],
            self.state_steps,
            lr=group["lr"],
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            beta3=group.get("beta3", None),
            alpha=group.get("alpha", 5.0),
            lion=group.get("lion", False),
            eps=group["eps"],
            weight_decay=group["weight_decay"],
            decay_norm_only=self.decay_norm_only,
            nesterov=group["nesterov"],
            cautious=group["cautious"],
            amsgrad=group["amsgrad"],
            maximize=group["maximize"],
            capturable=group["capturable"],
            precision=group["precision"],
            dynamic_scaling=group.get("dynamic_scaling", True),
            has_complex=self.extra.get("has_complex", False),
            grad_scale=grad_scale,
            found_inf=found_inf,
        )


def _chunk_tensor_lists(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avgs_slow: Optional[List[Tensor]],
    exp_avg_sqs: Optional[List[Tensor]],
    max_exp_avg_sqs: Optional[List[Tensor]],
    step_sizes: Optional[List[Tensor]],
    bc2_sqrts: Optional[List[Tensor]],
    max_chunk_elements: int = 2_000_000,
) -> List[Tuple]:
    """Partitions multi-tensor lists into memory-bounded chunks to cap peak transient allocations."""
    total_elements = sum(p.numel() for p in params)
    if len(params) <= 1 or total_elements <= max_chunk_elements:
        return [(
            params, grads, exp_avgs, exp_avgs_slow,
            exp_avg_sqs, max_exp_avg_sqs, step_sizes, bc2_sqrts,
        )]

    chunks = []
    cur_p, cur_g, cur_m = [], [], []
    cur_m_slow = [] if exp_avgs_slow is not None else None
    cur_v = [] if exp_avg_sqs is not None else None
    cur_max_v = [] if max_exp_avg_sqs is not None else None
    cur_step_size = [] if step_sizes is not None else None
    cur_bc2_sqrt = [] if bc2_sqrts is not None else None
    cur_count = 0

    for j, p in enumerate(params):
        p_numel = p.numel()
        if cur_count > 0 and (cur_count + p_numel > max_chunk_elements):
            chunks.append((cur_p, cur_g, cur_m, cur_m_slow, cur_v, cur_max_v, cur_step_size, cur_bc2_sqrt))
            cur_p, cur_g, cur_m = [], [], []
            if exp_avgs_slow is not None:
                cur_m_slow = []
            if exp_avg_sqs is not None:
                cur_v = []
            if max_exp_avg_sqs is not None:
                cur_max_v = []
            if step_sizes is not None:
                cur_step_size = []
                cur_bc2_sqrt = []
            cur_count = 0

        cur_p.append(p)
        cur_g.append(grads[j])
        cur_m.append(exp_avgs[j])
        if exp_avgs_slow is not None:
            cur_m_slow.append(exp_avgs_slow[j])
        if exp_avg_sqs is not None:
            cur_v.append(exp_avg_sqs[j])
        if max_exp_avg_sqs is not None:
            cur_max_v.append(max_exp_avg_sqs[j])
        if step_sizes is not None:
            cur_step_size.append(step_sizes[j])
            cur_bc2_sqrt.append(bc2_sqrts[j])
        cur_count += p_numel

    if cur_p:
        chunks.append((cur_p, cur_g, cur_m, cur_m_slow, cur_v, cur_max_v, cur_step_size, cur_bc2_sqrt))
    return chunks


class ApexW(Optimizer):
    r"""Implements the ApexW optimizer: a memory-efficient, fused variant of AdamW.

    ApexW is designed as an efficient drop-in replacement for :class:`torch.optim.AdamW`.
    By storing first and second moment statistics ($m$ and $v$) in compact 16-bit
    floating-point representation (`bfloat16` or `float16`) with dynamic range scaling
    and stochastic rounding, ApexW reduces optimizer state memory by ~50% (4 bytes per
    parameter instead of 8 bytes) while maintaining numerical fidelity.

    Features:
    - **AdEMAMix** (Pagliardini et al., 2024): Optional dual-speed first-order momentum
      tracking fast and slow gradient averages via an additional ``beta3`` parameter.
    - **Cautious Updates** (Liang et al., 2024): Optional directional alignment mask that
      zeroes out step components when the current gradient opposes historical momentum.
    - **Gradient Centralization** (Yong et al., ECCV 2020): Zero-mean gradient projection
      along fan-in dimensions of weight matrices to smooth optimization landscapes.
    - **Smart Weight Decay**: Automatically excludes 1D tensors (e.g., LayerNorm/RMSNorm
      weights, biases) from weight decay when ``decay_norm_only=True``.
    - **Fused & Multi-Tensor Kernels**: Vectorized C++ / CUDA kernels and PyTorch foreach
      dispatch with bounded chunking to minimize kernel launch and allocation overheads.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float] or Tuple[float, float, float], optional):
            Coefficients used for computing running averages of gradient and its
            square (default: (0.9, 0.99)). The default ``beta2=0.99`` is tuned for
            optimal convergence quality matching standard AdamW baselines while maintaining
            a safe 1.28× bf16 EMA headroom, preserving 50% state VRAM savings under 16-bit states.
            For exact PyTorch AdamW equivalence use adamw_compat=True or betas=(0.9, 0.999).
            If a 3-tuple is provided, the 3rd element is used as ``beta3``
            for AdEMAMix.
        eps (float, optional): Term added to the denominator to improve numerical
            stability (default: 1e-8).
        weight_decay (float, optional): Decoupled weight decay coefficient (default: 1e-2).
        amsgrad (bool, optional): Whether to use the AMSGrad variant (default: False).
        nesterov (bool, optional): Whether to use Nesterov momentum (default: False).
        cautious (bool, optional): Whether to apply cautious gradient masking (default: False).
        grad_centralization (bool, optional): Whether to apply gradient centralization to
            weight matrices (default: False).
        decay_norm_only (bool, optional): If True, applies weight decay only to 2D+ tensors,
            protecting 1D normalization weights and biases (default: True).
        adamw_compat (bool, optional): If ``True``, overrides ``decay_norm_only=False``, applying
            weight decay to all parameters including 1D norms and biases. Use this for exact
            numerical equivalence with ``torch.optim.AdamW`` (e.g. benchmarking). Default: ``False``.
        clip_grad_norm (float, optional): Maximum gradient L2 norm for in-optimizer clipping
            (default: None).
        beta3 (float, optional): Slow momentum coefficient for AdEMAMix (default: None).
        alpha (float, optional): Scaling weight for slow momentum in AdEMAMix (default: 5.0).
        lion (bool, optional): Whether to use sign-based Lion momentum updates (default: False).
        fused (bool, optional): Whether to use PyTorch's native fused C++ kernel when running
            in full FP32 mode on CUDA (default: None).
        precision (str, optional): Optimizer state storage precision. Options:
            ``"auto"`` (compact 16-bit for 2D+ or >= threshold tensors, full FP32 for 1D),
            ``"compact"`` (all 16-bit), or ``"full"`` (all 32-bit). Default: ``"auto"``.
        threshold (int, optional): Parameter element threshold for ``precision="auto"``
            (default: 65536).
        maximize (bool, optional): Maximize the objective with respect to the params, instead
            of minimizing (default: False).
        foreach (bool, optional): Whether foreach multi-tensor implementation of optimizer
            is used. If None, automatically determined based on device (default: None).
        capturable (bool, optional): Whether this instance is safe to capture in a CUDA graph
            (default: False).
        dynamic_scaling (bool, optional): Whether to apply dynamic range scaling to 16-bit
            variance states to prevent underflow during extended training (default: True).
        differentiable (bool, optional): Whether autograd should occur through the optimizer step.
            Not currently supported in ApexW (default: False; passing True raises NotImplementedError).
        backend (str, optional): Target execution engine for hardware acceleration:
            - ``"auto"`` or ``None``: Automatically selects the best hardware backend:
              * Google TPUs: global PyTorch XLA graph compilation and ASIC kernel fusion
              * NVIDIA CUDA & AMD ROCm: OpenAI Triton fused GPU kernel or custom CUDA extension
              * Apple Silicon (MPS): native vectorized Multi-Tensor (_multi_tensor) engine
              * CPU: PyTorch native multi-tensor foreach
            - ``"triton"``: Forces OpenAI Triton fused GPU kernel (CUDA & ROCm)
            - ``"cuda"``: Forces custom C++/CUDA kernel
            - ``"mps"`` / ``"foreach"``: Forces native vectorized multi-tensor engine
            - ``"xla"``: Forces PyTorch XLA graph compilation path for Google TPUs
    """

    _step_supports_amp_scaling = True

    def __new__(cls, *args, **kwargs):
        first_arg = args[0] if len(args) > 0 else kwargs.get("params", None)
        is_model = isinstance(first_arg, nn.Module)
        use_muon = kwargs.get("use_muon", False)

        if use_muon:
            from .hybrid import HybridApexW
            clean_kwargs = dict(kwargs)
            clean_kwargs.pop("use_muon", None)
            return HybridApexW(*args, **clean_kwargs)

        if is_model:
            args = (first_arg.parameters(),) + args[1:]
            if "params" in kwargs:
                kwargs["params"] = first_arg.parameters()

        return super().__new__(cls)

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Union[Tuple[float, float], Tuple[float, float, float]] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        nesterov: bool = False,
        cautious: bool = False,
        grad_centralization: bool = False,
        decay_norm_only: bool = True,   # Automatically protect 1D norms (RMSNorm, LayerNorm) & biases
        adamw_compat: bool = False,  # If True, matches vanilla AdamW: decays all params (overrides decay_norm_only)
        clip_grad_norm: Optional[float] = None,
        beta3: Optional[float] = None,
        alpha: float = 5.0,
        lion: bool = False,
        fused: Optional[bool] = None,
        precision: Union[str, torch.dtype] = "auto",  # "auto", "compact" (bf16), "full" (fp32), or torch dtype
        threshold: int = 65536,      # Size threshold: 2D+ or >= 64K tensors get compact BF16, 1D norms get FP32
        maximize: bool = False,
        foreach: Optional[bool] = None,  # None = auto-detect: True on CUDA/XPU/ROCm, fast single on CPU
        capturable: bool = False,
        dynamic_scaling: bool = True, # Layer-wise dynamic range scaling to protect precision during long runs
        differentiable: bool = False,
        backend: Optional[str] = None, # None / "auto", "triton", "cuda", "mps", "foreach", "xla"
        auto_pack: bool = False,       # Automatically pack small tensors into contiguous flat buffers
        pack_threshold: int = 50000,   # Maximum numel for tensors to be auto-packed
        use_muon: bool = False,        # If True, routes 2D hidden weights to ApexMuon and 1D/embed to ApexW
    ):
        precision = _normalize_precision(precision)
        if isinstance(params, nn.Module):
            params = params.parameters()
        if differentiable:
            raise NotImplementedError("ApexW does not currently support differentiable=True.")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if len(betas) > 2 and beta3 is None:
            beta3 = betas[2]
        if beta3 is not None and not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3 parameter: {beta3}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if clip_grad_norm is not None and clip_grad_norm <= 0.0:
            raise ValueError(f"Invalid clip_grad_norm value: {clip_grad_norm}")
        if backend is not None and backend not in ("auto", "triton", "cuda", "mps", "foreach", "_multi_tensor", "xla"):
            raise ValueError(f"Invalid backend '{backend}'. Must be 'auto', 'triton', 'cuda', 'mps', 'foreach', or 'xla'.")

        # adamw_compat overrides decay_norm_only to False and defaults betas to (0.9, 0.999)
        # for exact bit-for-bit hyperparameter and numerical parity with PyTorch's native AdamW baseline.
        if adamw_compat:
            decay_norm_only = False
            if betas in ((0.9, 0.95), (0.9, 0.99)):
                betas = (0.9, 0.999)

        # ── Beta2 × compact precision compatibility warning ────────────────────
        # At beta2 ≥ 0.999 (default AdamW), (1 - beta2) * 128 ≈ 0.13 — the
        # per-step gradient² contribution falls 8× below bf16 resolution,
        # causing v_t to stagnate after warmup.  ApexW auto-escalates v to fp32
        # when beta2 ≥ _BETA2_V_FP32_THRESHOLD (0.995), which costs 2 extra bytes
        # per parameter but corrects the numerical behaviour.
        # At 0.99 ≤ beta2 < 0.995 the effect is mild; a warning is issued.
        _beta2_val = betas[1]
        if not adamw_compat and precision in ("auto", "compact") and _beta2_val >= _BETA2_WARN_THRESHOLD:
            _margin = (1.0 - _beta2_val) * 128  # bf16 EMA headroom (mantissa bits)
            if _beta2_val >= _BETA2_V_FP32_THRESHOLD:
                # Severe: auto-escalate v to fp32 (handled in _init_group)
                warnings.warn(
                    f"ApexW: beta2={_beta2_val} with precision='{precision}' causes severe "
                    f"bf16 second-moment stagnation (bf16 margin ≈ {_margin:.2f}×). "
                    f"exp_avg_sq (v) is automatically stored in float32 to preserve "
                    f"numerical correctness (exp_avg_sq auto-promoted to float32; use betas=(0.9, 0.95) to keep 50% VRAM savings). "
                    f"Or use precision='full' to silence this with all-fp32 states.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # Mild (0.99 ≤ beta2 < 0.995): warn but don't escalate automatically
                warnings.warn(
                    f"ApexW: beta2={_beta2_val} with precision='{precision}' may cause mild "
                    f"bf16 second-moment precision loss (bf16 margin ≈ {_margin:.2f}×) "
                    f"on long training runs. "
                    f"Quick fix: use betas=(0.9, 0.95) for zero warnings + maximum performance. "
                    f"Dynamic range scaling will partially compensate at this setting.",
                    UserWarning,
                    stacklevel=2,
                )

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
            fused=fused,
            precision=precision,
            threshold=threshold,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            dynamic_scaling=dynamic_scaling,
            differentiable=differentiable,
            backend=backend,
            auto_pack=auto_pack,
            pack_threshold=pack_threshold,
        )
        super().__init__(params, defaults)
        self._workspace_pool = {}
        self._execution_plans: Dict[int, FastExecutionPlan] = {}
        self.auto_pack = auto_pack
        self.pack_threshold = pack_threshold
        self.packed_param_groups: List[Dict[str, Any]] = []
        self._param_to_packed: Dict[int, Tuple] = {}
        if self.auto_pack:
            self._setup_auto_packing()

        # Warn if pure FP32 parameters on CUDA are detected in auto precision mode
        if precision == "auto" and any(
            isinstance(p, torch.Tensor) and p.is_cuda and p.dtype == torch.float32
            for group in self.param_groups
            for p in group["params"]
        ):
            warnings.warn(
                "ApexW detected FP32 parameters on CUDA. With precision='auto' (default), "
                "optimizer states use float32 for native fused C++ kernel speed — matching "
                "AdamW(fused=True) throughput. Use 16-bit Mixed Precision (AMP) training "
                "to halve optimizer state VRAM. In pure FP32 CUDA training you can use "
                "precision='compact' to force bf16 states and cut VRAM in exchange for some "
                "step-time overhead.",
                UserWarning,
                stacklevel=2,
            )

    def _setup_auto_packing(self):
        """
        Inspects parameter groups, partitions small parameters by (device, dtype, is_1d),
        packs each subgroup into a single contiguous flat buffer, and repoints p.data so
        layer attributes update in-place with zero copy while preserving weight decay semantics.
        """
        self.state = StateDictWrapper(self, self.state)

        for group in self.param_groups:
            buckets = defaultdict(list)
            large_params = []

            for p in group["params"]:
                if p.requires_grad and p.numel() < self.pack_threshold:
                    is_1d = (p.dim() < 2)
                    buckets[(p.device, p.dtype, is_1d)].append(p)
                else:
                    large_params.append(p)

            new_group_params = list(large_params)

            for (dev, dt, is_1d), small_params in buckets.items():
                if len(small_params) > 1:
                    if is_1d:
                        flat_tensor = nn.Parameter(
                            torch.cat([p.detach().contiguous().view(-1) for p in small_params]),
                            requires_grad=True,
                        )
                        offset = 0
                        for p in small_params:
                            numel = p.numel()
                            p.data = flat_tensor.data[offset : offset + numel].view_as(p)
                            self._param_to_packed[id(p)] = (flat_tensor, offset, numel, p.shape)
                            offset += numel
                    else:
                        cat_flat = torch.cat([p.detach().contiguous().view(-1) for p in small_params]).unsqueeze(-1)
                        flat_tensor = nn.Parameter(cat_flat, requires_grad=True)
                        offset = 0
                        for p in small_params:
                            numel = p.numel()
                            p.data = flat_tensor.data[offset : offset + numel, 0].view_as(p)
                            self._param_to_packed[id(p)] = (flat_tensor, offset, numel, p.shape)
                            offset += numel

                    self.packed_param_groups.append({
                        "flat_param": flat_tensor,
                        "individual_params": small_params,
                        "is_1d": is_1d,
                    })
                    new_group_params.append(flat_tensor)
                else:
                    new_group_params.extend(small_params)

            group["params"] = new_group_params

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        if getattr(self, "auto_pack", False):
            for packed in self.packed_param_groups:
                for p in packed["individual_params"]:
                    if set_to_none:
                        p.grad = None
                    elif p.grad is not None:
                        p.grad.zero_()

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("nesterov", False)
            group.setdefault("cautious", False)
            group.setdefault("grad_centralization", False)
            group.setdefault("decay_norm_only", True)
            group.setdefault("dynamic_scaling", True)
            group.setdefault("clip_grad_norm", None)
            group.setdefault("beta3", None)
            group.setdefault("alpha", 5.0)
            group.setdefault("lion", False)
            group.setdefault("fused", None)
            group.setdefault("precision", "auto")
            group.setdefault("threshold", 65536)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("backend", None)
        self._execution_plans = {}

    def get_state_memory_mb(self) -> float:
        """Returns total memory allocated for optimizer states in MB."""
        total_bytes = 0
        for state in self.state.values():
            for v in state.values():
                if torch.is_tensor(v) and v.ndim >= 1:
                    total_bytes += v.nelement() * v.element_size()
        return total_bytes / (1024 * 1024)

    def get_memory_summary(self) -> str:
        """Returns a formatted comparison of optimizer state memory vs standard AdamW."""
        actual_mb = self.get_state_memory_mb()
        param_count = sum(p.numel() for group in self.param_groups for p in group["params"])
        adamw_mb = (param_count * 8) / (1024 * 1024)
        saved_mb = adamw_mb - actual_mb
        pct = (saved_mb / adamw_mb * 100.0) if adamw_mb > 0 else 50.0

        return (
            f"Optimizer State Memory:\n"
            f"  - ApexW Current State : {actual_mb:.2f} MB\n"
            f"  - PyTorch AdamW Equiv : {adamw_mb:.2f} MB\n"
            f"  - Memory Reduction    : {pct:.1f}% ({saved_mb:.2f} MB saved)"
        )

    def print_summary(self) -> None:
        """Prints a human-friendly summary of optimizer state memory savings."""
        print(self.get_memory_summary())

    def __repr__(self) -> str:
        param_count = sum(p.numel() for group in self.param_groups for p in group["params"])
        state_mb = self.get_state_memory_mb()
        adamw_mb = (param_count * 8) / (1024 * 1024)
        saved_pct = ((adamw_mb - state_mb) / max(adamw_mb, 1e-6)) * 100.0 if state_mb > 0 else 50.0
        g0 = self.param_groups[0]
        return (
            f"ApexW("
            f"lr={g0['lr']}, "
            f"betas={g0['betas']}, "
            f"weight_decay={g0['weight_decay']}, "
            f"precision='{g0['precision']}', "
            f"params={param_count:,}, "
            f"state_memory={state_mb:.2f}MB, "
            f"vram_saved={saved_pct:.1f}%)"
        )

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dict, preserving dynamic scaling metadata."""
        sd = super().state_dict()
        scaling_meta = {}
        for p, s in self.state.items():
            if "scale_v" in s:
                scaling_meta[id(p)] = {
                    "scale_v": s["scale_v"],
                    "sqrt_scale_v": s.get("sqrt_scale_v", math.sqrt(s["scale_v"])),
                }
        sd["_dynamic_scaling_meta"] = scaling_meta
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the optimizer state, restoring dynamic scaling factors and workspace metadata."""
        scaling_meta = state_dict.pop("_dynamic_scaling_meta", {}) if "_dynamic_scaling_meta" in state_dict else {}
        super().load_state_dict(state_dict)
        self._workspace_pool = {}
        self._execution_plans = {}
        for p, s in self.state.items():
            if "scale_v" in s:
                dev = p.device
                key = (dev, p.dtype, s.get("state_dtype", p.dtype))
                scale_info = {
                    "scale_v": float(s["scale_v"]),
                    "sqrt_scale_v": float(s.get("sqrt_scale_v", math.sqrt(s["scale_v"]))),
                }
                self._workspace_pool[key] = scale_info
                self._workspace_pool[id(p)] = scale_info

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avgs_slow,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        """Initialize parameter group and lazy-init optimizer state buffers."""
        has_complex = False
        has_slow_momentum = group["beta3"] is not None
        is_lion = group["lion"]

        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                if p.grad.is_sparse:
                    raise RuntimeError("ApexW does not support sparse gradients")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(),
                            device=p.device,
                        )
                        if (group["capturable"] or p.device.type == "xla" or group.get("backend") == "xla")
                        else torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
                    )

                    threshold = group.get("threshold", 65536)
                    beta2 = group["betas"][1]

                    group["precision"] = _normalize_precision(group.get("precision", "auto"))
                    if group.get("fused", None) is True or group["precision"] == "full":
                        compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group["precision"] == "auto":
                        # Smart Auto Partitioning:
                        # Pure FP32 CUDA context (not in AMP) → use FP32 states so the native
                        # C++ fused kernel (_fused_adamw_) is used automatically — same speed as
                        # AdamW(fused=True) but with ApexW's extra features.
                        # AMP context (autocast active or mixed-dtype params) → use compact bf16
                        # states for the 50% VRAM savings they were designed to deliver.
                        _is_cuda_param = p.is_cuda
                        _is_amp_active = (
                            torch.is_autocast_enabled()
                            or (hasattr(torch, "is_autocast_enabled") and torch.is_autocast_enabled("cuda"))
                            or p.dtype in (torch.float16, torch.bfloat16)
                        )
                        if _is_cuda_param and not _is_amp_active and p.dtype == torch.float32:
                            # Pure FP32 CUDA → FP32 states → native fused path (fastest)
                            compact_dtype = torch.float32
                        elif p.dtype == torch.float64:
                            # Double precision parameters always use float64 states
                            compact_dtype = torch.float64
                        else:
                            # Massive 2D+ tensors or tensors >= threshold (>99.8% of model parameters)
                            # receive compact bfloat16 state storage, cutting state VRAM in half (50% cut).
                            # Small 1D tensors (LayerNorm, RMSNorm, BatchNorm, biases < threshold)
                            # receive full precision (float32) for zero precision drift and fast native foreach.
                            is_massive = (p.dim() >= 2) or (p.numel() >= threshold)
                            if is_massive:
                                compact_dtype = _get_compact_dtype(p)
                            else:
                                compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group["precision"] == "compact":
                        compact_dtype = _get_compact_dtype(p)
                    else:  # "full"
                        compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32

                    # First-moment (m): always uses compact_dtype.
                    state_dtype_m = compact_dtype
                    # Second-moment (v): escalate to fp32 when beta2 is high enough
                    # to cause bf16 EMA stagnation (see _get_v_dtype_for_beta2).
                    # At beta2 >= 0.995, (1-beta2)*128 < 0.64x — updates are lost.
                    # Storing v in fp32 costs 2 extra bytes/param but keeps v tracking.
                    state_dtype_v = _get_v_dtype_for_beta2(compact_dtype, beta2)

                    # Record the per-moment dtypes for downstream code to use

                    state["state_dtype"] = state_dtype_m        # primary (backward-compat key)
                    state["state_dtype_v"] = state_dtype_v      # second-moment dtype

                    # Primary first-moment buffer
                    state["exp_avg"] = torch.zeros_like(
                        p, dtype=state_dtype_m, memory_format=torch.preserve_format
                    )

                    # AdEMAMix slow moment buffer (slow momentum, also first-order)
                    if has_slow_momentum:
                        state["exp_avg_slow"] = torch.zeros_like(
                            p, dtype=state_dtype_m, memory_format=torch.preserve_format
                        )

                    # Second moment buffer (variance EMA, not needed in Lion mode)
                    if not is_lion:
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=state_dtype_v, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, dtype=state_dtype_v, memory_format=torch.preserve_format
                            )

                    # Initialize dynamic scaling metadata immediately (non-XLA devices only)
                    # When states are pure FP32, bypass dynamic scaling entirely (scale_v=1.0)
                    _is_xla = (p.device.type == "xla" or group.get("backend") == "xla")
                    if not _is_xla:
                        if group.get("dynamic_scaling", True) and not is_lion and state_dtype_v in (torch.bfloat16, torch.float16):
                            init_scale = 1024.0 if state_dtype_v == torch.float16 else 1.0
                            state["scale_v"] = init_scale
                            state["sqrt_scale_v"] = math.sqrt(init_scale)
                        else:
                            state["scale_v"] = 1.0
                            state["sqrt_scale_v"] = 1.0

                # Defensive initialization if amsgrad or AdEMAMix was enabled after initial step
                state_dtype = state.get("state_dtype", p.dtype)
                state_dtype_v = state.get("state_dtype_v", state_dtype)
                if has_slow_momentum and "exp_avg_slow" not in state:
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, dtype=state_dtype, memory_format=torch.preserve_format
                    )
                if not is_lion and group["amsgrad"] and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, dtype=state_dtype_v, memory_format=torch.preserve_format
                    )
                _is_xla = (p.device.type == "xla" or group.get("backend") == "xla")
                if not _is_xla and "scale_v" not in state:
                    if group.get("dynamic_scaling", True) and not is_lion and state_dtype_v in (torch.bfloat16, torch.float16):
                        init_scale = 1024.0 if state_dtype_v == torch.float16 else 1.0
                        state["scale_v"] = init_scale
                        state["sqrt_scale_v"] = math.sqrt(init_scale)
                    else:
                        state["scale_v"] = 1.0
                        state["sqrt_scale_v"] = 1.0

                exp_avgs.append(state["exp_avg"])
                if has_slow_momentum:
                    exp_avgs_slow.append(state["exp_avg_slow"])
                if not is_lion:
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                if group.get("differentiable", False):
                    raise NotImplementedError("ApexW does not currently support differentiable=True.")

                state_steps.append(state["step"])

        return has_complex

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        if hasattr(self, "_accelerator_graph_capture_health_check"):
            self._accelerator_graph_capture_health_check()
        elif hasattr(self, "_cuda_graph_capture_health_check"):
            self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Synchronize individual parameter gradients into flat gradient buffers for auto-packed groups.
        # A packed group where only a subset of parameters received gradients cannot be processed
        # correctly with zero-fill, as zero gradients would corrupt momentum statistics and apply
        # spurious weight decay to skipped layers. Ensure all parameters in the group are present.
        if getattr(self, "auto_pack", False):
            for packed in self.packed_param_groups:
                flat_param = packed["flat_param"]
                individual_params = packed["individual_params"]
                is_1d = packed["is_1d"]

                grad_present = [p.grad is not None for p in individual_params]
                if any(grad_present):
                    if not all(grad_present):
                        missing = [
                            i for i, present in enumerate(grad_present) if not present
                        ]
                        raise RuntimeError(
                            f"ApexW auto_pack: packed group has {len(missing)} parameter(s) "
                            f"with grad=None (indices {missing}) while other parameters in the "
                            f"same packed buffer have gradients. Substituting zero gradients "
                            f"would silently apply momentum decay and weight decay to those "
                            f"parameters, corrupting training. "
                            f"Fix: set auto_pack=False, or ensure all parameters in a packed "
                            f"group always produce gradients together (e.g. use the same "
                            f"requires_grad mask, or exclude sparse/conditionally-used params "
                            f"with pack_threshold=0 for those parameter groups)."
                        )
                    grads = [p.grad.detach().contiguous().view(-1) for p in individual_params]
                    cat_grad = torch.cat(grads)
                    flat_param.grad = cat_grad if is_1d else cat_grad.unsqueeze(-1)
                else:
                    flat_param.grad = None

        for group in self.param_groups:
            group_id = id(group)
            plan = self._execution_plans.get(group_id, None)
            if plan is not None and plan.is_valid_for_step(group):
                grad_scale = getattr(self, "grad_scale", None)
                found_inf = getattr(self, "found_inf", None)

                compiling = is_compiling()
                is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
                if found_inf is not None and not is_capturing and not compiling:
                    if found_inf.item() > 0:
                        continue
                plan.execute(self, group, grad_scale, found_inf)
                continue

            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avgs_slow: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avgs_slow,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            if not params_with_grad:
                continue

            # Retrieve AMP grad_scale and found_inf if set by GradScaler
            grad_scale = getattr(self, "grad_scale", None)
            found_inf = getattr(self, "found_inf", None)

            compiling = is_compiling()
            is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()

            # Early AMP skip if inf/NaN detected by GradScaler in eager execution.
            # Even when capturable=True, if not currently inside an active CUDA graph capture,
            # we must immediately skip to prevent corrupting weights or momentum buffers.
            if found_inf is not None and not is_capturing and not compiling:
                if found_inf.item() > 0:
                    continue

            # In-optimizer gradient unscaling before clipping and centralization.
            # PyTorch GradScaler leaves gradients scaled when _step_supports_amp_scaling=True.
            # If clip_grad_norm or grad_centralization is requested, gradients must be unscaled
            # FIRST so clipping calculates the true Euclidean norm instead of the scaled norm.
            clip_norm = group["clip_grad_norm"]
            backend_grad_scale = grad_scale
            if grad_scale is not None and (clip_norm is not None or group.get("grad_centralization", False)):
                first_dev = grads[0].device
                scale_t = grad_scale.to(first_dev) if grad_scale.device != first_dev else grad_scale
                inv_scale_t = scale_t.reciprocal()
                torch._foreach_mul_(grads, inv_scale_t)
                backend_grad_scale = None  # Gradients are already unscaled; do not unscale again

            # In-optimizer gradient centralization (ECCV 2020)
            if group.get("grad_centralization", False):
                for g in grads:
                    if g.dim() >= 2:
                        dims = tuple(range(1, g.dim()))
                        g.sub_(g.mean(dim=dims, keepdim=True))

            # In-optimizer fused gradient norm clipping on true (unscaled) gradients
            if clip_norm is not None:
                norms = torch._foreach_norm(grads, 2.0)
                first_dev = grads[0].device
                total_norm = torch.linalg.vector_norm(
                    torch.stack([n.to(first_dev) for n in norms]), 2.0
                )
                clip_coef = clip_norm / (total_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
                torch._foreach_mul_(grads, clip_coef_clamped)

            beta1, beta2 = group["betas"]
            beta3 = group.get("beta3", None)
            alpha = group.get("alpha", 5.0)
            lion = group.get("lion", False)
            lr = group["lr"]
            wd = group["weight_decay"]
            decay_norm_only = group.get("decay_norm_only", False)
            eps = group["eps"]
            nesterov = group["nesterov"]
            cautious = group["cautious"]
            maximize = group["maximize"]
            amsgrad = group["amsgrad"]
            capturable = group["capturable"]
            foreach = group["foreach"]
            precision = group["precision"]
            fused = group.get("fused", False)

            backend = group.get("backend", None)
            is_xla = (params_with_grad[0].device.type == "xla")
            is_mps = (params_with_grad[0].device.type == "mps")
            is_cuda = params_with_grad[0].is_cuda
            is_accelerator = params_with_grad[0].device.type != "cpu"
            use_foreach = (foreach is True) or (foreach is None)

            # -------------------------------------------------------------
            # PATH 1: Google TPU Global Graph Compilation (PyTorch XLA)
            # -------------------------------------------------------------
            # Bypasses all CPU-device syncs and compiles the optimizer step
            # into a single fused TPU ASIC kernel using XLA HLO compiler.
            if is_xla or backend == "xla":
                self._xla_step(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avgs_slow,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    beta3=beta3,
                    alpha=alpha,
                    lion=lion,
                    eps=eps,
                    weight_decay=wd,
                    decay_norm_only=decay_norm_only,
                    nesterov=nesterov,
                    cautious=cautious,
                    amsgrad=amsgrad,
                    maximize=maximize,
                )
                continue

            inv_scale = 1.0
            if backend_grad_scale is not None and not is_capturing and not compiling:
                inv_scale = float(backend_grad_scale.reciprocal().item())

            # Triton and custom CUDA kernels require moment buffers to share the same storage dtype.
            # When ApexW uses split precision (e.g. m in bf16 and v in fp32 for extreme beta2),
            # we route through the multi-tensor PyTorch foreach engine.
            _states_have_mixed_moment_dtypes = any(
                self.state[p].get("state_dtype_v", self.state[p].get("state_dtype", p.dtype))
                != self.state[p].get("state_dtype", p.dtype)
                for p in params_with_grad
                if p in self.state
            )
            _dynamic_scaling_active = group.get("dynamic_scaling", True) and any(
                exp_avg_sqs[i].dtype in (torch.bfloat16, torch.float16)
                for i in range(len(exp_avg_sqs))
            )

            # -------------------------------------------------------------
            # PATH 2A: PyTorch Native C++/CUDA Fused Multi-Tensor Fast Path
            # -------------------------------------------------------------
            # When running in full precision on CUDA/CPU with standard AdamW features,
            # dispatch directly to PyTorch's native C++ _fused_adamw_ kernel.
            # Delivers 100% throughput parity with native AdamW(fused=True)
            # while seamlessly respecting decay_norm_only via sub-grouping.
            all_fp32 = all(p.dtype == torch.float32 for p in params_with_grad)
            states_fp32 = not _dynamic_scaling_active and not _states_have_mixed_moment_dtypes and all(
                exp_avgs[i].dtype == torch.float32 for i in range(len(exp_avgs))
            )
            # Homogeneous states: each param's m and v have the same dtype (no split-precision EMA).
            # _fused_adamw_ + _group_tensors_by_device_and_dtype handles per-dtype dispatching,
            # so all-fp32 and all-bf16 subgroups are handled correctly.
            # Also require param dtype == state dtype per-tensor to prevent dtype mismatch in the kernel.
            states_homogeneous = (
                not _states_have_mixed_moment_dtypes
                and len(exp_avgs) == len(exp_avg_sqs)
                and all(
                    exp_avgs[i].dtype == exp_avg_sqs[i].dtype
                    and exp_avgs[i].dtype == params_with_grad[i].dtype
                    for i in range(len(exp_avgs))
                )
            )
            _homogeneous_fast_path = (all_fp32 and states_fp32) or states_homogeneous
            can_native_fused = (
                (fused is True or (fused is None and (precision == "full" or _homogeneous_fast_path)))
                and (is_cuda or params_with_grad[0].device.type == "cpu")
                and hasattr(torch, "_fused_adamw_")
                and beta3 is None
                and not lion
                and not cautious
                and not nesterov
                and not capturable
                and not compiling
                and not maximize
                and not has_complex
            )
            if can_native_fused:
                self._fused_step(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=wd,
                    amsgrad=amsgrad,
                    maximize=maximize,
                    decay_norm_only=decay_norm_only,
                    grad_scale=backend_grad_scale,
                    found_inf=found_inf,
                )
                if not is_capturing and not compiling:
                    self._execution_plans[group_id] = FastExecutionPlan(
                        path_type="native_fused",
                        params=params_with_grad,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        state_steps=state_steps,
                        max_exp_avg_sqs=max_exp_avg_sqs,
                        decay_norm_only=decay_norm_only,
                    )
                continue

            # -------------------------------------------------------------
            # PATH 2: OpenAI Triton Fused GPU Kernel (NVIDIA CUDA & AMD ROCm)
            # -------------------------------------------------------------
            # Iterates over parameters and invokes the single-parameter Triton launcher for each.
            # Fuses AMP gradient unscaling, AdEMAMix momentum, and cautious masking in-kernel.
            can_use_triton = (
                (backend == "triton" or (backend in (None, "auto") and is_triton_available()))
                and is_cuda
                and not nesterov
                and not lion
                and not amsgrad
                and not capturable
                and not compiling
                and not maximize
                and not has_complex
            )
            # Only evaluate steps_are_uniform when Triton or custom CUDA is viable.
            # Avoid calling _get_value (Tensor.item()) under torch.compile to prevent Dynamo graph breaks.
            steps_are_uniform = False
            can_use_cuda_fused = (
                backend in (None, "auto", "cuda")
                and is_cuda
                and not cautious
                and not nesterov
                and beta3 is None
                and not lion
                and not amsgrad
                and not capturable
                and not compiling
                and not maximize
                and backend_grad_scale is None
                and found_inf is None
                and not _states_have_mixed_moment_dtypes
                and not _dynamic_scaling_active
                and is_cuda_kernel_available()
            )
            if can_use_triton or can_use_cuda_fused:
                steps_are_uniform = len(state_steps) <= 1 or all(
                    _get_value(s) == _get_value(state_steps[0]) for s in state_steps[1:]
                )

            if can_use_triton and steps_are_uniform:
                step_val = int(_get_value(state_steps[0])) + 1
                torch._foreach_add_(state_steps, 1)
                
                # Gather per-parameter scale_v for Triton kernel when dynamic scaling is active
                triton_scales_v = None
                if _dynamic_scaling_active:
                    # Dynamic recalibration check every 100 steps
                    if step_val > 1 and step_val % 100 == 0:
                        v_max = max(v.max().item() for v in exp_avg_sqs) if exp_avg_sqs else 1.0
                        first_scale = self.state[params_with_grad[0]].get("scale_v", 1.0)
                        factor = 1.0
                        if v_max < 0.25 and first_scale < 65536.0:
                            factor = 4.0
                        elif v_max > 16384.0 and first_scale > 1.0:
                            factor = 0.25
                        if factor != 1.0:
                            torch._foreach_mul_(exp_avg_sqs, factor)
                            for p in params_with_grad:
                                s = self.state[p].get("scale_v", 1.0) * factor
                                self.state[p]["scale_v"] = s
                                self.state[p]["sqrt_scale_v"] = math.sqrt(s)
                    triton_scales_v = {
                        id(p): self.state[p].get("scale_v", 1.0)
                        for p in params_with_grad
                        if p in self.state
                    }

                success = triton_multi_tensor_adamw_step(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=wd,
                    step=step_val,
                    decay_norm_only=decay_norm_only,
                    scales_v=triton_scales_v,
                    inv_scale=inv_scale,
                    exp_avgs_slow=exp_avgs_slow if beta3 is not None else None,
                    beta3=beta3,
                    alpha=alpha,
                    cautious=cautious,
                )
                if success:
                    if not is_capturing and not compiling:
                        self._execution_plans[group_id] = FastExecutionPlan(
                            path_type="triton",
                            params=params_with_grad,
                            exp_avgs=exp_avgs,
                            exp_avg_sqs=exp_avg_sqs,
                            state_steps=state_steps,
                            exp_avgs_slow=exp_avgs_slow if beta3 is not None else None,
                            decay_norm_only=decay_norm_only,
                        )
                    continue
                # If Triton execution did not succeed, roll back step counter for fallback
                torch._foreach_sub_(state_steps, 1)


            # -------------------------------------------------------------
            # PATH 3: Custom C++/CUDA 128-bit Vectorized Fused Kernel
            # -------------------------------------------------------------
            # High-performance multi-tensor fused CUDA kernel fast path
            # (SM 7.0+ Volta/Turing/Ampere/Ada/Hopper)
            if (
                backend in (None, "auto", "cuda")
                and is_cuda
                and not cautious
                and not nesterov
                and beta3 is None
                and not lion
                and not amsgrad
                and not capturable
                and not compiling
                and not maximize
                and backend_grad_scale is None
                and found_inf is None
                and steps_are_uniform
                and is_cuda_kernel_available()
            ):
                step_val = int(_get_value(state_steps[0])) + 1
                torch._foreach_add_(state_steps, 1)

                success = step_fused(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=wd,
                    step=step_val,
                    decay_norm_only=decay_norm_only,
                )
                if success:
                    if not is_capturing and not compiling:
                        self._execution_plans[group_id] = FastExecutionPlan(
                            path_type="cuda",
                            params=params_with_grad,
                            exp_avgs=exp_avgs,
                            exp_avg_sqs=exp_avg_sqs,
                            state_steps=state_steps,
                            exp_avgs_slow=exp_avgs_slow if beta3 is not None else None,
                            decay_norm_only=decay_norm_only,
                        )
                    continue
                # If CUDA execution did not succeed, roll back step counter for fallback
                torch._foreach_sub_(state_steps, 1)

            # -------------------------------------------------------------
            # PATH 4: Apple Silicon (MPS) & PyTorch Native Multi-Tensor
            # -------------------------------------------------------------
            # PyTorch translates native _foreach_ multi-tensor operations directly
            # to Apple Metal Performance Shaders on Apple Silicon (M1/M2/M3/M4 Macs)
            # or vectorized CPU multi-tensor execution.
            if use_foreach or is_mps or backend in ("mps", "foreach", "_multi_tensor"):
                try:
                    self._multi_tensor_step(
                        params_with_grad,
                        grads,
                        exp_avgs,
                        exp_avgs_slow,
                        exp_avg_sqs,
                        max_exp_avg_sqs,
                        state_steps,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        beta3=beta3,
                        alpha=alpha,
                        lion=lion,
                        eps=eps,
                        weight_decay=wd,
                        decay_norm_only=decay_norm_only,
                        nesterov=nesterov,
                        cautious=cautious,
                        amsgrad=amsgrad,
                        maximize=maximize,
                        capturable=capturable,
                        precision=precision,
                        dynamic_scaling=group.get("dynamic_scaling", True),
                        has_complex=has_complex,
                        grad_scale=backend_grad_scale,
                        found_inf=found_inf,
                    )
                    if not is_capturing and not compiling:
                        self._execution_plans[group_id] = FastExecutionPlan(
                            path_type="foreach",
                            params=params_with_grad,
                            exp_avgs=exp_avgs,
                            exp_avg_sqs=exp_avg_sqs,
                            state_steps=state_steps,
                            exp_avgs_slow=exp_avgs_slow,
                            max_exp_avg_sqs=max_exp_avg_sqs,
                            decay_norm_only=decay_norm_only,
                            extra={"has_complex": has_complex},
                        )
                except (NotImplementedError, AttributeError):
                    # Graceful universal fallback to pure PyTorch single-tensor step
                    self._single_tensor_step(
                        params_with_grad,
                        grads,
                        exp_avgs,
                        exp_avgs_slow,
                        exp_avg_sqs,
                        max_exp_avg_sqs,
                        state_steps,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        beta3=beta3,
                        alpha=alpha,
                        lion=lion,
                        eps=eps,
                        weight_decay=wd,
                        decay_norm_only=decay_norm_only,
                        nesterov=nesterov,
                        cautious=cautious,
                        amsgrad=amsgrad,
                        maximize=maximize,
                        capturable=capturable,
                        precision=precision,
                        dynamic_scaling=group.get("dynamic_scaling", True),
                        has_complex=has_complex,
                        grad_scale=backend_grad_scale,
                        found_inf=found_inf,
                    )
            else:
                self._single_tensor_step(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avgs_slow,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    beta3=beta3,
                    alpha=alpha,
                    lion=lion,
                    eps=eps,
                    weight_decay=wd,
                    decay_norm_only=decay_norm_only,
                    nesterov=nesterov,
                    cautious=cautious,
                    amsgrad=amsgrad,
                    maximize=maximize,
                    capturable=capturable,
                    precision=precision,
                    dynamic_scaling=group.get("dynamic_scaling", True),
                    has_complex=has_complex,
                    grad_scale=backend_grad_scale,
                    found_inf=found_inf,
                )
                if not is_capturing and not compiling:
                    self._execution_plans[group_id] = FastExecutionPlan(
                        path_type="single",
                        params=params_with_grad,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        state_steps=state_steps,
                        exp_avgs_slow=exp_avgs_slow,
                        max_exp_avg_sqs=max_exp_avg_sqs,
                        decay_norm_only=decay_norm_only,
                        extra={"has_complex": has_complex},
                    )

        return loss

    def _xla_step(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avgs_slow: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        lr: float,
        beta1: float,
        beta2: float,
        beta3: Optional[float],
        alpha: float,
        lion: bool,
        eps: float,
        weight_decay: float,
        decay_norm_only: bool,
        nesterov: bool,
        cautious: bool,
        amsgrad: bool,
        maximize: bool,
    ):
        """
        Global graph compilation execution path for Google TPUs (PyTorch XLA).

        Maintains ZERO CPU-TPU synchronization barriers:
        - Step counters updated purely on-device as XLA graph nodes.
        - Bias corrections computed as tensor math on device (no .item() calls).
        - Elementwise operations structured for XLA HLO compiler kernel fusion into TPU ASICs.
        """
        if len(params) == 0:
            return

        device = params[0].device

        # Step 1: Update step counters in-place on device (zero host barrier)
        for st in state_steps:
            st.add_(1)

        # Step 2: Compute shared device tensors for bias corrections (per-step, inside param loop)
        beta1_t = torch.tensor(beta1, dtype=torch.float32, device=device)
        beta2_t = torch.tensor(beta2, dtype=torch.float32, device=device)
        beta3_t = torch.tensor(beta3, dtype=torch.float32, device=device) if beta3 is not None else None

        # Step 3: Fused tensor updates
        for i, p in enumerate(params):
            # Bug 6 fix: use per-parameter step_t, not always state_steps[0]
            step_t = state_steps[i]

            # Compute bias correction scalars for this parameter's step
            bc1 = 1.0 - torch.pow(beta1_t, step_t)
            bc2 = 1.0 - torch.pow(beta2_t, step_t)
            bc2_sqrt = torch.sqrt(bc2)
            step_size = lr * bc2_sqrt / bc1
            eps_scaled = eps * bc2_sqrt

            slow_ratio = None
            if beta3 is not None:
                slow_ratio = alpha * (bc1 / (1.0 - torch.pow(beta3_t, step_t)))

            g = grads[i]
            m = exp_avgs[i]
            m_slow = exp_avgs_slow[i] if beta3 is not None else None
            v = exp_avg_sqs[i] if not lion else None
            max_v = max_exp_avg_sqs[i] if amsgrad else None

            if maximize:
                g = -g

            # Decoupled weight decay
            if weight_decay != 0.0:
                if decay_norm_only and p.dim() < 2:
                    pass
                else:
                    p.mul_(1.0 - lr * weight_decay)

            # Cast gradient to state dtype for mixed precision if needed
            g_comp = g if g.dtype == m.dtype else g.to(m.dtype)

            # Lion mode
            if lion:
                update = torch.lerp(g_comp, m, beta1).sign()
                p.add_(update.to(p.dtype), alpha=-lr)
                m.lerp_(g_comp, 1.0 - beta2)
                continue

            # AdamW mode
            g_comp = g if g.dtype == m.dtype else g.to(m.dtype)

            # Update first-order momentum
            m.lerp_(g_comp, 1.0 - beta1)
            if m_slow is not None:
                m_slow.lerp_(g_comp, 1.0 - beta3)

            # Update second-order momentum
            v.mul_(beta2).addcmul_(g_comp, g_comp, value=1.0 - beta2)

            if amsgrad:
                torch.maximum(max_v, v, out=max_v)
                v_target = max_v
            else:
                v_target = v

            denom = v_target.sqrt().add_(eps_scaled)

            # Compute numerator
            if m_slow is not None:
                num = m.add(m_slow, alpha=slow_ratio)
            elif nesterov:
                num = torch.lerp(g_comp, m, beta1)
            else:
                num = m

            if cautious:
                mask = (g.float() * num.float() >= 0).to(p.dtype)
                # Bug 7 fix: normalize mask to preserve update magnitude, matching _single_tensor_step
                mask_mean = mask.mean().clamp(min=1e-3)
                mask.div_(mask_mean)
                num = num * mask

            p.addcdiv_(num, denom, value=-step_size)

    def _fused_step(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool,
        maximize: bool,
        decay_norm_only: bool = False,
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
    ):
        """Vectorized execution using PyTorch native C++/CUDA _fused_adamw_ kernel."""
        tensor_lists = [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
        grouped = Optimizer._group_tensors_by_device_and_dtype(tensor_lists)
        for (device, _), ((d_p, d_g, d_m, d_v, d_max_v, d_steps), _) in grouped.items():
            device_grad_scale = (
                grad_scale.to(device, non_blocking=True) if grad_scale is not None else None
            )
            device_found_inf = (
                found_inf.to(device, non_blocking=True) if found_inf is not None else None
            )
            torch._foreach_add_(d_steps, 1)

            if decay_norm_only and weight_decay != 0.0:
                decay_idx = [i for i, p in enumerate(d_p) if p.dim() >= 2]
                no_decay_idx = [i for i, p in enumerate(d_p) if p.dim() < 2]

                if decay_idx:
                    torch._fused_adamw_(
                        [d_p[i] for i in decay_idx],
                        [d_g[i] for i in decay_idx],
                        [d_m[i] for i in decay_idx],
                        [d_v[i] for i in decay_idx],
                        [d_max_v[i] for i in decay_idx] if d_max_v else [],
                        [d_steps[i] for i in decay_idx],
                        amsgrad=amsgrad,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        eps=eps,
                        maximize=maximize,
                        grad_scale=device_grad_scale,
                        found_inf=device_found_inf,
                    )
                if no_decay_idx:
                    torch._fused_adamw_(
                        [d_p[i] for i in no_decay_idx],
                        [d_g[i] for i in no_decay_idx],
                        [d_m[i] for i in no_decay_idx],
                        [d_v[i] for i in no_decay_idx],
                        [d_max_v[i] for i in no_decay_idx] if d_max_v else [],
                        [d_steps[i] for i in no_decay_idx],
                        amsgrad=amsgrad,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=0.0,
                        eps=eps,
                        maximize=maximize,
                        grad_scale=device_grad_scale,
                        found_inf=device_found_inf,
                    )
            else:
                torch._fused_adamw_(
                    d_p,
                    d_g,
                    d_m,
                    d_v,
                    d_max_v,
                    d_steps,
                    amsgrad=amsgrad,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    eps=eps,
                    maximize=maximize,
                    grad_scale=device_grad_scale,
                    found_inf=device_found_inf,
                )
            if device_found_inf is not None:
                torch._foreach_sub_(d_steps, [device_found_inf] * len(d_steps))

    def _single_tensor_step(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avgs_slow: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        lr: float,
        beta1: float,
        beta2: float,
        beta3: Optional[float],
        alpha: float,
        lion: bool,
        eps: float,
        weight_decay: float,
        decay_norm_only: bool,
        nesterov: bool,
        cautious: bool,
        amsgrad: bool,
        maximize: bool,
        capturable: bool,
        precision: str,
        dynamic_scaling: bool = True,
        has_complex: bool = False,
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
    ):
        """Eager single-tensor update with folded scalar math."""
        compiling = is_compiling()

        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None and not is_capturing and not compiling:
            if found_inf.item() > 0:
                return

        inv_scale = None
        if grad_scale is not None:
            inv_scale = grad_scale.reciprocal()

        if capturable:
            for p, step_t in zip(params, state_steps):
                if p.device.type != step_t.device.type:
                    raise RuntimeError(
                        f"capturable requires param and step to be on the same device, "
                        f"but param is on {p.device} and step is on {step_t.device}"
                    )
                if p.device.type not in CAPTURABLE_SUPPORTED_DEVICES:
                    raise RuntimeError(
                        f"capturable is not supported on {p.device.type}"
                    )

        # Precompute step-level scalars once outside the loop for maximum CPU throughput
        decay_factor = 1.0 - lr * weight_decay if weight_decay != 0 else 1.0
        step_size = None
        eps_scaled = None
        slow_ratio = 0.0
        # Cache bias-correction scalars by (step, effective_eps) so parameters sharing
        # the same step count reuse precomputed scalars without redundant math.
        step_cache = {}

        for i, p in enumerate(params):
            g = grads[i] if not maximize else -grads[i]
            if inv_scale is not None:
                g = g * inv_scale
            m = exp_avgs[i]
            m_slow = exp_avgs_slow[i] if beta3 is not None else None
            v = exp_avg_sqs[i] if not lion else None
            step_t = state_steps[i]

            # Handle complex parameters
            if torch.is_complex(p):
                g = torch.view_as_real(g)
                m = torch.view_as_real(m)
                if m_slow is not None:
                    m_slow = torch.view_as_real(m_slow)
                if v is not None:
                    v = torch.view_as_real(v)
                p = torch.view_as_real(p)
                if amsgrad:
                    max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])

            # 1. Update step counter
            step_t.add_(1)

            # Epsilon underflow protection for FP16 compact states
            effective_eps = max(eps, 6.1e-5) if m.dtype == torch.float16 else eps

            if capturable or compiling:
                step = step_t
                bc1 = 1.0 - torch.pow(beta1, step)
                bc2 = 1.0 - torch.pow(beta2, step)
                bc2_sqrt = bc2.sqrt()
                step_size = lr / bc1
                step_size_neg = step_size.neg()
                slow_ratio = alpha * (bc1 / (1.0 - torch.pow(beta3, step))) if beta3 is not None else 0.0
            else:
                step = _get_value(step_t)
                cache_key = (step, effective_eps)
                if cache_key in step_cache:
                    step_size, eps_scaled, slow_ratio, bc2_sqrt = step_cache[cache_key]
                else:
                    bc1 = 1.0 - (beta1 ** step)
                    bc2 = 1.0 - (beta2 ** step)
                    bc2_sqrt = math.sqrt(bc2)
                    step_size = lr * bc2_sqrt / bc1
                    eps_scaled = effective_eps * bc2_sqrt
                    slow_ratio = alpha * (bc1 / (1.0 - (beta3 ** step))) if beta3 is not None else 0.0
                    step_cache[cache_key] = (step_size, eps_scaled, slow_ratio, bc2_sqrt)

            # 2. Decoupled weight decay
            if weight_decay != 0:
                if not decay_norm_only or p.dim() >= 2:
                    p.mul_(decay_factor)

            # 3. Lion Mode update
            if lion:
                state_dtype = m.dtype
                g_comp = g if g.dtype == state_dtype else g.to(state_dtype)
                # Lion update direction: sign(beta1 * m + (1 - beta1) * g)
                update = torch.lerp(g_comp, m, beta1).sign()
                p.add_(update, alpha=-lr)
                # Momentum update: m = beta2 * m + (1 - beta2) * g
                m.lerp_(g_comp, 1.0 - beta2)
                continue

            # 4. Standard AdamW / AdEMAMix update
            state_dtype = m.dtype

            # Numerical Contract: FP32 arithmetic with compact (16-bit) state storage.
            # Consistent with Triton GPU and C++/CUDA fused kernels, arithmetic is evaluated
            # in float32 to prevent truncation/underflow in gradient moving averages, then
            # stored back into the compact 16-bit state buffers.
            if state_dtype in (torch.bfloat16, torch.float16):
                g_f = g.float()
                m_f = m.float().lerp_(g_f, 1.0 - beta1)
                m.copy_(m_f)
                if m_slow is not None:
                    m_slow_f = m_slow.float().lerp_(g_f, 1.0 - beta3)
                    m_slow.copy_(m_slow_f)
            else:
                m.lerp_(g, 1.0 - beta1)
                if m_slow is not None:
                    m_slow.lerp_(g, 1.0 - beta3)

            # Dynamic range scaling for 16-bit variance states (prevents underflow)
            scale_v = 1.0
            sqrt_scale_v = 1.0
            if dynamic_scaling and v is not None and v.dtype in (torch.bfloat16, torch.float16):
                p_state = self.state[p]
                key = id(p)
                if "scale_v" not in p_state:
                    init_scale = 1024.0 if v.dtype == torch.float16 else 1.0
                    p_state["scale_v"] = init_scale
                    p_state["sqrt_scale_v"] = math.sqrt(init_scale)
                    self._workspace_pool[key] = {"scale_v": init_scale, "sqrt_scale_v": math.sqrt(init_scale)}
                elif key not in self._workspace_pool:
                    self._workspace_pool[key] = {
                        "scale_v": p_state["scale_v"],
                        "sqrt_scale_v": p_state.get("sqrt_scale_v", math.sqrt(p_state["scale_v"])),
                    }

                scale_info = self._workspace_pool[key]
                scale_v = scale_info["scale_v"]
                sqrt_scale_v = scale_info["sqrt_scale_v"]

                if not compiling and not capturable:
                    step_val = int(_get_value(step_t))
                    if step_val > 1 and step_val % 100 == 0:
                        v_max = v.max().item()
                        if v_max < 0.25 and scale_v < 65536.0:
                            factor = 4.0
                            scale_v *= factor
                            sqrt_scale_v = math.sqrt(scale_v)
                            v.mul_(factor)
                            scale_info["scale_v"] = scale_v
                            scale_info["sqrt_scale_v"] = sqrt_scale_v
                            p_state["scale_v"] = scale_v
                            p_state["sqrt_scale_v"] = sqrt_scale_v
                        elif v_max > 16384.0 and scale_v > 1.0:
                            factor = 0.25
                            scale_v *= factor
                            sqrt_scale_v = math.sqrt(scale_v)
                            v.mul_(factor)
                            scale_info["scale_v"] = scale_v
                            scale_info["sqrt_scale_v"] = sqrt_scale_v
                            p_state["scale_v"] = scale_v
                            p_state["sqrt_scale_v"] = sqrt_scale_v

            # Update second-order momentum with dynamic scale
            if state_dtype in (torch.bfloat16, torch.float16):
                v_f = v.float().mul_(beta2).addcmul_(g_f, g_f, value=(1.0 - beta2) * scale_v)
                v.copy_(v_f)
            else:
                v.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2) * scale_v)

            # Fast path for standard AdamW (zero extra allocations)
            if not amsgrad and not cautious and not nesterov and m_slow is None:
                if capturable or compiling:
                    if scale_v != 1.0:
                        denom = (v.sqrt() / sqrt_scale_v / (bc2_sqrt * step_size_neg)).add_(effective_eps / step_size_neg)
                    else:
                        denom = (v.sqrt() / (bc2_sqrt * step_size_neg)).add_(effective_eps / step_size_neg)
                    p.addcdiv_(m, denom)
                else:
                    if scale_v != 1.0:
                        denom = v.sqrt().add_(eps_scaled * sqrt_scale_v).div_(sqrt_scale_v)
                    else:
                        denom = v.sqrt().add_(eps_scaled)
                    p.addcdiv_(m, denom, value=-step_size)
                continue

            # Compute denominator (in-place sqrt + add_)
            if capturable or compiling:
                v_target = max_exp_avg_sqs[i] if amsgrad else v
                if amsgrad:
                    torch.maximum(v_target, v, out=v_target)
                if scale_v != 1.0:
                    denom = (v_target.sqrt() / sqrt_scale_v / (bc2_sqrt * step_size_neg)).add_(effective_eps / step_size_neg)
                else:
                    denom = (v_target.sqrt() / (bc2_sqrt * step_size_neg)).add_(effective_eps / step_size_neg)
            else:
                if amsgrad:
                    max_v = max_exp_avg_sqs[i]
                    torch.maximum(max_v, v, out=max_v)
                    if scale_v != 1.0:
                        denom = max_v.sqrt().add_(eps_scaled * sqrt_scale_v).div_(sqrt_scale_v)
                    else:
                        denom = max_v.sqrt().add_(eps_scaled)
                else:
                    if scale_v != 1.0:
                        denom = v.sqrt().add_(eps_scaled * sqrt_scale_v).div_(sqrt_scale_v)
                    else:
                        denom = v.sqrt().add_(eps_scaled)

            # Compute numerator
            m_f = m if m.dtype == torch.float32 else m.float()
            if m_slow is not None:
                m_slow_f = m_slow if m_slow.dtype == torch.float32 else m_slow.float()
                num = m_f.add(m_slow_f, alpha=slow_ratio)
            elif nesterov:
                g_f = g if g.dtype == torch.float32 else g.float()
                num = torch.lerp(g_f, m_f, beta1)
            else:
                num = m_f

            if cautious:
                mask = (g.float() * num >= 0).to(p.dtype)
                mask_mean = mask.mean().clamp(min=1e-3)
                mask.div_(mask_mean)
                num = num * mask

            if capturable or compiling:
                p.addcdiv_(num, denom)
            else:
                p.addcdiv_(num, denom, value=-step_size)

    def _multi_tensor_step(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avgs_slow: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        lr: float,
        beta1: float,
        beta2: float,
        beta3: Optional[float],
        alpha: float,
        lion: bool,
        eps: float,
        weight_decay: float,
        decay_norm_only: bool,
        nesterov: bool,
        cautious: bool,
        amsgrad: bool,
        maximize: bool,
        capturable: bool,
        precision: str,
        dynamic_scaling: bool = True,
        has_complex: bool = False,
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
    ):
        """Vectorized native C++ execution using torch._foreach_* multi-tensor operations."""
        if len(params) == 0:
            return

        compiling = is_compiling()

        if capturable and not compiling:
            for p, step_t in zip(params, state_steps):
                if p.device.type != step_t.device.type:
                    raise RuntimeError(
                        f"capturable requires param and step to be on the same device, "
                        f"but param is on {p.device} and step is on {step_t.device}"
                    )
                if p.device.type not in CAPTURABLE_SUPPORTED_DEVICES:
                    raise RuntimeError(
                        f"capturable is not supported on {p.device.type}"
                    )

        # Step 2: Smart Grouping Dispatcher
        # Group tensors strictly by (device, param_dtype, state_dtype)
        # to separate homogenous subgroups from mixed-precision subgroups
        subgroups = defaultdict(lambda: {
            "params": [], "grads": [], "exp_avgs": [], "steps": [],
            "exp_avgs_slow": [], "exp_avg_sqs": [], "max_exp_avg_sqs": []
        })

        for i, p in enumerate(params):
            m = exp_avgs[i]
            # When not capturable and not compiling, partition also by current step count
            # so each subgroup is guaranteed uniform step counts. This ensures bias-correction
            # scalars are mathematically exact for every parameter, even with intermittent gradients.
            s_val = int(_get_value(state_steps[i])) if (not capturable and not compiling) else None
            key = (p.device, p.dtype, m.dtype, s_val)
            sg = subgroups[key]
            sg["params"].append(p)
            sg["grads"].append(grads[i])
            sg["exp_avgs"].append(m)
            sg["steps"].append(state_steps[i])
            if beta3 is not None:
                sg["exp_avgs_slow"].append(exp_avgs_slow[i])
            if not lion:
                sg["exp_avg_sqs"].append(exp_avg_sqs[i])
                if amsgrad:
                    sg["max_exp_avg_sqs"].append(max_exp_avg_sqs[i])

        for (dev, param_dt, state_dt, _), sg in subgroups.items():
            d_params = sg["params"]
            d_grads = sg["grads"]
            d_exp_avgs = sg["exp_avgs"]
            d_steps = sg["steps"]
            d_exp_avgs_slow = sg["exp_avgs_slow"]
            d_exp_avg_sqs = sg["exp_avg_sqs"]
            d_max_exp_avg_sqs = sg["max_exp_avg_sqs"]

            # Step 1: Dispatch to custom multi-tensor batched CUDA kernel (1 launch for all tensors)
            # Guard: all params in this subgroup must have the same step counter.
            # When gradients are intermittent, step counts diverge and the kernel's
            # single-step bias correction would be wrong for the lagging parameters.
            can_cuda_fused = (
                dev.type == "cuda"
                and param_dt in (torch.float32, torch.bfloat16, torch.float16)
                and state_dt in (torch.float32, torch.bfloat16, torch.float16)
                and not cautious
                and not nesterov
                and beta3 is None
                and not lion
                and not amsgrad
                and not capturable
                and not compiling
                and not maximize
                and grad_scale is None
                and found_inf is None
                and is_cuda_kernel_available()
            )
            d_steps_uniform = False
            if can_cuda_fused:
                d_steps_uniform = len(d_steps) <= 1 or all(
                    _get_value(s) == _get_value(d_steps[0]) for s in d_steps[1:]
                )

            if can_cuda_fused and d_steps_uniform:
                step_val = int(_get_value(d_steps[0])) + 1
                torch._foreach_add_(d_steps, 1)
                success = fused_multi_tensor_adamw_cuda_step(
                    d_params,
                    d_grads,
                    d_exp_avgs,
                    d_exp_avg_sqs,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    eps=eps,
                    step=step_val,
                    decay_norm_only=decay_norm_only,
                )
                if success:
                    continue
                # If multi-tensor CUDA execution was not supported, undo step increment for fallback
                torch._foreach_sub_(d_steps, 1)

            # Handle complex parameters
            if has_complex and torch.is_complex(d_params[0]):
                d_params = [torch.view_as_real(t) for t in d_params]
                d_grads = [torch.view_as_real(t) for t in d_grads]
                d_exp_avgs = [torch.view_as_real(t) for t in d_exp_avgs]
                if beta3 is not None:
                    d_exp_avgs_slow = [torch.view_as_real(t) for t in d_exp_avgs_slow]
                if not lion:
                    d_exp_avg_sqs = [torch.view_as_real(t) for t in d_exp_avg_sqs]
                    if amsgrad:
                        d_max_exp_avg_sqs = [torch.view_as_real(t) for t in d_max_exp_avg_sqs]

            # Non-mutating gradient negation for maximization (preserves caller's gradient buffers)
            if maximize:
                d_grads = torch._foreach_neg(d_grads)

            # Handle AMP found_inf: check early before performing any unscaling
            if found_inf is not None:
                device_found_inf = found_inf
                if dev is not None and found_inf.device != dev:
                    device_found_inf = found_inf.to(dev)
                is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
                if not is_capturing and not compiling and device_found_inf.item() > 0:
                    continue

            # Handle AMP grad_scale: in-place unscaling (zero auxiliary VRAM allocations)
            if grad_scale is not None:
                device_grad_scale = grad_scale
                if dev is not None and grad_scale.device != dev:
                    device_grad_scale = grad_scale.to(dev)
                inv_scale = device_grad_scale.reciprocal()
                torch._foreach_mul_(d_grads, inv_scale)

            # Increment step counters
            if not compiling and d_steps[0].is_cpu:
                torch._foreach_add_(
                    d_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
                )
            else:
                torch._foreach_add_(d_steps, 1)

            use_tensor_steps = capturable or compiling

            # Compute bias correction scalars
            if use_tensor_steps:
                step_lr = max(lr, 1e-16) if isinstance(lr, (int, float)) else lr
                bias_correction1 = torch._foreach_pow(beta1, d_steps)
                bias_correction2 = torch._foreach_pow(beta2, d_steps)
                torch._foreach_sub_(bias_correction1, 1)
                torch._foreach_sub_(bias_correction2, 1)
                torch._foreach_neg_(bias_correction2)
                torch._foreach_div_(bias_correction1, step_lr)
                torch._foreach_reciprocal_(bias_correction1)
                torch._foreach_sqrt_(bias_correction2)

                step_size_list = bias_correction1
                bc2_sqrt_list = bias_correction2
                # Bug 4 fix: slow_ratio must include bias-correction ratio bc1/bc3
                if beta3 is not None:
                    if compiling:
                        bc1_t = 1.0 - (beta1 ** d_steps[0])
                        bc3_t = 1.0 - (beta3 ** d_steps[0])
                        slow_ratio = alpha * (bc1_t / bc3_t)
                    else:
                        step_val = _get_value(d_steps[0])
                        bc1_val = 1.0 - (beta1 ** step_val)
                        bc3_val = 1.0 - (beta3 ** step_val)
                        slow_ratio = alpha * (bc1_val / bc3_val)
                else:
                    slow_ratio = 0.0
            else:
                step = _get_value(d_steps[0])
                bc1 = 1.0 - (beta1 ** step)
                bc2 = 1.0 - (beta2 ** step)
                bc2_sqrt = math.sqrt(bc2)
                step_size = lr * bc2_sqrt / bc1
                eps_scaled = eps * bc2_sqrt
                slow_ratio = alpha * (bc1 / (1.0 - (beta3 ** step))) if beta3 is not None else 0.0

            # 1. Decoupled weight decay
            if weight_decay != 0:
                if decay_norm_only:
                    decay_params = [p for p in d_params if p.dim() >= 2]
                    if decay_params:
                        torch._foreach_mul_(decay_params, 1.0 - lr * weight_decay)
                else:
                    torch._foreach_mul_(d_params, 1.0 - lr * weight_decay)

            target_dtype = d_exp_avgs[0].dtype
            # Use first param's id as the workspace key so each subgroup gets its own
            # dynamic scale entry. The previous (dev, param_dt, state_dt) key caused
            # different parameter subgroups (e.g. embeddings vs MLP weights) sharing
            # the same dtype to overwrite each other's calibrated scale_v.
            key = id(d_params[0])

            # Dynamic range scaling check & recalibration across subgroup (protects 16-bit states from underflow)
            scale_v = 1.0
            sqrt_scale_v = 1.0
            if dynamic_scaling and not lion and target_dtype in (torch.bfloat16, torch.float16):
                first_p_state = self.state[d_params[0]]
                if "scale_v" not in first_p_state:
                    init_scale = 1024.0 if target_dtype == torch.float16 else 1.0
                    sqrt_init = math.sqrt(init_scale)
                    for p in d_params:
                        self.state[p]["scale_v"] = init_scale
                        self.state[p]["sqrt_scale_v"] = sqrt_init
                    self._workspace_pool[key] = {"scale_v": init_scale, "sqrt_scale_v": sqrt_init}
                elif key not in self._workspace_pool:
                    self._workspace_pool[key] = {
                        "scale_v": first_p_state["scale_v"],
                        "sqrt_scale_v": first_p_state.get("sqrt_scale_v", math.sqrt(first_p_state["scale_v"])),
                    }

                scale_info = self._workspace_pool[key]
                scale_v = scale_info["scale_v"]
                sqrt_scale_v = scale_info["sqrt_scale_v"]

                if not compiling and not capturable:
                    step_val = int(_get_value(d_steps[0]))
                    if step_val > 1 and step_val % 100 == 0:
                        v_max = max(v.max().item() for v in d_exp_avg_sqs) if d_exp_avg_sqs else 1.0
                        if v_max < 0.25 and scale_v < 65536.0:
                            factor = 4.0
                            scale_v *= factor
                            sqrt_scale_v = math.sqrt(scale_v)
                            torch._foreach_mul_(d_exp_avg_sqs, factor)
                            scale_info["scale_v"] = scale_v
                            scale_info["sqrt_scale_v"] = sqrt_scale_v
                            for p in d_params:
                                self.state[p]["scale_v"] = scale_v
                                self.state[p]["sqrt_scale_v"] = sqrt_scale_v
                        elif v_max > 16384.0 and scale_v > 1.0:
                            factor = 0.25
                            scale_v *= factor
                            sqrt_scale_v = math.sqrt(scale_v)
                            torch._foreach_mul_(d_exp_avg_sqs, factor)
                            scale_info["scale_v"] = scale_v
                            scale_info["sqrt_scale_v"] = sqrt_scale_v
                            for p in d_params:
                                self.state[p]["scale_v"] = scale_v
                                self.state[p]["sqrt_scale_v"] = sqrt_scale_v

            effective_eps = (
                max(eps, 6.1e-5)
                if (target_dtype == torch.float16)
                else eps
            )
            if not use_tensor_steps:
                eps_scaled = effective_eps * bc2_sqrt

            # Bounded Chunking: Partition tensor list so temporary memory never exceeds <= 16 MB
            chunks = _chunk_tensor_lists(
                d_params,
                d_grads,
                d_exp_avgs,
                d_exp_avgs_slow if beta3 is not None else None,
                d_exp_avg_sqs if not lion else None,
                d_max_exp_avg_sqs if amsgrad else None,
                step_size_list if use_tensor_steps else None,
                bc2_sqrt_list if use_tensor_steps else None,
            )

            for (
                c_params, c_grads, c_exp_avgs,
                c_exp_avgs_slow, c_exp_avg_sqs,
                c_max_exp_avg_sqs, c_step_size_list, c_bc2_sqrt_list
            ) in chunks:
                # Cast grads to compact dtype if needed
                if c_grads[0].dtype != target_dtype:
                    c_grads_comp = [g.to(target_dtype) for g in c_grads]
                else:
                    c_grads_comp = c_grads

                # 2. Lion Mode multi-tensor update
                if lion:
                    updates = torch._foreach_lerp(c_grads_comp, c_exp_avgs, beta1)
                    updates = [u.sign() for u in updates]
                    torch._foreach_add_(c_params, updates, alpha=-lr)
                    torch._foreach_lerp_(c_exp_avgs, c_grads_comp, 1.0 - beta2)
                    del updates, c_grads_comp
                    continue

                # Numerical Contract: FP32 arithmetic with compact (16-bit) state storage.
                # Fix 3: Zero-alloc cross-precision path — mixed-dtype _foreach_ ops update BF16/FP16
                # state tensors in-place directly from FP32 gradients, eliminating ~210 per-step
                # temporary FP32 tensor allocations (cudaMalloc calls) that existed in the old path.
                #
                # _foreach_lerp_ does NOT support mixed dtypes (bf16 += fp32 raises RuntimeError),
                # so we use the equivalent mul+add decomposition:
                #   m = m * beta1 + g * (1 - beta1)  ←→  lerp(m, g, 1-beta1)
                #
                # _foreach_add_(bf16, fp32, alpha=x) ✓ verified in-place cross-dtype
                # _foreach_addcmul_(bf16, fp32, fp32, value=x) ✓ verified in-place cross-dtype
                # _foreach_lerp_(bf16, fp32, weight) ✗ raises RuntimeError — do NOT use
                if target_dtype in (torch.bfloat16, torch.float16):
                    # Upcast grads only when needed (e.g., AMP FP16 grads with BF16 states).
                    # This single list is unavoidable; it shrinks allocs from 315→105 per chunk.
                    c_grads_f = c_grads if c_grads[0].dtype == torch.float32 else [g.float() for g in c_grads]

                    # Update first-order momentum in-place: m = m * beta1 + g * (1 - beta1)
                    torch._foreach_mul_(c_exp_avgs, beta1)
                    torch._foreach_add_(c_exp_avgs, c_grads_f, alpha=1.0 - beta1)

                    if beta3 is not None:
                        # Update slow EMA in-place with same zero-alloc pattern
                        torch._foreach_mul_(c_exp_avgs_slow, beta3)
                        torch._foreach_add_(c_exp_avgs_slow, c_grads_f, alpha=1.0 - beta3)

                    # Update second-order momentum in-place: v = v * beta2 + g^2 * ((1-beta2)*scale_v)
                    # _foreach_addcmul_(bf16, fp32, fp32) is verified to work cross-dtype in-place.
                    torch._foreach_mul_(c_exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(c_exp_avg_sqs, c_grads_f, c_grads_f, value=(1.0 - beta2) * scale_v)
                else:
                    # Update first-order momentum
                    # Use c_grads_comp (cast to target_dtype at the top of the chunk loop) so
                    # bf16 params with fp32 states (e.g. 1D norm layers on a bf16 model) work correctly.
                    torch._foreach_lerp_(c_exp_avgs, c_grads_comp, 1.0 - beta1)
                    if beta3 is not None:
                        torch._foreach_lerp_(c_exp_avgs_slow, c_grads_comp, 1.0 - beta3)

                    # Update second-order momentum with dynamic range scale
                    torch._foreach_mul_(c_exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(c_exp_avg_sqs, c_grads_comp, c_grads_comp, value=(1.0 - beta2) * scale_v)

                if amsgrad:
                    torch._foreach_maximum_(c_max_exp_avg_sqs, c_exp_avg_sqs)
                    v_sources = c_max_exp_avg_sqs
                else:
                    v_sources = c_exp_avg_sqs

                # Free c_grads_comp early if not needed for nesterov/cautious
                if not cautious and not nesterov:
                    del c_grads_comp

                # Vectorized denominator computation
                denoms = torch._foreach_sqrt(v_sources)

                if use_tensor_steps:
                    if scale_v != 1.0:
                        torch._foreach_div_(denoms, c_bc2_sqrt_list)
                        torch._foreach_add_(denoms, effective_eps * sqrt_scale_v)
                        torch._foreach_div_(denoms, c_step_size_list)
                        torch._foreach_div_(denoms, sqrt_scale_v)
                    else:
                        torch._foreach_div_(denoms, c_bc2_sqrt_list)
                        torch._foreach_add_(denoms, effective_eps)
                        torch._foreach_div_(denoms, c_step_size_list)
                else:
                    if scale_v != 1.0:
                        torch._foreach_add_(denoms, eps_scaled * sqrt_scale_v)
                        torch._foreach_div_(denoms, sqrt_scale_v)
                    else:
                        torch._foreach_add_(denoms, eps_scaled)

                # Standard AdamW path (Zero extra numerator allocations, natively handles mixed precision)
                if not cautious and not nesterov and beta3 is None:
                    if use_tensor_steps:
                        torch._foreach_addcdiv_(c_params, c_exp_avgs, denoms)
                    else:
                        torch._foreach_addcdiv_(c_params, c_exp_avgs, denoms, value=-step_size)
                    del denoms
                    continue

                # Numerator computation for advanced modes (AdEMAMix, Cautious, Nesterov)
                if c_params[0].dtype == torch.float32 and c_exp_avgs[0].dtype != torch.float32:
                    exp_avgs_f = [m.float() for m in c_exp_avgs]
                    exp_avgs_slow_f = [m.float() for m in c_exp_avgs_slow] if beta3 is not None else []
                else:
                    exp_avgs_f = c_exp_avgs
                    exp_avgs_slow_f = c_exp_avgs_slow

                if beta3 is not None:
                    nums = torch._foreach_add(exp_avgs_f, exp_avgs_slow_f, alpha=slow_ratio)
                elif nesterov and not cautious:
                    nums = None
                else:
                    nums = list(exp_avgs_f)

                if cautious:
                    if nesterov and beta3 is None:
                        grads_f = [g.float() for g in c_grads] if c_grads[0].dtype != torch.float32 else c_grads
                        nums = [torch.lerp(g, m, beta1) for g, m in zip(grads_f, exp_avgs_f)]
                    for idx in range(len(nums)):
                        g_f = c_grads[idx].float()
                        num_f = nums[idx].float()
                        mask = (g_f * num_f >= 0).to(c_params[idx].dtype)
                        mask_mean = mask.mean().clamp(min=1e-3)
                        mask.div_(mask_mean)
                        nums[idx] = nums[idx] * mask

                    if use_tensor_steps:
                        torch._foreach_addcdiv_(c_params, nums, denoms)
                    else:
                        torch._foreach_addcdiv_(c_params, nums, denoms, value=-step_size)
                    del nums

                elif nesterov and beta3 is None:
                    grads_f = [g.float() for g in c_grads] if c_grads[0].dtype != torch.float32 else c_grads
                    if use_tensor_steps:
                        nums = [torch.lerp(g, m, beta1) for g, m in zip(grads_f, exp_avgs_f)]
                        torch._foreach_addcdiv_(c_params, nums, denoms)
                        del nums
                    else:
                        torch._foreach_addcdiv_(c_params, exp_avgs_f, denoms, value=-step_size * beta1)
                        torch._foreach_addcdiv_(c_params, grads_f, denoms, value=-step_size * (1.0 - beta1))
                else:
                    if use_tensor_steps:
                        torch._foreach_addcdiv_(c_params, nums, denoms)
                    else:
                        torch._foreach_addcdiv_(c_params, nums, denoms, value=-step_size)
                    if beta3 is not None:
                        del nums

                del denoms
                if cautious or nesterov:
                    del c_grads_comp

            if maximize:
                del d_grads


def build_llm_param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    """
    Partitions model parameters into Decayed and Non-Decayed groups following
    frontier LLM recipes (LLaMA-3, DeepSeek-V3, Qwen-2.5).
    - 2D+ tensors (Linear weights, attention projections, MLP heads) -> weight_decay
    - 1D tensors (RMSNorm scales, LayerNorm, biases) -> weight_decay = 0.0
    - Embeddings -> optional isolation
    """
    decay_params: List[Tensor] = []
    nodecay_params: List[Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Exclude 1D parameters: RMSNorm gamma, LayerNorm weights, and biases
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            nodecay_params.append(param)
        elif no_decay_embeddings and ("embed" in name.lower() or "wte" in name.lower()):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay, "group_name": "decayed"},
        {"params": nodecay_params, "weight_decay": 0.0, "group_name": "non_decayed"},
    ]
