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
_CUDA_DTYPE_PAIRS = {
    (torch.float32, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.float32, torch.float16),
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float32),
    (torch.float16, torch.float16),
}


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


def _quantize_blockwise_int8(
    tensor: Tensor,
    block_size: int = 256,
    stochastic: bool = False,
    out_q: Optional[Tensor] = None,
    out_scales: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Quantizes a float tensor into block-wise INT8 matching the tensor's exact shape.

    Uses per-block float32 absolute maximum scaling to achieve dynamic INT8 representation.
    If stochastic=True, applies unbiased stochastic rounding so that E[q * scale] == x.
    Supports in-place writing to out_q and out_scales to eliminate temporary allocations.
    """
    shape = tensor.shape
    numel = tensor.numel()
    if numel == 0:
        return torch.empty_like(tensor, dtype=torch.int8), torch.empty((0, 1), device=tensor.device, dtype=torch.float32)

    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        blocks = tensor.reshape(-1, block_size).float()
    else:
        flat = torch.nn.functional.pad(tensor.reshape(-1), (0, pad))
        blocks = flat.view(-1, block_size).float()

    abs_max = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scales = (abs_max / 127.0).to(torch.float32)
    scaled = blocks / scales
    if stochastic:
        # Stochastic rounding: mathematically eliminates quantization bias in momentum accumulation
        noise = torch.empty_like(scaled).uniform_(-0.5, 0.5)
        q_blocks = (scaled + noise).round().clamp(-128, 127).to(torch.int8)
    else:
        q_blocks = scaled.round().clamp(-128, 127).to(torch.int8)

    q = q_blocks.view(shape) if pad == 0 else q_blocks.view(-1)[:numel].view(shape)
    if out_q is not None:
        out_q.copy_(q)
    if out_scales is not None:
        out_scales.copy_(scales)
    return (out_q if out_q is not None else q), (out_scales if out_scales is not None else scales)


def _dequantize_blockwise_int8(q: Tensor, scales: Tensor, block_size: int = 256) -> Tensor:
    """Dequantizes block-wise INT8 back into a float32 tensor matching q's shape."""
    shape = q.shape
    numel = q.numel()
    if numel == 0:
        return torch.empty_like(q, dtype=torch.float32)

    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        return (q.reshape(-1, block_size).float() * scales).view(shape)

    flat_q = torch.nn.functional.pad(q.reshape(-1), (0, pad))
    deq_blocks = flat_q.view(-1, block_size).float() * scales
    return deq_blocks.view(-1)[:numel].view(shape)


_V_DYNAMIC_RANGE_BITS = 32.0  # 2^-32 dynamic range ~= 2.3e-10 to 1.0 (over 9 orders of magnitude)
_V_SCALE_FACTOR = 254.0 / _V_DYNAMIC_RANGE_BITS
_V_INV_SCALE_FACTOR = _V_DYNAMIC_RANGE_BITS / 254.0
_V_MIN_RATIO = 2.0 ** (-_V_DYNAMIC_RANGE_BITS)

# Precomputed 256-element float32 lookup table for instant zero-overhead dequantization
_V_LOG_LUT = torch.zeros(256, dtype=torch.float32)
for _q in range(1, 256):
    _V_LOG_LUT[_q] = 2.0 ** ((_q - 1) * _V_INV_SCALE_FACTOR - _V_DYNAMIC_RANGE_BITS)


def _quantize_blockwise_uint8(
    tensor: Tensor,
    block_size: int = 256,
    out_q: Optional[Tensor] = None,
    out_scales: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Quantizes a non-negative float tensor (variance v) into block-wise logarithmic UINT8.

    Uses a 32-octave logarithmic dynamic range (over 9 orders of magnitude) with constant
    ~8.9% relative precision across all positive values. This guarantees that small variance
    components never underflow to zero or cause step-size explosions (g / sqrt(v + eps)),
    resolving the fundamental numerical instability that plagues 8-bit AdamW (bitsandbytes).
    """
    shape = tensor.shape
    numel = tensor.numel()
    if numel == 0:
        return torch.empty_like(tensor, dtype=torch.uint8), torch.empty((0, 1), device=tensor.device, dtype=torch.float32)

    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        blocks = tensor.reshape(-1, block_size).float()
    else:
        flat = torch.nn.functional.pad(tensor.reshape(-1), (0, pad))
        blocks = flat.view(-1, block_size).float()

    v_max = blocks.amax(dim=1, keepdim=True).clamp_min(1e-12)
    # Log-domain ratio mapping
    r = (blocks / v_max).clamp_min(_V_MIN_RATIO)
    log_r = torch.log2(r)
    q_blocks = torch.where(
        blocks <= 0,
        torch.zeros((), dtype=torch.uint8, device=tensor.device),
        torch.clamp((1.0 + (log_r + _V_DYNAMIC_RANGE_BITS) * _V_SCALE_FACTOR).round(), 1, 255).to(torch.uint8),
    )

    q = q_blocks.view(shape) if pad == 0 else q_blocks.view(-1)[:numel].view(shape)
    if out_q is not None:
        out_q.copy_(q)
    if out_scales is not None:
        out_scales.copy_(v_max)
    return (out_q if out_q is not None else q), (out_scales if out_scales is not None else v_max)


def _dequantize_blockwise_uint8(q: Tensor, scales: Tensor, block_size: int = 256) -> Tensor:
    """Dequantizes block-wise logarithmic UINT8 back into a float32 tensor matching q's shape."""
    shape = q.shape
    numel = q.numel()
    if numel == 0:
        return torch.empty_like(q, dtype=torch.float32)

    lut = _V_LOG_LUT.to(q.device)
    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        return (lut[q.reshape(-1, block_size).long()] * scales).view(shape)

    flat_q = torch.nn.functional.pad(q.reshape(-1), (0, pad))
    blocks = flat_q.view(-1, block_size).long()
    deq_blocks = lut[blocks] * scales
    return deq_blocks.view(-1)[:numel].view(shape)


def _is_peft_parameter(p: Tensor) -> bool:
    """Detects if a parameter belongs to a PEFT / LoRA adapter or quantized base model."""
    if getattr(p, "_is_lora", False):
        return True
    for attr in ("_param_name", "name"):
        val = getattr(p, attr, None)
        if isinstance(val, str) and any(k in val.lower() for k in ("lora_a", "lora_b", "lora_embedding", "adapter")):
            return True
    return False


def _detect_peft_context(param_groups: List[Dict[str, Any]]) -> bool:
    """Detects if optimizer is optimizing parameters within a PEFT / QLoRA model context."""
    for g in param_groups:
        params = g.get("params", [])
        for p in params:
            if isinstance(p, Tensor):
                if _is_peft_parameter(p):
                    return True
                if hasattr(p, "quant_state") or hasattr(p, "quant_type") or type(p).__name__ in ("Params4bit", "Linear4bit"):
                    return True
    return False


def _normalize_precision(precision: Union[str, torch.dtype]) -> str:
    """Normalize user-provided precision mode into '8bit', 'ultra', 'compact', 'auto', or 'full'.

    Accepts:
    - Standard strings: '8bit', 'ultra', 'compact', 'auto', 'full'
    - 8-bit aliases: '8bit', '8-bit', 'int8', 'uint8', 'q8', 'adamw_8bit', 'bnb_8bit' -> '8bit' (2 bytes/param)
    - 3-byte aliases: 'ultra', 'ultra_stable', '3byte', 'hybrid8' -> 'ultra' (block-wise INT8 m + 16-bit v, 3 bytes/param)
    - 16-bit aliases: 'compact', 'bf16', 'bfloat16', 'fp16', 'float16', 'half', '16', '16-bit' -> 'compact' (4 bytes/param)
    - FP32 aliases: 'full', 'fp32', 'float32', 'float', 'fp64', 'float64', 'double' -> 'full' (8 bytes/param)
    - PyTorch 8-bit dtypes: torch.int8, torch.uint8 -> '8bit'
    - PyTorch 16-bit dtypes: torch.bfloat16, torch.float16 -> 'compact'
    - PyTorch 32-bit dtypes: torch.float32, torch.float64 -> 'full'
    """
    if isinstance(precision, torch.dtype):
        if precision in (torch.float32, torch.float64):
            return "full"
        if precision in (torch.bfloat16, torch.float16):
            return "compact"
        if precision in (torch.int8, torch.uint8):
            return "8bit"
        return "full"
    if isinstance(precision, str):
        p_clean = precision.lower().strip()
        if p_clean in ("full", "fp32", "float32", "float", "fp64", "float64", "double"):
            return "full"
        if p_clean in ("compact", "bf16", "bfloat16", "fp16", "float16", "half", "16", "16-bit"):
            return "compact"
        if p_clean in ("8bit", "8-bit", "int8", "uint8", "q8", "adamw_8bit", "bnb_8bit"):
            return "8bit"
        if p_clean in ("ultra", "ultra_stable", "3byte", "hybrid8"):
            return "ultra"
        if p_clean in ("auto", "smart", "adaptive"):
            return "auto"
    raise ValueError(
        f"Invalid precision mode '{precision}'. Must be '8bit', 'ultra', 'compact', 'auto', 'full', "
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


def _get_v_dtype_for_beta2(compact_dtype: torch.dtype, beta2: float, force_compact_v: bool = False) -> torch.dtype:
    """Return the dtype to use for the second-moment (variance) buffer.

    When ``beta2 >= _BETA2_V_FP32_THRESHOLD`` and ``force_compact_v=False``, the per-step EMA contribution
    ``(1 - beta2) * g²`` falls below the bf16/fp16 quantisation step of the
    accumulated ``v_t``.  This causes ``v_t`` to stagnate — the EMA stops
    tracking gradient statistics — inflating the effective learning rate and
    destabilising long training runs.

    The fix is to store ``v`` (``exp_avg_sq``) in ``float32`` while keeping
    ``m`` (``exp_avg``) in the compact dtype.  This costs 2 extra bytes per
    parameter (6 vs 4 total) but remains 25 % cheaper than full AdamW (8 bytes).
    When ``force_compact_v=True``, this escalation is skipped and compact_dtype is preserved.
    """
    if force_compact_v:
        return compact_dtype
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
        "inactive_params_with_grad",
        "subgroups",
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
        # Triton path: pre-cached list of (scale_v, sqrt_scale_v, apply_wd) per param
        "scales_v_list",
        "fp16_indices",
        "apply_wd_list",
        "exp_avg_scales_list",
        "exp_avg_sq_scales_list",
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
        self.exp_avg_sqs = list(exp_avg_sqs) if bool(exp_avg_sqs) else None
        self.state_steps = list(state_steps)
        self.exp_avgs_slow = list(exp_avgs_slow) if bool(exp_avgs_slow) else None
        self.max_exp_avg_sqs = list(max_exp_avg_sqs) if bool(max_exp_avg_sqs) else None
        self.decay_norm_only = decay_norm_only
        self.extra = extra or {}
        self.param_ids = tuple(id(p) for p in self.params)

        # For fine-tuning, LoRA, and frozen backbones: track inactive params that have requires_grad
        group = self.extra.get("group", None)
        if group is not None:
            active_set = set(self.params)
            self.inactive_params_with_grad = [
                p for p in group.get("params", [])
                if getattr(p, "requires_grad", False) and p not in active_set
            ]
        else:
            self.inactive_params_with_grad = []

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
            self.subgroups = None
        elif path_type == "foreach":
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

            has_slow = bool(self.exp_avgs_slow)
            has_ams = bool(self.max_exp_avg_sqs)
            has_sq = bool(self.exp_avg_sqs)

            # Pre-partition parameters into execution subgroups
            subgroup_map = defaultdict(lambda: {
                "params": [], "indices": [], "exp_avgs": [],
                "exp_avg_sqs": [] if has_sq else None,
                "exp_avgs_slow": [] if has_slow else None,
                "max_exp_avg_sqs": [] if has_ams else None,
            })
            for i, p in enumerate(self.params):
                m = self.exp_avgs[i]
                v = self.exp_avg_sqs[i] if has_sq else None
                is_compact = m.dtype in (torch.bfloat16, torch.float16)
                apply_wd = (p.dim() >= 2) if decay_norm_only else True
                key = (p.device, is_compact, apply_wd)
                sg = subgroup_map[key]
                sg["params"].append(p)
                sg["indices"].append(i)
                sg["exp_avgs"].append(m)
                if has_sq:
                    sg["exp_avg_sqs"].append(v)
                if has_slow:
                    sg["exp_avgs_slow"].append(self.exp_avgs_slow[i])
                if has_ams:
                    sg["max_exp_avg_sqs"].append(self.max_exp_avg_sqs[i])

            self.subgroups = []
            for (dev, is_compact, apply_wd), sg in subgroup_map.items():
                self.subgroups.append({
                    "params": sg["params"],
                    "indices": sg["indices"],
                    "exp_avgs": sg["exp_avgs"],
                    "exp_avg_sqs": sg["exp_avg_sqs"],
                    "exp_avgs_slow": sg["exp_avgs_slow"],
                    "max_exp_avg_sqs": sg["max_exp_avg_sqs"],
                    "is_compact": is_compact,
                    "apply_wd": apply_wd,
                    "scale_v": 1.0,
                    "sqrt_scale_v": 1.0,
                })
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
            self.subgroups = None

        # Triton path caches: populated lazily on first execute_triton() call
        self.scales_v_list = None
        self.fp16_indices = [i for i, v in enumerate(self.exp_avg_sqs or []) if v.dtype == torch.float16]
        self.apply_wd_list = None
        self.exp_avg_scales_list = None
        self.exp_avg_sq_scales_list = None

    def is_valid_for_step(self, group: dict) -> bool:
        if len(self.params) == 0:
            return False
        for i, p in enumerate(self.params):
            if p.grad is None or id(p) != self.param_ids[i]:
                return False
        if self.inactive_params_with_grad:
            for p in self.inactive_params_with_grad:
                if p.grad is not None:
                    return False
        if group.get("amsgrad", False) != bool(self.max_exp_avg_sqs):
            return False
        if bool(group.get("beta3", None)) != bool(self.exp_avgs_slow):
            return False
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
        elif self.path_type in ("ultra", "8bit"):
            self.execute_ultra(opt, group, grad_scale, found_inf)
        elif self.path_type == "cuda":
            return self.execute_cuda(opt, group, grad_scale, found_inf)
        elif self.path_type == "foreach":
            self.execute_foreach(opt, group, grad_scale, found_inf)
        elif self.path_type == "single":
            self.execute_single(opt, group, grad_scale, found_inf)

    def execute_ultra(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        grads = [p.grad for p in self.params]
        opt._step_ultra(
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
            grad_scale=grad_scale,
            found_inf=found_inf,
        )

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
        fp16_indices = self.fp16_indices

        # Pre-cached scale_v and apply_wd aligned lists avoid per-step dict lookups.
        # Only rebuild on the rare 100-step dynamic-scaling recalibration boundary.
        needs_cache_rebuild = self.scales_v_list is None or self.apply_wd_list is None

        if dynamic_scaling and fp16_indices and step_val > 1 and step_val % 100 == 0:
            # Recalibrate only FP16 variance buffers; BF16 does not use range scaling.
            fp16_v = [self.exp_avg_sqs[i] for i in fp16_indices]
            v_max = max(v.max().item() for v in fp16_v)
            first_scale = opt.state[self.params[fp16_indices[0]]].get("scale_v", 1.0)
            factor = 1.0
            if v_max < 0.25 and first_scale < 65536.0:
                factor = 4.0
            elif v_max > 16384.0 and first_scale > 1.0:
                factor = 0.25
            if factor != 1.0:
                torch._foreach_mul_(fp16_v, factor)
                for i in fp16_indices:
                    p = self.params[i]
                    s = opt.state[p].get("scale_v", 1.0) * factor
                    opt.state[p]["scale_v"] = s
                    opt.state[p]["sqrt_scale_v"] = math.sqrt(s)
            needs_cache_rebuild = True

        if needs_cache_rebuild:
            # Build once and store aligned lists — no dicts, no per-step lookups
            if dynamic_scaling and fp16_indices:
                self.scales_v_list = [
                    opt.state[p].get("scale_v", 1.0)
                    if self.exp_avg_sqs[i].dtype == torch.float16 and p in opt.state
                    else 1.0
                    for i, p in enumerate(self.params)
                ]
            else:
                self.scales_v_list = None  # signals "all scale_v = 1.0" to the loop

            if wd != 0.0 and self.decay_norm_only:
                self.apply_wd_list = [p.dim() >= 2 for p in self.params]
            else:
                self.apply_wd_list = None  # signals "apply_wd same for all params"

        if self.exp_avg_scales_list is None:
            has_m_scales = any("exp_avg_scales" in opt.state[p] for p in self.params)
            self.exp_avg_scales_list = [opt.state[p].get("exp_avg_scales", None) for p in self.params] if has_m_scales else False
        if self.exp_avg_sq_scales_list is None:
            has_v_scales = any("exp_avg_sq_scales" in opt.state[p] for p in self.params)
            self.exp_avg_sq_scales_list = [opt.state[p].get("exp_avg_sq_scales", None) for p in self.params] if has_v_scales else False

        exp_avg_scales = self.exp_avg_scales_list if self.exp_avg_scales_list is not False else None
        exp_avg_sq_scales = self.exp_avg_sq_scales_list if self.exp_avg_sq_scales_list is not False else None
        bsize = 256 if (exp_avg_sq_scales is not None) else None

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
            block_size=bsize,
            exp_avgs_slow=self.exp_avgs_slow if beta3 is not None else None,
            beta3=beta3,
            alpha=alpha,
            cautious=cautious,
            skip_preflight=True,
            exp_avg_scales=exp_avg_scales,
            exp_avg_sq_scales=exp_avg_sq_scales,
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
        success = step_fused(
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
        if not success:
            torch._foreach_sub_(self.state_steps, 1)
        return success

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

        grads = [p.grad for p in self.params]

        # In-optimizer gradient centralization
        if group.get("grad_centralization", False):
            for g in grads:
                if g.dim() >= 2:
                    dims = tuple(range(1, g.dim()))
                    g.sub_(g.mean(dim=dims, keepdim=True))

        # In-optimizer gradient norm clipping
        clip_norm = group.get("clip_grad_norm")
        if clip_norm is not None:
            norms = torch._foreach_norm(grads, 2.0)
            dev = grads[0].device
            total_norm = torch.linalg.vector_norm(torch.stack([n.to(dev) for n in norms]), 2.0)
            clip_coef = clip_norm / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            if float(clip_coef_clamped.item()) < 1.0:
                torch._foreach_mul_(grads, clip_coef_clamped)

        # Fast path: single fused kernel launch when norm protection is not needed or wd is 0
        if not self.decay_norm_only or wd == 0.0 or not self.no_decay_params:
            if self.params:
                torch._fused_adamw_(
                    self.params,
                    grads,
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
            decay_grads = [p.grad for p in self.decay_params]
            no_decay_grads = [p.grad for p in self.no_decay_params]
            if self.decay_params:
                torch._fused_adamw_(
                    self.decay_params,
                    decay_grads,
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
                    no_decay_grads,
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
        if found_inf is not None:
            is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            if not is_capturing and not is_compiling():
                if found_inf.item() > 0:
                    return

        torch._foreach_add_(self.state_steps, 1)
        step = int(self.state_steps[0].item())

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        beta3 = group.get("beta3", None)
        alpha = group.get("alpha", 5.0)
        wd = group["weight_decay"]
        eps = group["eps"]
        lion = group.get("lion", False)
        nesterov = group.get("nesterov", False)
        cautious = group.get("cautious", False)
        amsgrad = group.get("amsgrad", False)
        maximize = group.get("maximize", False)

        bc1 = 1.0 - (beta1 ** step)
        bc2 = 1.0 - (beta2 ** step)
        bc2_sqrt = math.sqrt(bc2)
        step_size = lr * bc2_sqrt / bc1
        eps_scaled = eps * bc2_sqrt
        slow_ratio = alpha * (bc1 / (1.0 - (beta3 ** step))) if beta3 is not None else 0.0

        all_grads = [p.grad for p in self.params]
        if maximize:
            all_grads = torch._foreach_neg(all_grads)

        # In-optimizer gradient unscaling before norm clipping
        if grad_scale is not None and (group.get("clip_grad_norm") is not None or group.get("grad_centralization", False)):
            dev = all_grads[0].device
            scale_t = grad_scale.to(dev) if grad_scale.device != dev else grad_scale
            torch._foreach_mul_(all_grads, scale_t.reciprocal())
            grad_scale = None

        if group.get("grad_centralization", False):
            for g in all_grads:
                if g.dim() >= 2:
                    dims = tuple(range(1, g.dim()))
                    g.sub_(g.mean(dim=dims, keepdim=True))

        clip_norm = group.get("clip_grad_norm")
        if clip_norm is not None:
            norms = torch._foreach_norm(all_grads, 2.0)
            dev = all_grads[0].device
            total_norm = torch.linalg.vector_norm(torch.stack([n.to(dev) for n in norms]), 2.0)
            clip_coef = clip_norm / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            if float(clip_coef_clamped.item()) < 1.0:
                torch._foreach_mul_(all_grads, clip_coef_clamped)

        if grad_scale is not None:
            dev = all_grads[0].device
            scale_t = grad_scale.to(dev) if grad_scale.device != dev else grad_scale
            torch._foreach_mul_(all_grads, scale_t.reciprocal())

        if not self.subgroups or self.extra.get("has_complex", False):
            opt._multi_tensor_step(
                self.params, all_grads, self.exp_avgs,
                self.exp_avgs_slow if self.exp_avgs_slow else [],
                self.exp_avg_sqs if self.exp_avg_sqs else [],
                self.max_exp_avg_sqs if self.max_exp_avg_sqs else [],
                self.state_steps, lr=lr, beta1=beta1, beta2=beta2,
                beta3=beta3, alpha=alpha, lion=lion, eps=eps, weight_decay=wd,
                decay_norm_only=self.decay_norm_only, nesterov=nesterov,
                cautious=cautious, amsgrad=amsgrad, maximize=False,
                capturable=group["capturable"], precision=group["precision"],
                dynamic_scaling=group.get("dynamic_scaling", True),
                has_complex=self.extra.get("has_complex", False),
            )
            return

        # FP16 variance dynamic recalibration check (every 100 steps)
        if group.get("dynamic_scaling", True) and step > 1 and step % 100 == 0:
            for sg in self.subgroups:
                if sg["exp_avg_sqs"] and sg["exp_avg_sqs"][0].dtype == torch.float16:
                    v_max = max(v.max().item() for v in sg["exp_avg_sqs"])
                    cur_s = sg["scale_v"]
                    factor = 1.0
                    if v_max < 0.25 and cur_s < 65536.0:
                        factor = 4.0
                    elif v_max > 16384.0 and cur_s > 1.0:
                        factor = 0.25
                    if factor != 1.0:
                        sg["scale_v"] *= factor
                        sg["sqrt_scale_v"] = math.sqrt(sg["scale_v"])
                        torch._foreach_mul_(sg["exp_avg_sqs"], factor)

        for sg in self.subgroups:
            params = sg["params"]
            grads = [all_grads[idx] for idx in sg["indices"]]
            exp_avgs = sg["exp_avgs"]
            exp_avg_sqs = sg["exp_avg_sqs"]
            exp_avgs_slow = sg["exp_avgs_slow"]
            max_exp_avg_sqs = sg["max_exp_avg_sqs"]
            is_compact = sg["is_compact"]
            apply_wd = sg["apply_wd"]
            scale_v = sg["scale_v"]
            sqrt_scale_v = sg["sqrt_scale_v"]

            if wd != 0.0 and apply_wd:
                torch._foreach_mul_(params, 1.0 - lr * wd)

            if lion:
                c_grads = [g.to(exp_avgs[0].dtype) for g in grads] if grads[0].dtype != exp_avgs[0].dtype else grads
                updates = torch._foreach_lerp(c_grads, exp_avgs, beta1)
                updates = [u.sign() for u in updates]
                torch._foreach_add_(params, updates, alpha=-lr)
                torch._foreach_lerp_(exp_avgs, c_grads, 1.0 - beta2)
                continue

            if is_compact:
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
                if exp_avgs_slow is not None:
                    torch._foreach_mul_(exp_avgs_slow, beta3)
                    torch._foreach_add_(exp_avgs_slow, grads, alpha=1.0 - beta3)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=(1.0 - beta2) * scale_v)
            else:
                torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
                if exp_avgs_slow is not None:
                    torch._foreach_lerp_(exp_avgs_slow, grads, 1.0 - beta3)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=(1.0 - beta2) * scale_v)

            v_sources = exp_avg_sqs
            if amsgrad and max_exp_avg_sqs is not None:
                torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
                v_sources = max_exp_avg_sqs

            denoms = torch._foreach_sqrt(v_sources)
            if scale_v != 1.0:
                torch._foreach_add_(denoms, eps_scaled * sqrt_scale_v)
                torch._foreach_div_(denoms, sqrt_scale_v)
            else:
                torch._foreach_add_(denoms, eps_scaled)

            if nesterov and cautious:
                nums = [torch.lerp(g, m.to(g.dtype), beta1) for g, m in zip(grads, exp_avgs)]
                for idx in range(len(nums)):
                    mask = (grads[idx] * nums[idx] >= 0).to(params[idx].dtype)
                    mask_mean = mask.mean().clamp(min=1e-3)
                    mask.div_(mask_mean)
                    nums[idx] = nums[idx] * mask
                torch._foreach_addcdiv_(params, nums, denoms, value=-step_size)
            elif cautious:
                nums = list(exp_avgs)
                for idx in range(len(nums)):
                    mask = (grads[idx] * nums[idx] >= 0).to(params[idx].dtype)
                    mask_mean = mask.mean().clamp(min=1e-3)
                    mask.div_(mask_mean)
                    nums[idx] = nums[idx] * mask
                torch._foreach_addcdiv_(params, nums, denoms, value=-step_size)
            elif nesterov and beta3 is None:
                torch._foreach_addcdiv_(params, exp_avgs, denoms, value=-step_size * beta1)
                torch._foreach_addcdiv_(params, grads, denoms, value=-step_size * (1.0 - beta1))
            elif beta3 is not None and exp_avgs_slow is not None:
                nums = torch._foreach_add(exp_avgs, exp_avgs_slow, alpha=slow_ratio)
                torch._foreach_addcdiv_(params, nums, denoms, value=-step_size)
            else:
                torch._foreach_addcdiv_(params, exp_avgs, denoms, value=-step_size)


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
        dynamic_scaling (bool, optional): Whether to apply dynamic range scaling to FP16
            variance states (default: True). BF16 already has FP32-like exponent range.
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
            trainable_params = [p for p in first_arg.parameters() if p.requires_grad]
            if not trainable_params:
                trainable_params = list(first_arg.parameters())
            args = (trainable_params,) + args[1:]
            if "params" in kwargs:
                kwargs["params"] = trainable_params

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
        precision: Union[str, torch.dtype] = "compact",  # "compact" (bf16, default), "auto", "8bit" (2 bytes), "ultra" (3 bytes), "full" (fp32), or torch dtype
        threshold: int = 65536,      # Size threshold: 2D+ or >= 64K tensors get compact BF16, 1D norms get FP32
        maximize: bool = False,
        foreach: Optional[bool] = None,  # None = auto-detect: True on CUDA/XPU/ROCm, fast single on CPU
        capturable: bool = False,
        dynamic_scaling: bool = True, # Dynamic range scaling for FP16 variance states
        differentiable: bool = False,
        backend: Optional[str] = None, # None / "auto", "triton", "cuda", "mps", "foreach", "xla"
        auto_pack: bool = False,       # Automatically pack small tensors into contiguous flat buffers
        pack_threshold: int = 50000,   # Maximum numel for tensors to be auto-packed
        use_muon: bool = False,        # If True, routes 2D hidden weights to ApexMuon and 1D/embed to ApexW
        amp: Optional[bool] = None,    # Explicit mixed-precision indicator (True forces compact states for FP32 weights)
        stochastic_rounding: bool = True, # Unbiased stochastic rounding for 8-bit momentum quantization
        force_compact_v: bool = False,    # When True, prevents auto-escalation of v to FP32 even if beta2 >= 0.995
    ):
        precision = _normalize_precision(precision)
        if isinstance(params, nn.Module):
            if amp is None and any(p.dtype in (torch.float16, torch.bfloat16) for p in params.parameters()):
                amp = True
            trainable = [p for p in params.parameters() if p.requires_grad]
            params = trainable if trainable else list(params.parameters())
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
            amp=amp,
            stochastic_rounding=stochastic_rounding,
            force_compact_v=force_compact_v,
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

        # Warn if pure FP32 parameters on CUDA are detected in auto precision mode without AMP
        if (
            precision == "auto"
            and amp is not True
            and not _detect_peft_context(self.param_groups)
            and any(
                isinstance(p, torch.Tensor) and p.is_cuda and p.dtype == torch.float32
                for group in self.param_groups
                for p in group["params"]
            )
            and not any(
                isinstance(p, torch.Tensor) and p.dtype in (torch.float16, torch.bfloat16)
                for group in self.param_groups
                for p in group["params"]
            )
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
            group.setdefault("precision", "compact")
            group.setdefault("threshold", 65536)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("backend", None)
            group.setdefault("amp", None)
            group.setdefault("stochastic_rounding", True)
            group.setdefault("force_compact_v", False)
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
            # PyTorch's base Optimizer.load_state_dict forcefully casts all tensor state entries
            # to p.dtype. Here we restore ApexW's custom quantized (INT8) and compact dtypes.
            state_m_dt = s.get("state_dtype")
            state_v_dt = s.get("state_dtype_v")
            if s.get("is_8bit", False) or s.get("is_ultra", False) or state_m_dt == torch.int8:
                if "exp_avg" in s and s["exp_avg"].dtype != torch.int8:
                    s["exp_avg"] = s["exp_avg"].to(torch.int8)
                if "exp_avg_slow" in s and s["exp_avg_slow"].dtype != torch.int8:
                    s["exp_avg_slow"] = s["exp_avg_slow"].to(torch.int8)
                if "exp_avg_scales" in s and s["exp_avg_scales"].dtype != torch.float32:
                    s["exp_avg_scales"] = s["exp_avg_scales"].to(torch.float32)
            elif state_m_dt in (torch.bfloat16, torch.float16):
                if "exp_avg" in s and s["exp_avg"].dtype != state_m_dt:
                    s["exp_avg"] = s["exp_avg"].to(state_m_dt)
                if "exp_avg_slow" in s and s["exp_avg_slow"].dtype != state_m_dt:
                    s["exp_avg_slow"] = s["exp_avg_slow"].to(state_m_dt)

            if s.get("is_8bit", False) or state_v_dt == torch.uint8:
                if "exp_avg_sq" in s and s["exp_avg_sq"].dtype != torch.uint8:
                    s["exp_avg_sq"] = s["exp_avg_sq"].to(torch.uint8)
                if "exp_avg_sq_scales" in s and s["exp_avg_sq_scales"].dtype != torch.float32:
                    s["exp_avg_sq_scales"] = s["exp_avg_sq_scales"].to(torch.float32)
                if "max_exp_avg_sq" in s and s["max_exp_avg_sq"].dtype != torch.uint8:
                    s["max_exp_avg_sq"] = s["max_exp_avg_sq"].to(torch.uint8)
            elif state_v_dt in (torch.bfloat16, torch.float16):
                if "exp_avg_sq" in s and s["exp_avg_sq"].dtype != state_v_dt:
                    s["exp_avg_sq"] = s["exp_avg_sq"].to(state_v_dt)
                if "max_exp_avg_sq" in s and s["max_exp_avg_sq"].dtype != state_v_dt:
                    s["max_exp_avg_sq"] = s["max_exp_avg_sq"].to(state_v_dt)

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
        threshold = group.get("threshold", 65536)
        beta2 = group["betas"][1]
        group["precision"] = _normalize_precision(group.get("precision", "compact"))
        group_precision = group["precision"]

        # Cache whether any parameters in the model have 16-bit dtypes (computed once per group)
        _has_16bit_params = any(
            isinstance(param, torch.Tensor) and param.dtype in (torch.float16, torch.bfloat16)
            for g in self.param_groups
            for param in g.get("params", [])
        )

        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                if p.grad.is_sparse:
                    raise RuntimeError("ApexW does not support sparse gradients")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    fused = bool(group.get("fused", False))
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if (group["capturable"] or fused or p.device.type == "xla" or group.get("backend") == "xla")
                        else torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
                    )

                    is_peft = _is_peft_parameter(p) or _detect_peft_context(self.param_groups)
                    if group.get("fused", None) is True or group_precision == "full":
                        compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group_precision == "8bit":
                        is_massive = (p.dim() >= 2) or (p.numel() >= threshold)
                        if is_massive and not torch.is_complex(p) and p.dtype != torch.float64:
                            state["is_8bit"] = True
                            state["block_size_8bit"] = 256
                            compact_dtype = torch.int8
                        else:
                            state["is_8bit"] = False
                            compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group_precision == "auto":
                        # Smart Auto Partitioning:
                        # 2D+ tensors or tensors >= threshold receive compact 16-bit storage (50% VRAM cut).
                        # Small 1D tensors (LayerNorm, RMSNorm, biases) retain FP32 for zero drift.
                        _is_cuda_param = p.is_cuda
                        _has_16bit_grads = (p.grad.dtype in (torch.float16, torch.bfloat16))
                        # Robust AMP detection: don't rely only on autocast being active during step(),
                        # since optimizer.step() standardly runs outside the autocast context manager.
                        _is_amp_active = (
                            bool(group.get("amp", False))
                            or _has_16bit_params
                            or _has_16bit_grads
                            or torch.is_autocast_enabled()
                            or (hasattr(torch, "is_autocast_enabled") and torch.is_autocast_enabled("cuda"))
                            or p.dtype in (torch.float16, torch.bfloat16)
                            or is_peft
                        )
                        if _is_cuda_param and not _is_amp_active and p.dtype == torch.float32 and group.get("amp") is not True:
                            # Pure FP32 CUDA (no AMP) -> FP32 states for native fused kernel speed
                            compact_dtype = torch.float32
                        elif p.dtype == torch.float64:
                            compact_dtype = torch.float64
                        else:
                            is_massive = (p.dim() >= 2) or (p.numel() >= threshold)
                            if is_massive:
                                compact_dtype = _get_compact_dtype(p)
                            else:
                                compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group["precision"] == "ultra":
                        is_massive = (p.dim() >= 2) or (p.numel() >= threshold)
                        if is_massive and not torch.is_complex(p) and p.dtype != torch.float64:
                            state["is_ultra"] = True
                            state["ultra_block_size"] = 2048 if p.numel() >= 2048 else 256
                            compact_dtype = _get_compact_dtype(p)
                        else:
                            state["is_ultra"] = False
                            compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group["precision"] == "compact":
                        compact_dtype = _get_compact_dtype(p)
                    else:  # "full"
                        compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32

                    force_compact_v = group.get("force_compact_v", False)
                    if state.get("is_8bit", False):
                        state_dtype_m = torch.int8
                        state_dtype_v = torch.uint8
                    elif state.get("is_ultra", False):
                        state_dtype_m = torch.int8
                        state_dtype_v = _get_v_dtype_for_beta2(compact_dtype, beta2, force_compact_v=force_compact_v)
                    else:
                        state_dtype_m = compact_dtype
                        state_dtype_v = _get_v_dtype_for_beta2(compact_dtype, beta2, force_compact_v=force_compact_v)

                    # Record the per-moment dtypes for downstream code to use
                    state["state_dtype"] = state_dtype_m        # primary (backward-compat key)
                    state["state_dtype_v"] = state_dtype_v      # second-moment dtype

                    # Primary first-moment buffer
                    if state.get("is_8bit", False) or state.get("is_ultra", False):
                        bsize = state.get("block_size_8bit", state.get("ultra_block_size", 256))
                        q_m, scale_m = _quantize_blockwise_int8(
                            torch.zeros_like(p, dtype=torch.float32),
                            block_size=bsize,
                        )
                        state["exp_avg"] = q_m
                        state["exp_avg_scales"] = scale_m
                        if has_slow_momentum:
                            q_slow, scale_slow = _quantize_blockwise_int8(
                                torch.zeros_like(p, dtype=torch.float32),
                                block_size=bsize,
                            )
                            state["exp_avg_slow"] = q_slow
                            state["exp_avg_slow_scales"] = scale_slow
                    else:
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=state_dtype_m, memory_format=torch.preserve_format
                        )
                        if has_slow_momentum:
                            state["exp_avg_slow"] = torch.zeros_like(
                                p, dtype=state_dtype_m, memory_format=torch.preserve_format
                            )

                    # Second moment buffer (variance EMA, not needed in Lion mode)
                    if not is_lion:
                        if state.get("is_8bit", False):
                            bsize = state.get("block_size_8bit", 256)
                            q_v, scale_v = _quantize_blockwise_uint8(
                                torch.zeros_like(p, dtype=torch.float32),
                                block_size=bsize,
                            )
                            state["exp_avg_sq"] = q_v
                            state["exp_avg_sq_scales"] = scale_v
                        else:
                            state["exp_avg_sq"] = torch.zeros_like(
                                p, dtype=state_dtype_v, memory_format=torch.preserve_format
                            )
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, dtype=state_dtype_v, memory_format=torch.preserve_format
                            )

                    # FP16 needs range scaling; BF16 already has FP32's exponent range.
                    _is_xla = (p.device.type == "xla" or group.get("backend") == "xla")
                    if not _is_xla:
                        if group.get("dynamic_scaling", True) and not is_lion and state_dtype_v == torch.float16:
                            state["scale_v"] = 1024.0
                            state["sqrt_scale_v"] = 32.0
                        else:
                            state["scale_v"] = 1.0
                            state["sqrt_scale_v"] = 1.0

                # Defensive initialization if amsgrad or AdEMAMix was enabled after initial step
                state_dtype = state.get("state_dtype", p.dtype)
                state_dtype_v = state.get("state_dtype_v", state_dtype)
                if has_slow_momentum and "exp_avg_slow" not in state:
                    if state.get("is_ultra", False):
                        q_slow, scale_slow = _quantize_blockwise_int8(
                            torch.zeros_like(p, dtype=torch.float32),
                            block_size=state.get("ultra_block_size", 2048),
                        )
                        state["exp_avg_slow"] = q_slow
                        state["exp_avg_slow_scales"] = scale_slow
                    else:
                        state["exp_avg_slow"] = torch.zeros_like(
                            p, dtype=state_dtype, memory_format=torch.preserve_format
                        )
                if not is_lion and group["amsgrad"] and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, dtype=state_dtype_v, memory_format=torch.preserve_format
                    )
                _is_xla = (p.device.type == "xla" or group.get("backend") == "xla")
                if not _is_xla and "scale_v" not in state:
                    if group.get("dynamic_scaling", True) and not is_lion and state_dtype_v == torch.float16:
                        state["scale_v"] = 1024.0
                        state["sqrt_scale_v"] = 32.0
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
                if plan.execute(self, group, grad_scale, found_inf) is not False:
                    continue
                self._execution_plans.pop(group_id, None)

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

            # -------------------------------------------------------------
            # PATH 2 / PATH 3 viability evaluation
            # -------------------------------------------------------------
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
            if backend in (None, "auto") and can_use_triton and precision not in ("8bit", "ultra"):
                # For small tensors (e.g. CNNs, LoRA adapters, biases, norms), multi-tensor
                # Foreach executes in a single C++ kernel launch and is ~10-20x faster than
                # launching individual Triton kernels sequentially from Python.
                avg_elements = sum(p.numel() for p in params_with_grad) // len(params_with_grad)
                if avg_elements < 65536 and len(params_with_grad) > 8:
                    can_use_triton = False

            steps_are_uniform = len(state_steps) <= 1 or all(
                _get_value(s) == _get_value(state_steps[0]) for s in state_steps[1:]
            )

            # -------------------------------------------------------------
            # PATH 1B: Ultra / 8-Bit (Quantized Optimizer State Execution)
            # -------------------------------------------------------------
            # If precision is 8-bit or ultra and Triton GPU kernel is not viable
            # (e.g. CPU, MPS, or custom non-Triton backend), route to _step_ultra.
            if precision in ("8bit", "ultra") and not (can_use_triton and steps_are_uniform):
                self._step_ultra(
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
                    grad_scale=backend_grad_scale,
                    found_inf=found_inf,
                )
                if not is_capturing and not compiling:
                    self._execution_plans[group_id] = FastExecutionPlan(
                        path_type="ultra",
                        params=params_with_grad,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        state_steps=state_steps,
                        exp_avgs_slow=exp_avgs_slow if beta3 is not None else None,
                        max_exp_avg_sqs=max_exp_avg_sqs if amsgrad else None,
                        decay_norm_only=decay_norm_only,
                        extra={"group": group},
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
                exp_avg_sqs[i].dtype == torch.float16
                for i in range(len(exp_avg_sqs))
            )

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
                and (found_inf is None or (not is_capturing and not compiling))
                and not _states_have_mixed_moment_dtypes
                and not _dynamic_scaling_active
                and all(
                    p.device == params_with_grad[0].device
                    and g.device == p.device and m.device == p.device and v.device == p.device
                    and p.is_contiguous() and g.is_contiguous()
                    and m.is_contiguous() and v.is_contiguous()
                    and p.numel() == g.numel() == m.numel() == v.numel()
                    and g.dtype == p.dtype and m.dtype == v.dtype
                    and (p.dtype, m.dtype) in _CUDA_DTYPE_PAIRS
                    for p, g, m, v in zip(params_with_grad, grads, exp_avgs, exp_avg_sqs)
                )
                and is_cuda_kernel_available()
            )
            if can_use_cuda_fused:
                can_use_triton = False

            # Custom kernel preference: don't let PyTorch's native _fused_adamw_ preempt
            # custom CUDA or Triton implementations when custom kernels are requested or viable,
            # unless the user explicitly requested fused=True.
            custom_kernel_viable = (can_use_cuda_fused or can_use_triton)
            prefers_custom = (backend in ("cuda", "triton")) or (backend in (None, "auto") and custom_kernel_viable and fused is not True)

            # -------------------------------------------------------------
            # PATH 2A: PyTorch Native C++/CUDA Fused Multi-Tensor Fast Path
            # -------------------------------------------------------------
            # Used for full-precision CUDA/CPU when custom CUDA/Triton kernels are not active,
            # or when fused=True is explicitly requested.
            all_fp32 = all(p.dtype == torch.float32 for p in params_with_grad)
            states_fp32 = not _dynamic_scaling_active and not _states_have_mixed_moment_dtypes and all(
                exp_avgs[i].dtype == torch.float32 for i in range(len(exp_avgs))
            )
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
                not prefers_custom
                and (fused is True or (fused is None and (precision == "full" or _homogeneous_fast_path)))
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
                        extra={"group": group},
                    )
                continue

            # Only evaluate steps_are_uniform when Triton or custom CUDA is viable.
            steps_are_uniform = False

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
                    fp16_indices = [i for i, v in enumerate(exp_avg_sqs) if v.dtype == torch.float16]
                    # Dynamic recalibration check every 100 steps, limited to FP16 states.
                    if fp16_indices and step_val > 1 and step_val % 100 == 0:
                        fp16_v = [exp_avg_sqs[i] for i in fp16_indices]
                        v_max = max(v.max().item() for v in fp16_v)
                        first_scale = self.state[params_with_grad[fp16_indices[0]]].get("scale_v", 1.0)
                        factor = 1.0
                        if v_max < 0.25 and first_scale < 65536.0:
                            factor = 4.0
                        elif v_max > 16384.0 and first_scale > 1.0:
                            factor = 0.25
                        if factor != 1.0:
                            torch._foreach_mul_(fp16_v, factor)
                            for i in fp16_indices:
                                p = params_with_grad[i]
                                s = self.state[p].get("scale_v", 1.0) * factor
                                self.state[p]["scale_v"] = s
                                self.state[p]["sqrt_scale_v"] = math.sqrt(s)
                    triton_scales_v = {
                        id(p): self.state[p].get("scale_v", 1.0) if exp_avg_sqs[i].dtype == torch.float16 else 1.0
                        for i, p in enumerate(params_with_grad)
                        if p in self.state
                    }

                m_scales = [self.state[p].get("exp_avg_scales", None) for p in params_with_grad] if any("exp_avg_scales" in self.state[p] for p in params_with_grad) else None
                v_scales = [self.state[p].get("exp_avg_sq_scales", None) for p in params_with_grad] if any("exp_avg_sq_scales" in self.state[p] for p in params_with_grad) else None

                bsize_triton = 256 if (v_scales is not None) else None
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
                    block_size=bsize_triton,
                    exp_avgs_slow=exp_avgs_slow if beta3 is not None else None,
                    beta3=beta3,
                    alpha=alpha,
                    cautious=cautious,
                    exp_avg_scales=m_scales,
                    exp_avg_sq_scales=v_scales,
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
                            extra={"group": group},
                        )
                    continue
                # If Triton execution did not succeed, roll back step counter for fallback
                torch._foreach_sub_(state_steps, 1)


            # -------------------------------------------------------------
            # PATH 3: Custom C++/CUDA 128-bit Vectorized Fused Kernel
            # -------------------------------------------------------------
            # High-performance multi-tensor fused CUDA kernel fast path
            # (SM 7.0+ Volta/Turing/Ampere/Ada/Hopper)
            if can_use_cuda_fused and steps_are_uniform:
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
                            extra={"group": group},
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
                            extra={"has_complex": has_complex, "group": group},
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
                        extra={"has_complex": has_complex, "group": group},
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

    def _step_ultra(
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
        capturable: bool = False,
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
    ):
        """Ultra-compact 8-bit optimizer state execution path.

        Dynamically quantizes first-order momentum (m) into signed INT8 blocks with
        per-block float32 absolute maximum scales (block size 2048), while retaining
        second-order momentum (v) in compact float to guarantee exact numerical stability
        and loss convergence without 0-clamping or division explosions.
        """
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None and not is_capturing and not compiling:
            if found_inf.item() > 0:
                return

        inv_scale = None
        if grad_scale is not None:
            inv_scale = grad_scale.reciprocal()

        decay_factor = 1.0 - lr * weight_decay if weight_decay != 0 else 1.0

        sq_idx = 0
        for i, p in enumerate(params):
            g = grads[i] if not maximize else -grads[i]
            if inv_scale is not None:
                g = g * inv_scale

            step_t = state_steps[i]
            step_t.add_(1)
            step = _get_value(step_t)

            p_state = self.state[p]
            is_8bit = p_state.get("is_8bit", False)
            is_ultra = p_state.get("is_ultra", False)
            block_size = p_state.get("block_size_8bit", p_state.get("ultra_block_size", 256))
            stochastic = p_state.get("stochastic_rounding", True)

            # Decoupled weight decay
            if weight_decay != 0:
                if not decay_norm_only or p.dim() >= 2:
                    p.mul_(decay_factor)

            # Lion mode: only tracks first-order momentum
            if lion:
                if is_8bit or is_ultra:
                    m = _dequantize_blockwise_int8(p_state["exp_avg"], p_state["exp_avg_scales"], block_size=block_size)
                    g_f = g.float()
                    update = torch.lerp(g_f, m, beta1).sign()
                    p.add_(update.to(p.dtype), alpha=-lr)
                    m.lerp_(g_f, 1.0 - beta2)
                    _quantize_blockwise_int8(
                        m,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=p_state["exp_avg"],
                        out_scales=p_state["exp_avg_scales"],
                    )
                else:
                    m = p_state["exp_avg"]
                    g_c = g if g.dtype == m.dtype else g.to(m.dtype)
                    update = torch.lerp(g_c, m, beta1).sign()
                    p.add_(update.to(p.dtype), alpha=-lr)
                    m.lerp_(g_c, 1.0 - beta2)
                continue

            # AdamW / AdEMAMix path
            v = exp_avg_sqs[sq_idx]
            max_v = max_exp_avg_sqs[sq_idx] if amsgrad else None
            sq_idx += 1

            # Bias correction
            bc1 = 1.0 - (beta1 ** step)
            bc2 = 1.0 - (beta2 ** step)
            bc2_sqrt = math.sqrt(bc2)
            step_size = lr * bc2_sqrt / bc1
            eps_scaled = eps * bc2_sqrt

            g_f = g.float()

            if is_8bit:
                m = _dequantize_blockwise_int8(p_state["exp_avg"], p_state["exp_avg_scales"], block_size=block_size)
                m.lerp_(g_f, 1.0 - beta1)

                m_slow = None
                if beta3 is not None and "exp_avg_slow_scales" in p_state:
                    m_slow = _dequantize_blockwise_int8(
                        p_state["exp_avg_slow"], p_state["exp_avg_slow_scales"], block_size=block_size
                    )
                    m_slow.lerp_(g_f, 1.0 - beta3)

                v_val = _dequantize_blockwise_uint8(p_state["exp_avg_sq"], p_state["exp_avg_sq_scales"], block_size=block_size)
                v_val.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)

                if amsgrad:
                    torch.maximum(max_v, v_val, out=max_v)
                    v_active = max_v
                else:
                    v_active = v_val

                denom = v_active.sqrt().add_(eps_scaled)

                if m_slow is not None:
                    slow_ratio = alpha * (bc1 / (1.0 - (beta3 ** step)))
                    num = m.add(m_slow, alpha=slow_ratio)
                elif nesterov:
                    num = torch.lerp(g_f, m, beta1)
                else:
                    num = m

                if cautious:
                    mask = (num * g_f) > 0
                    mask_sum = mask.sum()
                    if mask_sum > 0:
                        scale = mask.numel() / mask_sum.clamp_min(1)
                        num = num * mask * scale

                update = (num / denom).mul_(step_size)
                p.sub_(update.to(p.dtype))

                # Requantize both moments back to 8-bit storage in-place
                _quantize_blockwise_int8(
                    m,
                    block_size=block_size,
                    stochastic=stochastic,
                    out_q=p_state["exp_avg"],
                    out_scales=p_state["exp_avg_scales"],
                )
                if m_slow is not None:
                    _quantize_blockwise_int8(
                        m_slow,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=p_state["exp_avg_slow"],
                        out_scales=p_state["exp_avg_slow_scales"],
                    )

                _quantize_blockwise_uint8(
                    v_val,
                    block_size=block_size,
                    out_q=p_state["exp_avg_sq"],
                    out_scales=p_state["exp_avg_sq_scales"],
                )
            elif is_ultra:
                m = _dequantize_blockwise_int8(p_state["exp_avg"], p_state["exp_avg_scales"], block_size=block_size)
                m.lerp_(g_f, 1.0 - beta1)

                m_slow = None
                if beta3 is not None and "exp_avg_slow_scales" in p_state:
                    m_slow = _dequantize_blockwise_int8(
                        p_state["exp_avg_slow"], p_state["exp_avg_slow_scales"], block_size=block_size
                    )
                    m_slow.lerp_(g_f, 1.0 - beta3)

                # Second-order momentum update evaluated in float32 arithmetic
                v_f = v.float().mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                v.copy_(v_f.to(v.dtype))

                if amsgrad:
                    torch.maximum(max_v, v, out=max_v)
                    v_active = max_v.float()
                else:
                    v_active = v_f

                denom = v_active.sqrt().add_(eps_scaled)

                # Numerator
                if m_slow is not None:
                    slow_ratio = alpha * (bc1 / (1.0 - (beta3 ** step)))
                    num = m.add(m_slow, alpha=slow_ratio)
                elif nesterov:
                    num = torch.lerp(g_f, m, beta1)
                else:
                    num = m

                if cautious:
                    mask = (num * g_f) > 0
                    mask_sum = mask.sum()
                    if mask_sum > 0:
                        scale = mask.numel() / mask_sum.clamp_min(1)
                        num = num * mask * scale

                update = (num / denom).mul_(step_size)
                p.sub_(update.to(p.dtype))

                # Requantize updated momentum back to INT8 storage in-place
                _quantize_blockwise_int8(
                    m,
                    block_size=block_size,
                    stochastic=stochastic,
                    out_q=p_state["exp_avg"],
                    out_scales=p_state["exp_avg_scales"],
                )
                if m_slow is not None:
                    _quantize_blockwise_int8(
                        m_slow,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=p_state["exp_avg_slow"],
                        out_scales=p_state["exp_avg_slow_scales"],
                    )
            else:
                # Fallback for 1D biases and small normalization tensors (< threshold)
                m = p_state["exp_avg"]
                m_slow = p_state.get("exp_avg_slow") if beta3 is not None else None

                m_f = m.float().lerp_(g_f, 1.0 - beta1)
                m.copy_(m_f.to(m.dtype))

                if m_slow is not None:
                    m_slow_f = m_slow.float().lerp_(g_f, 1.0 - beta3)
                    m_slow.copy_(m_slow_f.to(m_slow.dtype))

                v_f = v.float().mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                v.copy_(v_f.to(v.dtype))

                if amsgrad:
                    torch.maximum(max_v, v, out=max_v)
                    v_active = max_v.float()
                else:
                    v_active = v_f

                denom = v_active.sqrt().add_(eps_scaled)

                if m_slow is not None:
                    slow_ratio = alpha * (bc1 / (1.0 - (beta3 ** step)))
                    num = m_f.add(m_slow_f, alpha=slow_ratio)
                elif nesterov:
                    num = torch.lerp(g_f, m_f, beta1)
                else:
                    num = m_f

                if cautious:
                    mask = (num * g_f) > 0
                    mask_sum = mask.sum()
                    if mask_sum > 0:
                        scale = mask.numel() / mask_sum.clamp_min(1)
                        num = num * mask * scale

                update = (num / denom).mul_(step_size)
                p.sub_(update.to(p.dtype))

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
        for i, (p, step) in enumerate(zip(params, state_steps)):
            if step.device != p.device or step.dtype != torch.float32:
                step = step.to(device=p.device, dtype=torch.float32)
                self.state[p]["step"] = step
                state_steps[i] = step

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

            p_state = self.state[p]
            is_8bit = p_state.get("is_8bit", False)
            is_ultra = p_state.get("is_ultra", False)
            if is_8bit or is_ultra or m.dtype == torch.int8:
                block_size = p_state.get("block_size_8bit", p_state.get("ultra_block_size", 256))
                stochastic = p_state.get("stochastic_rounding", True)
                g_f = g.float()
                m_deq = _dequantize_blockwise_int8(m, p_state["exp_avg_scales"], block_size=block_size)
                m_deq.lerp_(g_f, 1.0 - beta1)

                m_slow_deq = None
                if m_slow is not None and "exp_avg_slow_scales" in p_state:
                    m_slow_deq = _dequantize_blockwise_int8(m_slow, p_state["exp_avg_slow_scales"], block_size=block_size)
                    m_slow_deq.lerp_(g_f, 1.0 - beta3)

                if is_8bit or (v is not None and v.dtype == torch.uint8):
                    v_deq = _dequantize_blockwise_uint8(v, p_state["exp_avg_sq_scales"], block_size=block_size)
                    v_deq.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                    if amsgrad:
                        torch.maximum(max_exp_avg_sqs[i], v_deq, out=max_exp_avg_sqs[i])
                        v_target = max_exp_avg_sqs[i]
                    else:
                        v_target = v_deq
                    denom = v_target.sqrt().add_(eps_scaled)
                    q_v, s_v = _quantize_blockwise_uint8(v_deq, block_size=block_size)
                    v.copy_(q_v)
                    p_state["exp_avg_sq_scales"].copy_(s_v)
                else:
                    v_f = v.float().mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                    v.copy_(v_f.to(v.dtype))
                    if amsgrad:
                        torch.maximum(max_exp_avg_sqs[i], v, out=max_exp_avg_sqs[i])
                        v_target = max_exp_avg_sqs[i].float()
                    else:
                        v_target = v_f
                    denom = v_target.sqrt().add_(eps_scaled)

                if m_slow_deq is not None:
                    num = m_deq.add(m_slow_deq, alpha=slow_ratio)
                elif nesterov:
                    num = torch.lerp(g_f, m_deq, beta1)
                else:
                    num = m_deq

                if cautious:
                    mask = (num * g_f) > 0
                    mask_sum = mask.sum()
                    if mask_sum > 0:
                        scale = mask.numel() / mask_sum.clamp_min(1)
                        num = num * mask * scale

                p.addcdiv_(num.to(p.dtype), denom.to(p.dtype), value=-step_size)

                q_m, s_m = _quantize_blockwise_int8(m_deq, block_size=block_size, stochastic=stochastic)
                m.copy_(q_m)
                p_state["exp_avg_scales"].copy_(s_m)
                if m_slow_deq is not None:
                    q_slow, s_slow = _quantize_blockwise_int8(m_slow_deq, block_size=block_size, stochastic=stochastic)
                    m_slow.copy_(q_slow)
                    p_state["exp_avg_slow_scales"].copy_(s_slow)
                continue

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

            # Dynamic range scaling is only needed for FP16 variance states
            scale_v = 1.0
            sqrt_scale_v = 1.0
            if dynamic_scaling and v is not None and v.dtype == torch.float16:
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
                and not (dynamic_scaling and d_exp_avg_sqs and d_exp_avg_sqs[0].dtype == torch.float16)
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

            # FP16 variance states use periodic range recalibration
            scale_v = 1.0
            sqrt_scale_v = 1.0
            if dynamic_scaling and not lion and target_dtype == torch.float16:
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
                # Only cast grads if required by _foreach_lerp (Lion or non-compact target dtype)
                if lion or target_dtype not in (torch.bfloat16, torch.float16):
                    c_grads_comp = [g.to(target_dtype) for g in c_grads] if c_grads[0].dtype != target_dtype else c_grads
                else:
                    c_grads_comp = None

                # 2. Lion Mode multi-tensor update
                if lion:
                    updates = torch._foreach_lerp(c_grads_comp, c_exp_avgs, beta1)
                    updates = [u.sign() for u in updates]
                    torch._foreach_add_(c_params, updates, alpha=-lr)
                    torch._foreach_lerp_(c_exp_avgs, c_grads_comp, 1.0 - beta2)
                    del updates, c_grads_comp
                    continue

                # Numerical Contract: FP32 arithmetic with compact (16-bit) state storage.
                # Zero-alloc cross-precision path — mixed-dtype _foreach_ ops update BF16/FP16
                # state tensors in-place directly from FP32 gradients without temporary allocations.
                if target_dtype in (torch.bfloat16, torch.float16):
                    # Upcast grads only when needed (e.g., AMP FP16 grads with BF16 states).
                    c_grads_f = c_grads if c_grads[0].dtype == torch.float32 else [g.float() for g in c_grads]

                    # Update first-order momentum in-place: m = m * beta1 + g * (1 - beta1)
                    torch._foreach_mul_(c_exp_avgs, beta1)
                    torch._foreach_add_(c_exp_avgs, c_grads_f, alpha=1.0 - beta1)

                    if beta3 is not None:
                        # Update slow EMA in-place with same zero-alloc pattern
                        torch._foreach_mul_(c_exp_avgs_slow, beta3)
                        torch._foreach_add_(c_exp_avgs_slow, c_grads_f, alpha=1.0 - beta3)

                    # Update second-order momentum in-place: v = v * beta2 + g^2 * ((1-beta2)*scale_v)
                    torch._foreach_mul_(c_exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(c_exp_avg_sqs, c_grads_f, c_grads_f, value=(1.0 - beta2) * scale_v)
                else:
                    # Update first-order momentum
                    # Use c_grads_comp (cast to target_dtype) so bf16 params with fp32 states work correctly.
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
                if c_grads_comp is not None and not cautious and not nesterov:
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
                if (cautious or nesterov) and c_grads_comp is not None:
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
