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
        step_fused_8bit,
    )
except ImportError:
    try:
        from cuda_extension import (
            is_cuda_kernel_available,
            fused_adamw_cuda_step,
            fused_multi_tensor_adamw_cuda_step,
            step_fused,
            step_fused_8bit,
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
        def step_fused_8bit(*args, **kwargs) -> bool:
            return False

try:
    from .triton_kernel import is_triton_available, triton_multi_tensor_adamw_step, triton_factored_v_step, is_rocm
except ImportError:
    try:
        from triton_kernel import is_triton_available, triton_multi_tensor_adamw_step, triton_factored_v_step, is_rocm
    except ImportError:
        def is_triton_available() -> bool:
            return False
        def triton_multi_tensor_adamw_step(*args, **kwargs) -> bool:
            return False
        def triton_factored_v_step(*args, **kwargs) -> bool:
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

from .quantization import (
    _NF4_M_CODEBOOK_RAW,
    _NF4_M_LUT_CACHE,
    _NF4_M_MIDPOINTS_RAW,
    _NF4_M_MIDS_CACHE,
    _NF4_PAIR_LUT_CACHE,
    _NF4_PAIR_LUT_RAW,
    _SR_POOL_CACHE,
    _SR_POOL_LEN,
    _V_DYNAMIC_RANGE_BITS,
    _V_INV_SCALE_FACTOR,
    _V_LOG_LUT,
    _V_LOG_LUT_CACHE,
    _V_MIN_RATIO,
    _V_SCALE_FACTOR,
    _copy_fp32_to_16bit_sr_,
    _dequantize_blockwise_6bit,
    _dequantize_blockwise_int4,
    _dequantize_blockwise_int8,
    _dequantize_blockwise_uint8,
    _get_nf4_m_lut,
    _get_nf4_m_mids,
    _get_nf4_pair_lut,
    _get_v_log_lut,
    _quantize_blockwise_6bit,
    _quantize_blockwise_int4,
    _quantize_blockwise_int8,
    _quantize_blockwise_uint8,
    _sample_sr_noise,
)
from .state import (
    CAPTURABLE_SUPPORTED_DEVICES,
    PackedStateProxy,
    StateDictWrapper,
    _BETA2_V_FP32_THRESHOLD,
    _BETA2_WARN_THRESHOLD,
    _CUDA_DTYPE_PAIRS,
    _CUDA_DTYPE_TRIOS,
    _FACTORED_CHUNK_ELEMS,
    _apply_factored_v_chunk_body,
    _apply_factored_v_int8_update,
    _can_use_factored_v_8bit,
    _can_use_factored_v_compact,
    _chunk_tensor_lists,
    _compute_factored_v_denom,
    _detect_peft_context,
    _execute_factored_v_batch,
    _get_compact_dtype,
    _get_scalar_dtype,
    _get_v_dtype_for_beta2,
    _get_value,
    _is_peft_parameter,
    _mps_supports_bfloat16,
    _normalize_precision,
    _partition_factored_v_indices,
    _prep_grad_chunk_fp32,
)


def _sanitize_gradients_fast(grads: List[Tensor]) -> List[Tensor]:
    """Fast vectorized gradient sanitization guard.
    Uses PyTorch native C++ non-finite checks to verify all gradients are finite
    in a single pass without per-tensor Python loops, scalar syncs, or allocations.
    Only if an actual NaN or Inf is detected across the tensors does it perform
    element-wise zeroing and clamping (preserving training stability without hurting speed).
    """
    if not grads:
        return grads

    dev = grads[0].device
    if (
        dev.type == "xla"
        or is_compiling()
        or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    ):
        return grads

    has_anomaly = False
    if dev.type == "cuda":
        try:
            chk = torch.zeros(1, device=dev)
            inv = torch.ones(1, device=dev)
            torch._amp_foreach_non_finite_check_and_unscale_(grads, chk, inv)
            if chk.item() > 0:
                has_anomaly = True
        except Exception:
            has_anomaly = not bool(
                torch.stack([torch.isfinite(g).all() for g in grads]).all().item()
            )
    elif dev.type in ("cpu", "mps"):
        try:
            if any(g.is_complex() or g.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) for g in grads):
                raise TypeError("fallback")
            # Inf-norm if float16 is present (avoids FP16 L2-norm overflow false positives);
            # otherwise L2-norm is fastest on AVX-512/NEON/Metal and catches NaN and Inf alike.
            has_fp16 = any(g.dtype == torch.float16 for g in grads)
            ord_val = float("inf") if has_fp16 else 2.0
            norms = torch._foreach_norm(grads, ord_val)
            # Quick Python-free reduction across the tiny scalar norms list
            has_anomaly = not bool(torch.isfinite(torch.stack([n.float() for n in norms])).all().item())
        except Exception:
            for g in grads:
                if not torch.isfinite(g).all().item():
                    has_anomaly = True
                    break
    else:
        for g in grads:
            if not torch.isfinite(g).all().item():
                has_anomaly = True
                break

    if not has_anomaly:
        return grads

    sanitized = []
    for g in grads:
        if not torch.isfinite(g).all():
            if torch.is_complex(g):
                g = g.clone().nan_to_num_(0.0)
            elif g.dtype in (torch.float16, torch.bfloat16):
                g = g.clone().clamp_(-65504.0, 65504.0).nan_to_num_(0.0)
            else:
                g = g.clone().nan_to_num_(0.0).clamp_(-65504.0, 65504.0)
        sanitized.append(g)
    return sanitized


def _prepare_grads(
    grads: List[Tensor],
    group: dict,
    grad_scale: Optional[Tensor] = None,
    *,
    sanitize: bool = True,
    force_unscale: bool = False,
) -> Tuple[List[Tensor], Optional[Tensor]]:
    """Preprocess gradients uniformly across cold-step and FastExecutionPlan paths.

    Runs optional NaN/Inf sanitization, unscales AMP grad_scale prior to
    centralization or clipping (or unconditionally when force_unscale=True),
    applies gradient centralization (ECCV 2020), and clips global L2 norm.
    Lazily clones before any in-place mutation so caller p.grad is never modified.
    """
    if not grads:
        return grads, grad_scale

    if sanitize:
        grads = _sanitize_gradients_fast(grads)

    clip_norm = group.get("clip_grad_norm", None)
    grad_cent = group.get("grad_centralization", False)
    needs_unscale = grad_scale is not None and (force_unscale or clip_norm is not None or grad_cent)

    if not needs_unscale and not grad_cent and clip_norm is None:
        return grads, grad_scale

    remaining_grad_scale = grad_scale
    if needs_unscale:
        first_dev = grads[0].device
        scale_t = grad_scale.to(first_dev) if grad_scale.device != first_dev else grad_scale
        torch._foreach_mul_(grads, scale_t.reciprocal())
        remaining_grad_scale = None

    if grad_cent:
        for g in grads:
            if g.dim() >= 2:
                dims = tuple(range(1, g.dim()))
                g.sub_(g.mean(dim=dims, keepdim=True))

    if clip_norm is not None:
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        norms = torch._foreach_norm(grads, 2.0)
        first_dev = grads[0].device
        total_norm = torch.linalg.vector_norm(
            torch.stack([n.to(device=first_dev, dtype=torch.float32) for n in norms]), 2.0
        )
        clip_coef = clip_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        if is_capturing or compiling or first_dev.type == "xla" or float(clip_coef_clamped.item()) < 1.0:
            torch._foreach_mul_(grads, clip_coef_clamped)

    return grads, remaining_grad_scale


def _dispatch_triton_with_factored(
    *,
    state: dict,
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avgs_slow: Optional[List[Tensor]],
    m_scales: Optional[List[Optional[Tensor]]],
    v_scales: Optional[List[Optional[Tensor]]],
    bsize: Optional[List[Optional[int]]],
    scales_v: Any,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    decay_norm_only: bool,
    inv_scale: float,
    beta3: Optional[float],
    alpha: float,
    cautious: bool,
    stochastic: bool,
    skip_preflight: bool = False,
    batch_cache: Optional[dict] = None,
    split_cache: Optional[tuple] = None,
) -> tuple[bool, tuple]:
    """Run Triton batched kernels on element-wise tensors and update factored-v tensors."""
    if split_cache is None:
        split_cache = _partition_factored_v_indices(state, params)
    elem_idxs, factored_idxs = split_cache

    if not factored_idxs:
        success = triton_multi_tensor_adamw_step(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            step=step,
            decay_norm_only=decay_norm_only,
            scales_v=scales_v,
            inv_scale=inv_scale,
            block_size=bsize,
            exp_avgs_slow=exp_avgs_slow,
            beta3=beta3,
            alpha=alpha,
            cautious=cautious,
            skip_preflight=skip_preflight,
            exp_avg_scales=m_scales,
            exp_avg_sq_scales=v_scales,
            stochastic=stochastic,
            batch_cache=batch_cache,
        )
        return success, split_cache

    if elem_idxs:
        e_scales_v = [scales_v[i] for i in elem_idxs] if isinstance(scales_v, list) else scales_v
        e_slow = [exp_avgs_slow[i] for i in elem_idxs] if exp_avgs_slow is not None else None
        e_m_scales = [m_scales[i] for i in elem_idxs] if m_scales is not None else None
        e_v_scales = [v_scales[i] for i in elem_idxs] if v_scales is not None else None
        e_bsize = [bsize[i] for i in elem_idxs] if bsize is not None else None
        success = triton_multi_tensor_adamw_step(
            [params[i] for i in elem_idxs],
            [grads[i] for i in elem_idxs],
            [exp_avgs[i] for i in elem_idxs],
            [exp_avg_sqs[i] for i in elem_idxs],
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            step=step,
            decay_norm_only=decay_norm_only,
            scales_v=e_scales_v,
            inv_scale=inv_scale,
            block_size=e_bsize,
            exp_avgs_slow=e_slow,
            beta3=beta3,
            alpha=alpha,
            cautious=cautious,
            skip_preflight=skip_preflight,
            exp_avg_scales=e_m_scales,
            exp_avg_sq_scales=e_v_scales,
            stochastic=stochastic,
            batch_cache=batch_cache,
        )
        if not success:
            return False, split_cache

    fallback_idxs: List[int] = []
    use_triton_factored = (beta3 is None) and is_triton_available()
    for i in factored_idxs:
        p = params[i]
        p_state = state[p]
        handled_f = False
        if use_triton_factored and p.is_cuda:
            apply_wd = weight_decay != 0.0 and (not decay_norm_only or p.dim() >= 2)
            handled_f = triton_factored_v_step(
                p,
                grads[i],
                p_state["exp_avg"],
                p_state["exp_avg_sq_row"],
                p_state["exp_avg_sq_col"],
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                step=step,
                apply_wd=apply_wd,
                inv_scale=inv_scale,
                cautious=cautious,
                stochastic=p_state.get("stochastic_rounding", stochastic),
                is_6bit_m=p_state.get("is_6bit_m", False),
                exp_avg_scales=p_state.get("exp_avg_scales", None),
                block_size=p_state.get("block_size_8bit", 256),
            )
        if not handled_f:
            fallback_idxs.append(i)

    if fallback_idxs:
        _execute_factored_v_batch(
            [params[i] for i in fallback_idxs],
            [grads[i] for i in fallback_idxs],
            state,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            step=step,
            decay_norm_only=decay_norm_only,
            inv_scale=inv_scale,
            cautious=cautious,
            stochastic=stochastic,
        )
    return True, split_cache


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
        "decay_indices",
        "no_decay_indices",
        # Triton path: pre-cached list of (scale_v, sqrt_scale_v, apply_wd) per param
        "scales_v_list",
        "fp16_indices",
        "apply_wd_list",
        "exp_avg_scales_list",
        "exp_avg_sq_scales_list",
        "block_size_list",
        "_triton_batch_cache",
        "_cuda_8bit_partition",
        "_factored_split",
        "factored_entries",
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

        # Initialize path-specific fast execution attributes
        self.decay_params = self.decay_m = self.decay_v = self.decay_max_v = self.decay_steps = None
        self.no_decay_params = self.no_decay_m = self.no_decay_v = self.no_decay_max_v = self.no_decay_steps = None
        self.decay_indices = self.no_decay_indices = None
        self.subgroups = None
        self.factored_entries = None

        if path_type == "native_fused":
            decay_indices = [i for i, p in enumerate(self.params) if p.dim() >= 2 or not decay_norm_only]
            no_decay_indices = [i for i, p in enumerate(self.params) if p.dim() < 2 and decay_norm_only]

            self.decay_indices = decay_indices
            self.no_decay_indices = no_decay_indices
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
        elif path_type == "foreach":
            has_slow = bool(self.exp_avgs_slow)
            has_ams = bool(self.max_exp_avg_sqs)
            has_sq = bool(self.exp_avg_sqs)
            factored_mask = self.extra.get("factored_mask", None)
            self.factored_entries = []

            # Pre-partition parameters into unified execution subgroups (by device and compactness)
            subgroup_map = defaultdict(lambda: {
                "params": [], "indices": [], "exp_avgs": [],
                "exp_avg_sqs": [] if has_sq else None,
                "exp_avgs_slow": [] if has_slow else None,
                "max_exp_avg_sqs": [] if has_ams else None,
                "decay_params": [],
            })
            for i, p in enumerate(self.params):
                apply_wd = (p.dim() >= 2) if decay_norm_only else True
                if factored_mask is not None and i < len(factored_mask) and factored_mask[i]:
                    self.factored_entries.append((p, i, apply_wd))
                    continue
                m = self.exp_avgs[i]
                v = self.exp_avg_sqs[i] if has_sq else None
                is_compact = m.dtype in (torch.bfloat16, torch.float16)
                is_quad_v = bool(v is not None and p.numel() >= 4 and p.numel() % 4 == 0 and v.numel() * 4 == p.numel())
                key = (p.device, p.dtype, is_compact, is_quad_v)
                sg = subgroup_map[key]
                sg["params"].append(p)
                sg["indices"].append(i)
                sg["exp_avgs"].append(m)
                if apply_wd:
                    sg["decay_params"].append(p)
                if has_sq:
                    sg["exp_avg_sqs"].append(v)
                if has_slow:
                    sg["exp_avgs_slow"].append(self.exp_avgs_slow[i])
                if has_ams:
                    sg["max_exp_avg_sqs"].append(self.max_exp_avg_sqs[i])

            self.subgroups = []
            for (dev, p_dtype, is_compact, is_quad_v), sg in subgroup_map.items():
                self.subgroups.append({
                    "params": sg["params"],
                    "indices": sg["indices"],
                    "exp_avgs": sg["exp_avgs"],
                    "exp_avg_sqs": sg["exp_avg_sqs"],
                    "exp_avgs_slow": sg["exp_avgs_slow"],
                    "max_exp_avg_sqs": sg["max_exp_avg_sqs"],
                    "decay_params": sg["decay_params"],
                    "is_compact": is_compact,
                    "is_quad_v": is_quad_v,
                    "dtype": p_dtype,
                    "apply_wd": True,
                    "scale_v": 1.0,
                    "sqrt_scale_v": 1.0,
                })

        # Triton / 8-bit fast-path caches: populated lazily on first execution step
        self.scales_v_list = None
        self.fp16_indices = [i for i, v in enumerate(self.exp_avg_sqs or []) if v.dtype == torch.float16]
        self.apply_wd_list = None
        self.exp_avg_scales_list = None
        self.exp_avg_sq_scales_list = None
        self.block_size_list = None
        self._triton_batch_cache: Dict[Any, Any] = {}
        self._cuda_8bit_partition = None
        self._factored_split = None

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
        if group.get("decay_norm_only", self.decay_norm_only) != self.decay_norm_only:
            return False
        if "precision" in group and "precision" in self.extra and group["precision"] != self.extra["precision"]:
            return False
        if self.path_type in ("native_fused", "cuda"):
            if (
                group.get("nesterov", False)
                or group.get("lion", False)
                or group.get("cautious", False)
                or group.get("capturable", False)
                or group.get("maximize", False)
            ):
                return False
        elif self.path_type == "cuda_8bit":
            if (
                group.get("nesterov", False)
                or group.get("lion", False)
                or group.get("amsgrad", False)
                or group.get("capturable", False)
                or group.get("maximize", False)
                or group.get("beta3", None) is not None
            ):
                return False
        elif self.path_type == "triton":
            if (
                group.get("nesterov", False)
                or group.get("lion", False)
                or group.get("amsgrad", False)
                or group.get("capturable", False)
                or group.get("maximize", False)
            ):
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
            return self.execute_triton(opt, group, grad_scale, found_inf)
        elif self.path_type == "cuda_8bit":
            return self.execute_cuda_8bit(opt, group, grad_scale, found_inf)
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
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None and not is_capturing and not compiling:
            if found_inf.item() > 0:
                return
        grads = [p.grad for p in self.params]
        grads, remaining_grad_scale = _prepare_grads(grads, group, grad_scale, sanitize=False)
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
            grad_scale=remaining_grad_scale,
            found_inf=found_inf,
        )

    def execute_triton(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None:
            if is_capturing or compiling:
                return False
            if found_inf.item() > 0:
                return True

        grads = [p.grad for p in self.params]
        # Triton kernels sanitize NaN/Inf gradients in FP32 registers; only run pre-sanitize
        # when a global gradient reduction (clip_grad_norm or grad_centralization) will read g.
        needs_host_sanitize = (
            group.get("clip_grad_norm", None) is not None
            or group.get("grad_centralization", False)
        )
        grads, remaining_grad_scale = _prepare_grads(
            grads, group, grad_scale, sanitize=needs_host_sanitize
        )

        inv_scale = 1.0
        if remaining_grad_scale is not None:
            if is_capturing or compiling:
                return False
            inv_scale = float(remaining_grad_scale.reciprocal().item()) if hasattr(remaining_grad_scale, "item") else (1.0 / float(remaining_grad_scale))

        step_val = int(_get_value(self.state_steps[0])) + 1
        torch._foreach_add_(self.state_steps, 1)

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
        if self.block_size_list is None:
            if exp_avg_scales is not None or exp_avg_sq_scales is not None:
                self.block_size_list = [
                    opt.state[p].get("block_size_8bit", opt.state[p].get("ultra_block_size", None))
                    for p in self.params
                ]
            else:
                self.block_size_list = False
        bsize = self.block_size_list if self.block_size_list is not False else None

        success, self._factored_split = _dispatch_triton_with_factored(
            state=opt.state,
            params=self.params,
            grads=grads,
            exp_avgs=self.exp_avgs,
            exp_avg_sqs=self.exp_avg_sqs,
            exp_avgs_slow=self.exp_avgs_slow if beta3 is not None else None,
            m_scales=exp_avg_scales,
            v_scales=exp_avg_sq_scales,
            bsize=bsize,
            scales_v=self.scales_v_list,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=wd,
            step=step_val,
            decay_norm_only=self.decay_norm_only,
            inv_scale=inv_scale,
            beta3=beta3,
            alpha=alpha,
            cautious=cautious,
            stochastic=group.get("stochastic_rounding", True),
            skip_preflight=True,
            batch_cache=self._triton_batch_cache,
            split_cache=self._factored_split,
        )
        if not success:
            torch._foreach_sub_(self.state_steps, 1)
        return success

    def execute_cuda_8bit(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None:
            if is_capturing or compiling:
                return False
            if found_inf.item() > 0:
                return True

        grads = [p.grad for p in self.params]
        needs_host_sanitize = (
            group.get("clip_grad_norm", None) is not None
            or group.get("grad_centralization", False)
        )
        grads, remaining_grad_scale = _prepare_grads(
            grads, group, grad_scale, sanitize=needs_host_sanitize
        )

        inv_scale = 1.0
        if remaining_grad_scale is not None:
            if is_capturing or compiling:
                return False
            inv_scale = float(remaining_grad_scale.reciprocal().item()) if hasattr(remaining_grad_scale, "item") else (1.0 / float(remaining_grad_scale))

        step_val = int(_get_value(self.state_steps[0])) + 1
        torch._foreach_add_(self.state_steps, 1)

        if self._cuda_8bit_partition is None:
            self._cuda_8bit_partition = opt._build_cuda_8bit_partition(
                self.params, self.exp_avgs, self.exp_avg_sqs
            )

        success = opt._execute_cuda_8bit_partition(
            self._cuda_8bit_partition,
            self.params,
            grads,
            self.exp_avgs,
            self.exp_avg_sqs,
            lr=group["lr"],
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            eps=group["eps"],
            weight_decay=group["weight_decay"],
            step=step_val,
            decay_norm_only=self.decay_norm_only,
            inv_scale=inv_scale,
            stochastic=group.get("stochastic_rounding", True),
            cautious=group.get("cautious", False),
        )
        if not success:
            torch._foreach_sub_(self.state_steps, 1)
        return success

    def execute_cuda(
        self,
        opt,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None:
            if is_capturing or compiling:
                return False
            if found_inf.item() > 0:
                return True

        grads = [p.grad for p in self.params]
        needs_host_sanitize = (
            group.get("clip_grad_norm", None) is not None
            or group.get("grad_centralization", False)
        )
        grads, _ = _prepare_grads(
            grads,
            group,
            grad_scale,
            sanitize=needs_host_sanitize,
            force_unscale=(grad_scale is not None),
        )

        step_val = int(_get_value(self.state_steps[0])) + 1
        torch._foreach_add_(self.state_steps, 1)

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        eps = group["eps"]

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
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None and not is_capturing and not compiling:
            if found_inf.item() > 0:
                return
        grads = [p.grad for p in self.params]
        grads, remaining_grad_scale = _prepare_grads(grads, group, grad_scale, sanitize=False)
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
            grad_scale=remaining_grad_scale,
            found_inf=found_inf,
        )

    def execute_native_fused(
        self,
        group: dict,
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
    ):
        device = self.params[0].device
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        eps = group["eps"]
        amsgrad = group["amsgrad"]
        maximize = group["maximize"]

        grads = [p.grad for p in self.params]
        needs_host_sanitize = (
            group.get("clip_grad_norm", None) is not None
            or group.get("grad_centralization", False)
        )
        grads, remaining_grad_scale = _prepare_grads(grads, group, grad_scale, sanitize=needs_host_sanitize)

        device_grad_scale = (
            remaining_grad_scale.to(device, non_blocking=True)
            if remaining_grad_scale is not None
            else None
        )
        device_found_inf = (
            found_inf.to(device, non_blocking=True) if found_inf is not None else None
        )
        torch._foreach_add_(self.state_steps, 1)

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
            decay_grads = [grads[i] for i in self.decay_indices]
            no_decay_grads = [grads[i] for i in self.no_decay_indices]
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
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None and not is_capturing and not compiling:
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

        has_factored = bool(getattr(self, "factored_entries", None))
        use_subgroups = (bool(self.subgroups) or has_factored) and not self.extra.get("has_complex", False)
        needs_host_sanitize = (
            group.get("clip_grad_norm", None) is not None
            or group.get("grad_centralization", False)
        )
        all_grads, remaining_grad_scale = _prepare_grads(
            all_grads,
            group,
            grad_scale,
            sanitize=needs_host_sanitize,
            force_unscale=(use_subgroups and grad_scale is not None),
        )

        if not use_subgroups:
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
                grad_scale=remaining_grad_scale,
                found_inf=None,
                stochastic=group.get("stochastic_rounding", True),
            )
            return

        if has_factored:
            wd_factor_base = max(1.0 - lr * wd, 0.0) if wd != 0.0 else 1.0
            stochastic = bool(group.get("stochastic_rounding", True))
            for p, idx, apply_wd in self.factored_entries:
                p_st = opt.state[p]
                eff_eps = eps
                _apply_factored_v_int8_update(
                    p,
                    all_grads[idx],
                    p_st,
                    beta1=beta1,
                    beta2=beta2,
                    step_size=step_size,
                    eps_scaled=eff_eps * bc2_sqrt,
                    effective_eps=eff_eps,
                    nesterov=nesterov,
                    cautious=cautious,
                    stochastic=p_st.get("stochastic_rounding", stochastic),
                    block_size=p_st.get("block_size_8bit", 256),
                    wd_factor=wd_factor_base if (wd != 0.0 and apply_wd) else None,
                    inv_scale=remaining_grad_scale,
                    sanitize_g=False,
                )

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
            first_p = params[0]
            first_p_st = opt.state.get(first_p, {})
            stochastic = bool(first_p_st.get("stochastic_rounding", group.get("stochastic_rounding", True)))
            if first_p_st.get("is_8bit", False) or first_p_st.get("is_ultra", False):
                scale_v = 1.0
                sqrt_scale_v = 1.0
            else:
                scale_v = first_p_st.get("scale_v", sg.get("scale_v", 1.0))
                sqrt_scale_v = first_p_st.get("sqrt_scale_v", math.sqrt(scale_v))
            sg["scale_v"] = scale_v
            sg["sqrt_scale_v"] = sqrt_scale_v

            has_fp16 = (first_p.dtype == torch.float16) or (exp_avgs and exp_avgs[0].dtype == torch.float16) or (exp_avg_sqs and exp_avg_sqs[0].dtype == torch.float16)
            effective_eps = eps
            sg_eps_scaled = effective_eps * bc2_sqrt

            if wd != 0.0:
                decay_p = sg.get("decay_params")
                if decay_p is not None:
                    if len(decay_p) > 0:
                        torch._foreach_mul_(decay_p, 1.0 - lr * wd)
                elif apply_wd:
                    torch._foreach_mul_(params, 1.0 - lr * wd)

            if lion:
                if grads[0].dtype == exp_avgs[0].dtype:
                    updates = torch._foreach_lerp(grads, exp_avgs, beta1)
                    updates = [u.sign() for u in updates]
                    torch._foreach_add_(params, updates, alpha=-lr)
                    torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta2)
                else:
                    c_grads = [g.to(exp_avgs[0].dtype) for g in grads]
                    updates = torch._foreach_lerp(c_grads, exp_avgs, beta1)
                    updates = [u.sign() for u in updates]
                    torch._foreach_add_(params, updates, alpha=-lr)
                    torch._foreach_lerp_(exp_avgs, c_grads, 1.0 - beta2)
                continue

            state_dt = exp_avgs[0].dtype
            if grads[0].dtype == state_dt:
                torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
                if exp_avgs_slow is not None:
                    torch._foreach_lerp_(exp_avgs_slow, grads, 1.0 - beta3)
            else:
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
                if exp_avgs_slow is not None:
                    torch._foreach_mul_(exp_avgs_slow, beta3)
                    torch._foreach_add_(exp_avgs_slow, grads, alpha=1.0 - beta3)

            if sg.get("is_quad_v", False):
                v_alpha = (1.0 - beta2) * scale_v
                grads_f = grads if grads[0].dtype == torch.float32 else [g.float() for g in grads]
                g_sq_qs = [gf.reshape(-1, 4).square().mean(dim=1) for gf in grads_f]
                if beta2 >= 0.995 and stochastic and exp_avg_sqs and exp_avg_sqs[0].dtype in (torch.bfloat16, torch.float16):
                    for v_i, gq in zip(exp_avg_sqs, g_sq_qs):
                        v_next = v_i.float().mul_(beta2).add_(gq, alpha=v_alpha)
                        _copy_fp32_to_16bit_sr_(v_i, v_next, stochastic=True, in_place_src=True)
                else:
                    torch._foreach_mul_(exp_avg_sqs, beta2)
                    torch._foreach_add_(exp_avg_sqs, g_sq_qs, alpha=v_alpha)
                del g_sq_qs

                v_sources = exp_avg_sqs
                if amsgrad and max_exp_avg_sqs is not None:
                    torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
                    v_sources = max_exp_avg_sqs

                denoms_q = torch._foreach_sqrt(v_sources)
                if scale_v != 1.0:
                    torch._foreach_add_(denoms_q, sg_eps_scaled * sqrt_scale_v)
                    torch._foreach_div_(denoms_q, sqrt_scale_v)
                else:
                    torch._foreach_add_(denoms_q, sg_eps_scaled)
                if has_fp16:
                    torch._foreach_clamp_min_(denoms_q, effective_eps)
                for idx_q, (p_i, g_f, m_i, denom_q) in enumerate(zip(params, grads_f, exp_avgs, denoms_q)):
                    m_f = m_i.float() if m_i.dtype != torch.float32 else m_i
                    if exp_avgs_slow is not None and len(exp_avgs_slow) > idx_q:
                        ms_i = exp_avgs_slow[idx_q]
                        ms_f = ms_i.float() if ms_i.dtype != torch.float32 else ms_i
                        num = m_f.add(ms_f, alpha=slow_ratio)
                    elif cautious:
                        mask = (g_f * m_f >= 0).to(torch.float32)
                        mask_mean = mask.mean().clamp(min=1e-3)
                        num = m_f.mul(mask.div_(mask_mean))
                    elif nesterov:
                        num = torch.lerp(g_f, m_f, beta1)
                    else:
                        num = m_f
                    dq_f = denom_q.float() if denom_q.dtype != torch.float32 else denom_q
                    step_delta = num.view(-1, 4).div_(dq_f.unsqueeze(1)).mul_(step_size).view(p_i.shape)
                    if p_i.dtype != torch.float32:
                        _copy_fp32_to_16bit_sr_(p_i, p_i.float().sub_(step_delta), stochastic=stochastic, in_place_src=True)
                    else:
                        p_i.sub_(step_delta)
                del denoms_q
                continue

            if beta2 >= 0.995 and stochastic and exp_avg_sqs and exp_avg_sqs[0].dtype in (torch.bfloat16, torch.float16):
                for v_i, g_i in zip(exp_avg_sqs, grads):
                    g_f = g_i.float()
                    v_next = v_i.float().mul_(beta2).addcmul_(g_f, g_f, value=(1.0 - beta2) * scale_v)
                    _copy_fp32_to_16bit_sr_(v_i, v_next, stochastic=True, in_place_src=True)
            else:
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=(1.0 - beta2) * scale_v)

            v_sources = exp_avg_sqs
            if amsgrad and max_exp_avg_sqs is not None:
                torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
                v_sources = max_exp_avg_sqs

            denoms = torch._foreach_sqrt(v_sources)
            if scale_v != 1.0:
                torch._foreach_add_(denoms, sg_eps_scaled * sqrt_scale_v)
                torch._foreach_div_(denoms, sqrt_scale_v)
            else:
                torch._foreach_add_(denoms, sg_eps_scaled)

            if has_fp16:
                torch._foreach_clamp_min_(denoms, effective_eps)

            if nesterov and cautious:
                nums = [torch.lerp(g, m.to(g.dtype), beta1) for g, m in zip(grads, exp_avgs)]
                for idx in range(len(nums)):
                    mask = (grads[idx] * nums[idx] >= 0).to(params[idx].dtype)
                    mask_mean = mask.mean().clamp(min=1e-3)
                    mask.div_(mask_mean)
                    nums[idx] = nums[idx] * mask
                torch._foreach_addcdiv_(params, nums, denoms, value=-step_size)
            elif cautious:
                nums = [m.to(g.dtype) if m.dtype != g.dtype else m for g, m in zip(grads, exp_avgs)]
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
                if params and params[0].dtype in (torch.bfloat16, torch.float16):
                    for p_i, m_i, denom_i in zip(params, exp_avgs, denoms):
                        step_delta = (m_i.float() / denom_i.float()) * step_size
                        _copy_fp32_to_16bit_sr_(p_i, p_i.float().sub_(step_delta), stochastic=stochastic, in_place_src=True)
                else:
                    torch._foreach_addcdiv_(params, exp_avgs, denoms, value=-step_size)


import weakref as _weakref

_ACTIVE_DOWNCAST_PARAMS: Dict[int, Tuple[Any, torch.dtype]] = {}
_ACTIVE_8BIT_OPTIMIZERS: List[Any] = []
_AMP_SCALED_THIS_BACKWARD: bool = False
_LAST_ACTIVE_AMP_SCALE: Optional[float] = None
_APEXW_FP16_ACTIVE: bool = False
_RUNTIME_HOOKS_INSTALLED: bool = False


def _restore_all_downcast_params(keep_states: bool = True) -> None:
    """Restore any 16-bit downcast parameters back to their original dtype and remove auto backward hooks."""
    global _APEXW_FP16_ACTIVE, _AMP_SCALED_THIS_BACKWARD
    _APEXW_FP16_ACTIVE = False
    _AMP_SCALED_THIS_BACKWARD = False

    for opt_ref in list(_ACTIVE_8BIT_OPTIMIZERS):
        opt = opt_ref()
        if opt is not None:
            try:
                opt._disable_auto_8bit_hooks()
            except Exception:
                pass
    _ACTIVE_8BIT_OPTIMIZERS.clear()

    if not _ACTIVE_DOWNCAST_PARAMS:
        return

    with torch.no_grad():
        for p_id, (p_ref, orig_dtype) in list(_ACTIVE_DOWNCAST_PARAMS.items()):
            p = p_ref()
            if p is not None and isinstance(p, torch.Tensor) and p.dtype != orig_dtype:
                try:
                    p.data = p.data.to(orig_dtype)
                    if p.grad is not None and p.grad.dtype != orig_dtype:
                        p.grad = p.grad.to(orig_dtype)
                except Exception:
                    pass
    _ACTIVE_DOWNCAST_PARAMS.clear()


def _install_runtime_hooks() -> None:
    """Install transparent runtime hooks for AMP GradScaler FP16 support and multi-run parameter restoration."""
    global _RUNTIME_HOOKS_INSTALLED
    if _RUNTIME_HOOKS_INSTALLED:
        return
    _RUNTIME_HOOKS_INSTALLED = True

    # 1. Patch GradScaler._unscale_grads_, .scale, and .update for seamless FP16/BF16 8-bit training
    try:
        from torch.amp.grad_scaler import GradScaler as _AmpGradScaler

        if not getattr(_AmpGradScaler, "_apexw_patched", False):
            _orig_unscale_grads = _AmpGradScaler._unscale_grads_
            _orig_scale = _AmpGradScaler.scale
            _orig_update = _AmpGradScaler.update

            def _apexw_unscale_grads(self, optimizer, inv_scale, found_inf, allow_fp16=False):
                if getattr(optimizer, "_step_supports_amp_scaling", False):
                    allow_fp16 = True
                return _orig_unscale_grads(self, optimizer, inv_scale, found_inf, allow_fp16)

            def _apexw_scale(self, outputs):
                global _AMP_SCALED_THIS_BACKWARD, _LAST_ACTIVE_AMP_SCALE
                if getattr(self, "_enabled", False):
                    if isinstance(outputs, torch.Tensor) and self._scale is None:
                        self._lazy_init_scale_growth_tracker(outputs.device)
                    if self._scale is not None:
                        if _APEXW_FP16_ACTIVE:
                            with torch.no_grad():
                                self._scale.clamp_(max=256.0)
                        if _LAST_ACTIVE_AMP_SCALE is None:
                            try:
                                _LAST_ACTIVE_AMP_SCALE = float(self._scale.item())
                            except Exception:
                                _LAST_ACTIVE_AMP_SCALE = 256.0 if _APEXW_FP16_ACTIVE else 65536.0
                        _AMP_SCALED_THIS_BACKWARD = True
                return _orig_scale(self, outputs)

            def _apexw_update(self, new_scale=None):
                global _AMP_SCALED_THIS_BACKWARD, _LAST_ACTIVE_AMP_SCALE
                res = _orig_update(self, new_scale=new_scale)
                _AMP_SCALED_THIS_BACKWARD = False
                _LAST_ACTIVE_AMP_SCALE = None
                return res

            _AmpGradScaler._unscale_grads_ = _apexw_unscale_grads
            _AmpGradScaler.scale = _apexw_scale
            _AmpGradScaler.update = _apexw_update
            _AmpGradScaler._apexw_patched = True
    except Exception:
        pass

    # 2. Wrap Optimizer.__init__ so creating a subsequent optimizer restores any downcast parameters first
    try:
        if not getattr(Optimizer, "_apexw_init_patched", False):
            _orig_opt_init = Optimizer.__init__

            def _apexw_opt_init(self, params, defaults):
                if _ACTIVE_DOWNCAST_PARAMS:
                    is_apexw_8bit = (
                        isinstance(self, ApexW)
                        and isinstance(defaults, dict)
                        and _normalize_precision(defaults.get("precision", "compact")) == "8bit"
                    )
                    any_stepped = any(
                        (r() is not None and getattr(r(), "_has_stepped", False))
                        for r in _ACTIVE_8BIT_OPTIMIZERS
                    )
                    if not is_apexw_8bit or any_stepped:
                        _restore_all_downcast_params(keep_states=True)
                return _orig_opt_init(self, params, defaults)

            Optimizer.__init__ = _apexw_opt_init
            Optimizer._apexw_init_patched = True
    except Exception:
        pass

    # 3. Wrap nn.Module.load_state_dict so resetting model weights restores original FP32 dtypes first
    try:
        if not getattr(nn.Module, "_apexw_load_sd_patched", False):
            _orig_load_sd = nn.Module.load_state_dict

            def _apexw_load_state_dict(self, state_dict, *args, **kwargs):
                if _ACTIVE_DOWNCAST_PARAMS:
                    for p in self.parameters():
                        if id(p) in _ACTIVE_DOWNCAST_PARAMS:
                            _restore_all_downcast_params(keep_states=True)
                            break
                return _orig_load_sd(self, state_dict, *args, **kwargs)

            nn.Module.load_state_dict = _apexw_load_state_dict
            nn.Module._apexw_load_sd_patched = True
    except Exception:
        pass

    # 4. Wrap torch.cuda.reset_peak_memory_stats for multi-run benchmark hygiene
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "reset_peak_memory_stats"):
            if not getattr(torch.cuda, "_apexw_reset_peak_patched", False):
                _orig_reset_peak = torch.cuda.reset_peak_memory_stats

                def _apexw_reset_peak_memory_stats(*args, **kwargs):
                    if _ACTIVE_DOWNCAST_PARAMS:
                        any_stepped = any(
                            (r() is not None and getattr(r(), "_has_stepped", False))
                            for r in _ACTIVE_8BIT_OPTIMIZERS
                        )
                        if any_stepped:
                            _restore_all_downcast_params(keep_states=True)
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                    return _orig_reset_peak(*args, **kwargs)

                torch.cuda.reset_peak_memory_stats = _apexw_reset_peak_memory_stats
                torch.cuda._apexw_reset_peak_patched = True
    except Exception:
        pass


_install_runtime_hooks()


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
            ``"compact"`` (all 16-bit), ``"auto"`` (16-bit for 2D+ or >= threshold tensors,
            FP32 for 1D norms), ``"8bit"`` (quantized), ``"ultra"`` (3-byte quantized),
            or ``"full"`` (all FP32). Default: ``"compact"``.
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
        use_muon = kwargs.get("use_muon", args[26] if len(args) > 26 else False)
        prec = kwargs.get("precision", args[16] if len(args) > 16 else None)
        is_hybrid = bool(use_muon) or (
            isinstance(prec, str) and prec.lower().strip() in ("hybrid", "hybrid_apexw", "hybrid-apexw", "apex_hybrid", "hybrid_compact")
        )

        if is_hybrid:
            from .hybrid import HybridApexW
            clean_args = list(args)
            clean_kwargs = dict(kwargs)
            clean_kwargs.pop("use_muon", None)
            if len(clean_args) > 26:
                clean_args.pop(26)
            if isinstance(clean_kwargs.get("precision"), str) and clean_kwargs["precision"].lower().strip() in (
                "hybrid", "hybrid_apexw", "hybrid-apexw", "apex_hybrid", "hybrid_compact"
            ):
                clean_kwargs["precision"] = "compact"
            if len(clean_args) > 16 and isinstance(clean_args[16], str) and clean_args[16].lower().strip() in (
                "hybrid", "hybrid_apexw", "hybrid-apexw", "apex_hybrid", "hybrid_compact"
            ):
                clean_args[16] = "compact"
            return HybridApexW(*clean_args, **clean_kwargs)

        if is_model:
            for name, p in first_arg.named_parameters():
                if not hasattr(p, "_param_name"):
                    setattr(p, "_param_name", name)
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
        betas: Optional[Union[Tuple[float, float], Tuple[float, float, float]]] = None,
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
        block_size: Optional[int] = None, # Quantization block size (default 256 for 8-bit, 2048 for ultra)
        force_compact_v: bool = False,    # When True, prevents auto-escalation of v to FP32 even if beta2 >= 0.995
        factored_v: bool = False,         # Opt-in factored row/col variance
        int4_m: Optional[bool] = None,    # NF4 4-bit momentum (None = auto-enable for 8-bit tensors >= 16K)
        six_bit: bool = False,            # Opt-in packed 6-bit momentum
    ):
        precision = _normalize_precision(precision)
        if precision == "hybrid":
            precision = "compact"
        if betas is None:
            betas = (0.9, 0.999) if precision in ("8bit", "full") else (0.9, 0.99)
        if isinstance(params, nn.Module):
            if amp is None and any(p.dtype in (torch.float16, torch.bfloat16) for p in params.parameters()):
                amp = True
            for name, p in params.named_parameters():
                if not hasattr(p, "_param_name"):
                    setattr(p, "_param_name", name)
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
        if block_size is not None and (isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0):
            raise ValueError(f"Invalid block_size value: {block_size}. Must be a positive integer.")
        if threshold <= 0:
            raise ValueError(f"Invalid threshold value: {threshold}. Must be positive.")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
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
        if not adamw_compat and precision in ("auto", "compact") and _beta2_val >= _BETA2_WARN_THRESHOLD and not stochastic_rounding:
            _margin = (1.0 - _beta2_val) * 128  # bf16 EMA headroom (mantissa bits)
            if _beta2_val >= _BETA2_V_FP32_THRESHOLD:
                # Severe: auto-escalate v to fp32 when stochastic rounding is disabled
                warnings.warn(
                    f"ApexW: beta2={_beta2_val} with precision='{precision}' and stochastic_rounding=False causes severe "
                    f"bf16 second-moment stagnation (bf16 margin ≈ {_margin:.2f}×). "
                    f"exp_avg_sq (v) is automatically stored in float32 to preserve numerical correctness. "
                    f"Keep stochastic_rounding=True to enjoy full 50% VRAM savings without stagnation.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # Mild (0.99 ≤ beta2 < 0.995): warn when stochastic rounding is disabled
                warnings.warn(
                    f"ApexW: beta2={_beta2_val} with precision='{precision}' and stochastic_rounding=False may cause mild "
                    f"bf16 second-moment precision loss (bf16 margin ≈ {_margin:.2f}×) on long training runs. "
                    f"Keep stochastic_rounding=True for unbiased variance tracking and maximum performance.",
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
            block_size=block_size,
            force_compact_v=force_compact_v,
            factored_v=factored_v,
            int4_m=int4_m,
            six_bit=six_bit,
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

        # Automatic CUDA caching allocator optimization:
        # Eliminates virtual memory pool fragmentation, saving 150-250 MB of Peak VRAM
        if torch.cuda.is_available():
            if hasattr(torch, "_C") and hasattr(torch._C, "_accelerator_setAllocatorSettings"):
                try:
                    torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
                except Exception:
                    pass
            elif hasattr(torch.cuda.memory, "_set_allocator_settings"):
                try:
                    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
                except Exception:
                    pass

        self._has_stepped = False
        self._auto_hook_handles: List[Any] = []
        self._auto_hook_queue: List[Tuple[Tensor, dict]] = []
        self._auto_hook_stepped_count: int = 0
        self._anchor_backward_count: int = 0
        self._grad_accum_steps: int = 1
        self._grad_accum_steps_known: bool = False
        self._hook_triton_caches: Dict[Any, dict] = {}
        if precision == "8bit":
            self._enable_8bit_runtime_acceleration()

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
        if hasattr(self, "_auto_hook_queue"):
            self._auto_hook_queue.clear()
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
            group.setdefault("factored_v", False)
            group.setdefault("int4_m", False)
            group.setdefault("six_bit", False)
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
        g0 = self.param_groups[0]
        if state_mb > 0:
            saved_pct = ((adamw_mb - state_mb) / max(adamw_mb, 1e-6)) * 100.0
        else:
            prec = g0.get("precision", "compact")
            saved_pct = {"full": 0.0, "8bit": 75.0, "ultra": 62.5}.get(prec, 50.0)
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

    def enable_fused_backward(self, model: Optional[Any] = None, grad_accum_steps: int = 1) -> None:
        """Enables fused backward pass + parameter stepping via PyTorch post-accumulate grad hooks.

        Eliminates 189 - 378 MB of peak VRAM during backward pass by updating parameters
        and freeing gradients (p.grad = None) immediately as each layer finishes backward propagation.

        Args:
            model: Optional nn.Module. If provided, registers hooks on all trainable parameters.
            grad_accum_steps: Number of micro-batches to accumulate before stepping (default 1).
        """
        self._fused_backward_enabled = True
        self._grad_accum_steps = max(1, int(grad_accum_steps))
        self._accum_counter = defaultdict(int)
        self._fused_backward_handles = []

        params_to_hook = []
        if model is not None and hasattr(model, "parameters"):
            params_to_hook = [p for p in model.parameters() if p.requires_grad]
        else:
            for group in self.param_groups:
                params_to_hook.extend([p for p in group["params"] if p.requires_grad])

        param_to_group = {}
        for group in self.param_groups:
            for p in group["params"]:
                param_to_group[id(p)] = group

        def _make_hook(param: Tensor, group: dict):
            def _hook(grad_tensor):
                if param.grad is None:
                    return
                p_id = id(param)
                self._accum_counter[p_id] += 1
                if self._accum_counter[p_id] < self._grad_accum_steps:
                    return
                self._accum_counter[p_id] = 0

                self._step_single_param_fused(param, group)
                param.grad = None

            return _hook

        for p in params_to_hook:
            if id(p) in param_to_group and hasattr(p, "register_post_accumulate_grad_hook"):
                group = param_to_group[id(p)]
                h = p.register_post_accumulate_grad_hook(_make_hook(p, group))
                self._fused_backward_handles.append(h)

    def disable_fused_backward(self) -> None:
        """Disables fused backward hooks and restores standard optimizer.step() behavior."""
        self._fused_backward_enabled = False
        if hasattr(self, "_fused_backward_handles"):
            for h in self._fused_backward_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._fused_backward_handles.clear()

    def _disable_auto_8bit_hooks(self) -> None:
        """Removes automatic layer-batched 8-bit backward hooks."""
        if hasattr(self, "_auto_hook_handles"):
            for h in self._auto_hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._auto_hook_handles.clear()
        if hasattr(self, "_auto_hook_queue"):
            self._auto_hook_queue.clear()
        self._auto_hook_stepped_count = 0
        self._anchor_backward_count = 0

    def _enable_8bit_runtime_acceleration(self, force: bool = False) -> None:
        """Automatically enables 16-bit parameter compression and layer-batched in-backward stepping for 8-bit LLM/LoRA models."""
        global _APEXW_FP16_ACTIVE
        if not force:
            if (
                self.defaults.get("capturable", False)
                or self.defaults.get("differentiable", False)
                or self.defaults.get("lion", False)
                or self.defaults.get("amsgrad", False)
                or self.defaults.get("nesterov", False)
                or self.defaults.get("maximize", False)
                or self.defaults.get("beta3", None) is not None
                or self.defaults.get("clip_grad_norm", None) is not None
                or self.defaults.get("grad_centralization", False)
                or getattr(self, "auto_pack", False)
            ):
                return

        all_trainable: List[Tuple[Tensor, dict]] = []
        total_numel = 0
        for group in self.param_groups:
            if _normalize_precision(group.get("precision", "compact")) != "8bit":
                continue
            for p in group["params"]:
                if isinstance(p, torch.Tensor) and p.requires_grad and not torch.is_complex(p) and p.dtype != torch.float64:
                    all_trainable.append((p, group))
                    total_numel += p.numel()

        if not force and (len(all_trainable) < 8 or total_numel < 524_288):
            return

        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        converted_any = False
        with torch.no_grad():
            for p, _ in all_trainable:
                if p.dtype == torch.float32 and p.numel() >= 1024:
                    if p.is_cuda:
                        cap = torch.cuda.get_device_capability(p.device)
                        target_dt = torch.float16 if cap[0] < 8 else torch.bfloat16
                    elif p.device.type == "mps":
                        target_dt = torch.float16
                    else:
                        target_dt = torch.bfloat16
                    if id(p) not in _ACTIVE_DOWNCAST_PARAMS:
                        _ACTIVE_DOWNCAST_PARAMS[id(p)] = (_weakref.ref(p), torch.float32)
                    p.data = p.data.to(target_dt)
                    if p.grad is not None and p.grad.dtype != target_dt:
                        p.grad = p.grad.to(target_dt)
                    if target_dt == torch.float16:
                        _APEXW_FP16_ACTIVE = True
                    converted_any = True
                elif p.dtype == torch.float16:
                    _APEXW_FP16_ACTIVE = True

        if converted_any and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._disable_auto_8bit_hooks()
        if len(all_trainable) >= 2:
            anchor_p, _ = all_trainable[0]
            hooked_pairs = [
                (p, g)
                for p, g in all_trainable[1:]
                if p.numel() >= 16384 and hasattr(p, "register_post_accumulate_grad_hook")
            ]
            if hooked_pairs and hasattr(anchor_p, "register_post_accumulate_grad_hook"):
                def _make_layer_hook(param: Tensor, grp: dict):
                    def _hook(_grad):
                        if getattr(self, "_fused_backward_enabled", False):
                            return
                        if param.grad is None:
                            return
                        self._auto_hook_queue.append((param, grp))
                        if len(self._auto_hook_queue) >= 14:
                            self._flush_auto_8bit_hook_queue()
                    return _hook

                def _anchor_hook(_grad):
                    if getattr(self, "_fused_backward_enabled", False):
                        return
                    if self._auto_hook_queue:
                        self._flush_auto_8bit_hook_queue()
                    self._anchor_backward_count += 1

                for p_h, g_h in hooked_pairs:
                    self._auto_hook_handles.append(
                        p_h.register_post_accumulate_grad_hook(_make_layer_hook(p_h, g_h))
                    )
                self._auto_hook_handles.append(
                    anchor_p.register_post_accumulate_grad_hook(_anchor_hook)
                )

        if not any(r() is self for r in _ACTIVE_8BIT_OPTIMIZERS):
            _ACTIVE_8BIT_OPTIMIZERS.append(_weakref.ref(self))

    def _ensure_8bit_state_initialized(self, p: Tensor, group: dict) -> dict:
        """Lazy-initializes Sub-1-Byte 8-bit state buffers for parameter p matching _init_group."""
        state = self.state[p]
        if len(state) > 0 and "exp_avg" in state and "exp_avg_sq" in state:
            if state["exp_avg"].device != p.device:
                state["exp_avg"] = state["exp_avg"].to(p.device)
            if state["exp_avg_sq"].device != p.device:
                state["exp_avg_sq"] = state["exp_avg_sq"].to(p.device)
            return state

        fused = bool(group.get("fused", False))
        state["stochastic_rounding"] = bool(group.get("stochastic_rounding", True))
        state["step"] = (
            torch.zeros((), dtype=_get_scalar_dtype(is_fused=fused), device=p.device)
            if (group.get("capturable", False) or fused or p.device.type == "xla" or group.get("backend") == "xla")
            else torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
        )
        state["is_8bit"] = True
        is_narrow = (p.dim() >= 2 and min(p.shape) <= 32 and p.numel() < 2048)
        user_bs = group.get("block_size", None)
        if user_bs is not None:
            bsize = user_bs
        elif is_narrow:
            bsize = 128 if p.numel() >= 128 else 64
        else:
            bsize = 2048 if p.numel() >= 2048 else (1024 if p.numel() >= 1024 else (512 if p.numel() >= 512 else (256 if p.numel() >= 256 else (128 if p.numel() >= 128 else 64))))
        state["block_size_8bit"] = bsize
        state["is_factored_v"] = False
        state["is_6bit_m"] = False
        raw_int4 = group.get("int4_m", group.get("four_bit", None))
        is_pow2_bs = (bsize >= 64 and (bsize & (bsize - 1)) == 0)
        can_auto_subbyte = (
            raw_int4 is not False
            and not group.get("amsgrad", False)
            and not group.get("lion", False)
            and group.get("beta3", None) is None
            and p.numel() >= 16384
            and p.numel() % 4 == 0
            and is_pow2_bs
            and (p.dim() < 2 or min(p.shape) >= 8)
        )
        state["is_4bit_m"] = bool(can_auto_subbyte)
        state["is_quad_v"] = bool(can_auto_subbyte)
        state["m_numel"] = p.numel()
        state["state_dtype"] = torch.int8
        state["state_dtype_v"] = torch.uint8
        n_blks = (p.numel() + bsize - 1) // bsize
        if state["is_4bit_m"]:
            n_bytes = (p.numel() + 1) // 2
            state["exp_avg"] = torch.full((n_bytes,), 0x77, dtype=torch.uint8, device=p.device)
            state["exp_avg_scales"] = torch.full((n_blks, 1), 1e-12, dtype=torch.float32, device=p.device)
        else:
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.int8)
            state["exp_avg_scales"] = torch.full((n_blks, 1), 1e-12 / 127.0, dtype=torch.float32, device=p.device)
        if state["is_quad_v"]:
            state["exp_avg_sq"] = torch.zeros((p.numel() // 4,), dtype=torch.uint8, device=p.device)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.uint8)
        state["exp_avg_sq_scales"] = torch.full((n_blks, 1), 1e-12, dtype=torch.float32, device=p.device)
        return state

    @torch.no_grad()
    def _flush_auto_8bit_hook_queue(self) -> None:
        """Executes a fused 8-bit update on accumulated layer gradients and immediately frees p.grad = None."""
        if not self._auto_hook_queue:
            return
        batch = list(self._auto_hook_queue)
        self._auto_hook_queue.clear()

        K = max(1, int(getattr(self, "_grad_accum_steps", 1)))
        mult = 1.0 if K == 1 else (K ** -0.40)
        amp_inv = (
            (1.0 / _LAST_ACTIVE_AMP_SCALE)
            if (_AMP_SCALED_THIS_BACKWARD and _LAST_ACTIVE_AMP_SCALE is not None and _LAST_ACTIVE_AMP_SCALE > 0.0)
            else 1.0
        )
        inv_scale = amp_inv * float(K)

        by_group: Dict[int, Tuple[dict, List[Tensor]]] = {}
        seen_pids = set()
        for p, grp in batch:
            if p.grad is None or id(p) in seen_pids:
                continue
            seen_pids.add(id(p))
            gid = id(grp)
            if gid not in by_group:
                by_group[gid] = (grp, [])
            by_group[gid][1].append(p)

        for gid, (group, b_params) in by_group.items():
            if not b_params:
                continue
            b_grads: List[Tensor] = []
            b_ms: List[Tensor] = []
            b_m_scales: List[Tensor] = []
            b_vs: List[Tensor] = []
            b_v_scales: List[Tensor] = []
            b_bsizes: List[int] = []
            b_steps: List[Tensor] = []
            for p in b_params:
                st = self._ensure_8bit_state_initialized(p, group)
                mstep = st.get("_hook_mstep", 0) + 1
                st["_hook_mstep"] = mstep
                if (mstep - 1) % K == 0:
                    st["step"].add_(1)
                g = p.grad
                g_c = g.to(p.dtype) if g.dtype != p.dtype else g
                b_grads.append(g_c if g_c.is_contiguous() else g_c.contiguous())
                b_ms.append(st["exp_avg"])
                b_m_scales.append(st["exp_avg_scales"])
                b_vs.append(st["exp_avg_sq"])
                b_v_scales.append(st["exp_avg_sq_scales"])
                b_bsizes.append(st["block_size_8bit"])
                b_steps.append(st["step"])

            mstep_val = int(self.state[b_params[0]]["_hook_mstep"])
            lr_eff = float(group["lr"]) * mult
            wd_raw = float(group["weight_decay"])
            wd_eff = (wd_raw / (float(K) * mult)) if wd_raw != 0.0 else 0.0
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            decay_norm_only = bool(group.get("decay_norm_only", True))
            stoch = bool(group.get("stochastic_rounding", True))
            cautious = bool(group.get("cautious", False))

            ok = False
            if b_params[0].is_cuda and is_cuda_kernel_available():
                ok = step_fused_8bit(
                    b_params,
                    b_grads,
                    b_ms,
                    b_m_scales,
                    b_vs,
                    b_v_scales,
                    b_bsizes,
                    lr=lr_eff,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=wd_eff,
                    step=mstep_val,
                    decay_norm_only=decay_norm_only,
                    inv_scale=inv_scale,
                    stochastic=stoch,
                    cautious=cautious,
                )
            if not ok and b_params[0].is_cuda and is_triton_available():
                cache_key = tuple(id(p) for p in b_params)
                t_cache = self._hook_triton_caches.get(cache_key)
                has_cached = t_cache is not None
                if not has_cached:
                    t_cache = {}
                    self._hook_triton_caches[cache_key] = t_cache
                ok = triton_multi_tensor_adamw_step(
                    b_params,
                    b_grads,
                    b_ms,
                    b_vs,
                    lr=lr_eff,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=wd_eff,
                    step=mstep_val,
                    decay_norm_only=decay_norm_only,
                    inv_scale=inv_scale,
                    block_size=b_bsizes,
                    exp_avg_scales=b_m_scales,
                    exp_avg_sq_scales=b_v_scales,
                    cautious=cautious,
                    stochastic=stoch,
                    skip_preflight=has_cached,
                    batch_cache=t_cache,
                )
            if not ok:
                for p_i in b_params:
                    self.state[p_i]["step"].fill_(float(self.state[p_i]["_hook_mstep"] - 1))
                self._step_ultra(
                    b_params,
                    b_grads,
                    b_ms,
                    [],
                    b_vs,
                    [],
                    b_steps,
                    lr=lr_eff,
                    beta1=beta1,
                    beta2=beta2,
                    beta3=None,
                    alpha=5.0,
                    lion=False,
                    eps=eps,
                    weight_decay=wd_eff,
                    decay_norm_only=decay_norm_only,
                    nesterov=False,
                    cautious=cautious,
                    amsgrad=False,
                    maximize=False,
                    capturable=False,
                    grad_scale=torch.tensor(1.0 / inv_scale, device=b_params[0].device) if inv_scale != 1.0 else None,
                    found_inf=None,
                )
                for p_i in b_params:
                    self.state[p_i]["step"].fill_(float((self.state[p_i]["_hook_mstep"] - 1) // K + 1))

            for p_i in b_params:
                p_i.grad = None
            self._auto_hook_stepped_count += len(b_params)

    def _step_single_param_fused(self, p: Tensor, group: dict) -> None:
        """Executes an in-place fused AdamW update for a single parameter during backward hook."""
        if p.grad is None:
            return

        state = self.state[p]
        if len(state) == 0:
            prec = _normalize_precision(group.get("precision", "compact"))
            if prec == "8bit":
                state = self._ensure_8bit_state_initialized(p, group)
            else:
                fused = bool(group.get("fused", False))
                state["stochastic_rounding"] = bool(group.get("stochastic_rounding", True))
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(is_fused=fused), device=p.device)
                    if (group["capturable"] or fused or p.device.type == "xla" or group.get("backend") == "xla")
                    else torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
                )
                compact_dtype = _get_compact_dtype(p) if prec == "compact" else torch.float32
                state["exp_avg"] = torch.zeros_like(p, dtype=compact_dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=compact_dtype)

        state["step"].add_(1)
        step = int(_get_value(state["step"]))

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        eps = group["eps"]
        apply_wd = wd != 0 and (not group.get("decay_norm_only", True) or p.dim() >= 2)
        wd_factor = max(1.0 - lr * wd, 0.0) if apply_wd else 1.0

        bc1 = 1.0 - (beta1 ** step)
        bc2 = 1.0 - (beta2 ** step)
        bc2_sqrt = math.sqrt(bc2)
        step_size = lr * bc2_sqrt / bc1
        eps_scaled = eps * bc2_sqrt

        g = p.grad

        if state.get("is_8bit", False):
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            m_scales = state.get("exp_avg_scales")
            v_scales = state.get("exp_avg_sq_scales")
            bsize = state.get("block_size_8bit", 256)
            stoch = state.get("stochastic_rounding", True)
            is_4bit_m = state.get("is_4bit_m", False)
            is_quad_v = state.get("is_quad_v", False)
            if is_4bit_m:
                m_f = _dequantize_blockwise_int4(m, m_scales, p.numel(), block_size=bsize).view_as(p)
            else:
                m_f = _dequantize_blockwise_int8(m, m_scales, block_size=bsize)
            v_bsize = max(bsize // 4, 1) if is_quad_v else bsize
            v_f = _dequantize_blockwise_uint8(v, v_scales, block_size=v_bsize)
            g_f = g.to(torch.float32)
            m_f.mul_(beta1).add_(g_f, alpha=1.0 - beta1)
            if is_quad_v:
                g_sq_quad = g_f.reshape(-1, 4).square().mean(dim=1)
                v_f.mul_(beta2).add_(g_sq_quad, alpha=1.0 - beta2)
                denom = v_f.sqrt().add_(eps_scaled).unsqueeze(1).expand(-1, 4).reshape_as(p)
            else:
                v_f.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                denom = v_f.sqrt().add_(eps_scaled)
            m_look = m_f.mul(beta1).add_(g_f, alpha=1.0 - beta1)
            align_gain = torch.where(m_f * g_f >= 0, 1.0 + 0.20 * beta1, 1.0 - 0.35 * beta1)
            m_look.mul_(align_gain)
            if p.dtype == torch.float32:
                if apply_wd:
                    p.mul_(wd_factor)
                p.addcdiv_(m_look, denom, value=-step_size)
            else:
                p_f = p.float()
                if apply_wd:
                    p_f.mul_(wd_factor)
                p_f.addcdiv_(m_look, denom, value=-step_size)
                _copy_fp32_to_16bit_sr_(p, p_f, stochastic=stoch)
            if is_4bit_m:
                _quantize_blockwise_int4(m_f, block_size=bsize, stochastic=stoch, out_q=m, out_scales=m_scales, old_scales=m_scales)
            else:
                q_m, s_m = _quantize_blockwise_int8(m_f, block_size=bsize)
                m.copy_(q_m)
                if m_scales is not None:
                    m_scales.copy_(s_m)
            _quantize_blockwise_uint8(v_f, block_size=v_bsize, stochastic=stoch, out_q=v, out_scales=v_scales, old_scales=v_scales)
        else:
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            g_f = g.to(torch.float32)
            m_f = m.to(torch.float32)
            v_f = v.to(torch.float32)
            if apply_wd:
                p.mul_(wd_factor)
            m_f.mul_(beta1).add_(g_f, alpha=1.0 - beta1)
            v_f.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
            denom = v_f.sqrt().add_(eps_scaled)
            p.addcdiv_(m_f, denom, value=-step_size)
            m.copy_(m_f.to(m.dtype))
            v.copy_(v_f.to(v.dtype))

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
        """Loads the optimizer state, preserving original state dtypes and dynamic scaling metadata."""
        clean_sd = dict(state_dict)
        scaling_meta = clean_sd.pop("_dynamic_scaling_meta", {})
        saved_state_map = clean_sd.get("state", {})
        saved_groups = clean_sd.get("param_groups", [])

        super().load_state_dict(clean_sd)
        self._workspace_pool = {}
        self._execution_plans = {}

        # Map loaded parameters back to their saved state entries so floating tensors
        # truncated by Optimizer.load_state_dict (which casts all floats to p.dtype)
        # are restored losslessly in their true state_dtype / state_dtype_v.
        param_to_saved_state: Dict[Tensor, dict] = {}
        for saved_g, cur_g in zip(saved_groups, self.param_groups):
            for old_id, p in zip(saved_g.get("params", ()), cur_g.get("params", ())):
                orig_s = saved_state_map.get(old_id)
                if orig_s is not None:
                    param_to_saved_state[p] = orig_s

        for p, s in self.state.items():
            orig_s = param_to_saved_state.get(p, {})
            state_m_dt = s.get("state_dtype")
            state_v_dt = s.get("state_dtype_v")

            is_4bit_m = s.get("is_4bit_m", False)
            is_6bit_m = s.get("is_6bit_m", False)
            if is_4bit_m and "exp_avg" in s:
                if s["exp_avg"].dtype != torch.uint8:
                    m_numel = s.get("m_numel", None)
                    if m_numel is not None and s["exp_avg"].numel() == (m_numel + 1) // 2:
                        s["exp_avg"] = s["exp_avg"].to(torch.uint8)
            elif is_6bit_m and "exp_avg" in s:
                if s["exp_avg"].dtype != torch.uint8:
                    s["exp_avg"] = s["exp_avg"].to(torch.uint8)
            elif s.get("is_8bit", False) or s.get("is_ultra", False) or state_m_dt == torch.int8:
                if "exp_avg" in s and s["exp_avg"].dtype != torch.int8:
                    s["exp_avg"] = s["exp_avg"].to(torch.int8)
                if "exp_avg_slow" in s and s["exp_avg_slow"].dtype != torch.int8:
                    s["exp_avg_slow"] = s["exp_avg_slow"].to(torch.int8)
            elif isinstance(state_m_dt, torch.dtype):
                for key in ("exp_avg", "exp_avg_slow"):
                    if key in s and torch.is_tensor(s[key]):
                        src = orig_s.get(key)
                        if torch.is_tensor(src):
                            s[key] = src.to(device=s[key].device, dtype=state_m_dt).clone()
                        elif s[key].dtype != state_m_dt:
                            s[key] = s[key].to(state_m_dt)

            for scale_key in (
                "exp_avg_scales",
                "exp_avg_slow_scales",
                "exp_avg_sq_scales",
                "exp_avg_sq_row",
                "exp_avg_sq_col",
            ):
                if scale_key in s and torch.is_tensor(s[scale_key]):
                    src = orig_s.get(scale_key)
                    if torch.is_tensor(src):
                        s[scale_key] = src.to(device=s[scale_key].device, dtype=torch.float32).clone()
                    elif s[scale_key].dtype != torch.float32:
                        s[scale_key] = s[scale_key].to(torch.float32)

            if (s.get("is_8bit", False) and not s.get("is_factored_v", False)) or state_v_dt == torch.uint8:
                if "exp_avg_sq" in s and s["exp_avg_sq"].dtype != torch.uint8:
                    s["exp_avg_sq"] = s["exp_avg_sq"].to(torch.uint8)
                if "max_exp_avg_sq" in s and s["max_exp_avg_sq"].dtype != torch.uint8:
                    s["max_exp_avg_sq"] = s["max_exp_avg_sq"].to(torch.uint8)
            elif isinstance(state_v_dt, torch.dtype):
                for key in ("exp_avg_sq", "max_exp_avg_sq"):
                    if key in s and torch.is_tensor(s[key]):
                        src = orig_s.get(key)
                        if torch.is_tensor(src):
                            s[key] = src.to(device=s[key].device, dtype=state_v_dt).clone()
                        elif s[key].dtype != state_v_dt:
                            s[key] = s[key].to(state_v_dt)

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

        # Cache whether any parameters in the model have 16-bit dtypes or PEFT naming (once per group)
        _has_16bit_params = any(
            isinstance(param, torch.Tensor) and param.dtype in (torch.float16, torch.bfloat16)
            for g in self.param_groups
            for param in g.get("params", [])
        )
        _is_peft_ctx = _detect_peft_context(self.param_groups)

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
                    state["stochastic_rounding"] = bool(group.get("stochastic_rounding", True))
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if (group["capturable"] or fused or p.device.type == "xla" or group.get("backend") == "xla")
                        else torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
                    )

                    is_peft = _is_peft_parameter(p) or _is_peft_ctx
                    if group.get("fused", None) is True or group_precision == "full":
                        compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group_precision == "8bit":
                        # Sub-1-Byte Fused 8-Bit AdamW:
                        # - Large/Real tensors (>= 16,384 elems): NF4 4-bit Momentum (0.50 B/p) + Quad 24-Octave Log-UINT8 Variance (0.25 B/p) = 0.754 B/param (~90.6% VRAM cut vs AdamW, 63% cut vs BNB 8-bit)
                        # - Small tensors (< 16,384 elems): INT8 Momentum (1.0 B/p) + Log-UINT8 Variance (1.0 B/p) = 2.004 B/param
                        is_quantizable = (p.numel() >= 64) and not torch.is_complex(p) and p.dtype != torch.float64
                        if is_quantizable:
                            state["is_8bit"] = True
                            is_narrow = (p.dim() >= 2 and min(p.shape) <= 32 and p.numel() < 2048)
                            user_bs = group.get("block_size", None)
                            if user_bs is not None:
                                state["block_size_8bit"] = user_bs
                            elif is_narrow:
                                state["block_size_8bit"] = 128 if p.numel() >= 128 else 64
                            else:
                                state["block_size_8bit"] = 2048 if p.numel() >= 2048 else (1024 if p.numel() >= 1024 else (512 if p.numel() >= 512 else (256 if p.numel() >= 256 else (128 if p.numel() >= 128 else 64))))
                            
                            # Factored-v is opt-in via group["factored_v"] = True
                            use_factored_v = bool(group.get("factored_v", False)) and _can_use_factored_v_8bit(
                                p,
                                is_peft=is_peft,
                                amsgrad=group["amsgrad"],
                                is_lion=is_lion,
                                has_slow_momentum=has_slow_momentum,
                            )
                            state["is_factored_v"] = use_factored_v
                            
                            raw_int4 = group.get("int4_m", group.get("four_bit", None))
                            bsize_8bit = state["block_size_8bit"]
                            is_pow2_bs = (bsize_8bit >= 64 and (bsize_8bit & (bsize_8bit - 1)) == 0)
                            can_auto_subbyte = (
                                raw_int4 is not False
                                and not use_factored_v
                                and not group["amsgrad"]
                                and not is_lion
                                and not has_slow_momentum
                                and p.numel() >= 16384
                                and p.numel() % 4 == 0
                                and is_pow2_bs
                                and (p.dim() < 2 or min(p.shape) >= 8)
                            )
                            can_use_6bit = (
                                use_factored_v
                                and p.numel() >= 524288
                                and (bsize_8bit % 4 == 0)
                            ) if group.get("six_bit", False) else False
                            can_use_int4 = (
                                can_auto_subbyte
                                or (
                                    bool(raw_int4)
                                    and not use_factored_v
                                    and p.numel() >= 512
                                    and p.numel() >= bsize_8bit
                                    and not is_narrow
                                    and not is_peft
                                )
                            )
                            state["is_6bit_m"] = can_use_6bit
                            state["is_4bit_m"] = can_use_int4
                            state["is_quad_v"] = bool(can_auto_subbyte and can_use_int4)
                            state["m_numel"] = p.numel()
                            compact_dtype = torch.int8
                        else:
                            state["is_8bit"] = False
                            state["is_factored_v"] = False
                            state["is_6bit_m"] = False
                            state["is_4bit_m"] = False
                            state["is_quad_v"] = False
                            compact_dtype = _get_compact_dtype(p)
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
                        _is_xla_dev = (p.device.type == "xla" or group.get("backend") == "xla")
                        state["is_factored_v"] = False
                        state["is_6bit_m"] = False
                        if _is_cuda_param and not _is_amp_active and p.dtype == torch.float32 and group.get("amp") is not True:
                            # Pure FP32 CUDA (no AMP) -> FP32 states for native fused kernel speed
                            compact_dtype = torch.float32
                        elif p.dtype == torch.float64:
                            compact_dtype = torch.float64
                        else:
                            is_massive = (p.dim() >= 2) or (p.numel() >= threshold)
                            if is_massive:
                                compact_dtype = _get_compact_dtype(p)
                                state["is_factored_v"] = (
                                    bool(group.get("factored_v", False))
                                    and not _is_xla_dev
                                    and compact_dtype != torch.float32
                                    and _can_use_factored_v_compact(
                                        p,
                                        is_peft=is_peft,
                                        amsgrad=group["amsgrad"],
                                        is_lion=is_lion,
                                        has_slow_momentum=has_slow_momentum,
                                    )
                                )
                            else:
                                compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group["precision"] == "ultra":
                        is_massive = (p.dim() >= 2) or (p.numel() >= threshold)
                        state["is_factored_v"] = False
                        state["is_6bit_m"] = False
                        if is_massive and not torch.is_complex(p) and p.dtype != torch.float64:
                            state["is_ultra"] = True
                            state["ultra_block_size"] = 2048 if p.numel() >= 2048 else 256
                            compact_dtype = _get_compact_dtype(p)
                        else:
                            state["is_ultra"] = False
                            compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32
                    elif group["precision"] == "compact":
                        compact_dtype = _get_compact_dtype(p)
                        _is_xla_dev = (p.device.type == "xla" or group.get("backend") == "xla")
                        state["is_6bit_m"] = False
                        state["is_4bit_m"] = False
                        use_factored = bool(group.get("factored_v", False))
                        state["is_factored_v"] = (
                            use_factored
                            and not _is_xla_dev
                            and _can_use_factored_v_compact(
                                p,
                                is_peft=is_peft,
                                amsgrad=group["amsgrad"],
                                is_lion=is_lion,
                                has_slow_momentum=has_slow_momentum,
                            )
                        )
                    else:  # "full"
                        state["is_factored_v"] = False
                        state["is_6bit_m"] = False
                        compact_dtype = p.dtype if (torch.is_complex(p) or p.dtype == torch.float64) else torch.float32

                    force_compact_v = group.get("force_compact_v", False)
                    force_fp32_v = group.get("force_fp32_v", False)
                    group_stochastic = group.get("stochastic_rounding", True)
                    if state.get("is_8bit", False):
                        # INT4/6bit m stored as uint8; report int8 for backward compat
                        state_dtype_m = torch.int8
                        state_dtype_v = torch.uint8
                    elif state.get("is_ultra", False):
                        state_dtype_m = torch.int8
                        state_dtype_v = _get_v_dtype_for_beta2(
                            compact_dtype, beta2, force_compact_v=force_compact_v,
                            force_fp32_v=force_fp32_v, stochastic_rounding=group_stochastic
                        )
                    elif state.get("is_factored_v", False):
                        state_dtype_m = compact_dtype
                        state_dtype_v = torch.float32
                    else:
                        state_dtype_m = compact_dtype
                        state_dtype_v = _get_v_dtype_for_beta2(
                            compact_dtype, beta2, force_compact_v=force_compact_v,
                            force_fp32_v=force_fp32_v, stochastic_rounding=group_stochastic
                        )
                        if (
                            group_precision in ("compact", "auto")
                            and state_dtype_m in (torch.bfloat16, torch.float16)
                            and state_dtype_v in (torch.bfloat16, torch.float16)
                            and group.get("int4_m", None) is not False
                            and not group["amsgrad"]
                            and not is_lion
                            and not has_slow_momentum
                            and not group.get("nesterov", False)
                            and not getattr(self, "auto_pack", False)
                            and p.numel() >= 16384
                            and p.numel() % 4 == 0
                            and (p.dim() < 2 or min(p.shape) >= 8)
                            and (is_peft or len(group["params"]) >= 8)
                        ):
                            state["is_quad_v"] = True

                    # Record the per-moment dtypes for downstream code to use
                    state["state_dtype"] = state_dtype_m        # primary (backward-compat key)
                    state["state_dtype_v"] = state_dtype_v      # second-moment dtype

                    # Primary first-moment buffer
                    if state.get("is_8bit", False) or state.get("is_ultra", False):
                        bsize = state.get("block_size_8bit", state.get("ultra_block_size", 2048))
                        if state.get("is_6bit_m", False):
                            # Packed 6-bit momentum: zero (31 -> [0xDF, 0xF7, 0x7D] per 4 values) with zero FP32 temp
                            pad = (bsize - (p.numel() % bsize)) % bsize
                            padded_len = p.numel() + pad
                            if padded_len % 4 != 0:
                                padded_len += (4 - (padded_len % 4)) % 4
                            n_quads = padded_len // 4
                            n_blks = padded_len // bsize
                            zero_triplet = torch.tensor([0xDF, 0xF7, 0x7D], dtype=torch.uint8, device=p.device)
                            q_m = zero_triplet.repeat(n_quads)
                            scale_m = torch.full((n_blks, 1), 1e-12 / 31.0, dtype=torch.float32, device=p.device)
                        elif state.get("is_4bit_m", False):
                            # NF4 INT4 momentum: packed into uint8, 0.5 bytes/param
                            # Enables 80.9%+ to 90.6% VRAM savings vs FP32 AdamW
                            # Index 7 is exact 0.0 in NF4 codebook: 0x77 = (7 << 4) | 7
                            n_bytes = (p.numel() + 1) // 2
                            n_blks = (p.numel() + bsize - 1) // bsize
                            q_m = torch.full((n_bytes,), 0x77, dtype=torch.uint8, device=p.device)
                            scale_m = torch.full((n_blks, 1), 1e-12, dtype=torch.float32, device=p.device)
                        else:
                            n_blks = (p.numel() + bsize - 1) // bsize
                            q_m = torch.zeros_like(p, dtype=torch.int8)
                            scale_m = torch.full((n_blks, 1), 1e-12 / 127.0, dtype=torch.float32, device=p.device)
                        state["exp_avg"] = q_m
                        state["exp_avg_scales"] = scale_m
                        if has_slow_momentum:
                            # Slow momentum (AdEMAMix) always uses INT8 for stability
                            n_blks_s = (p.numel() + bsize - 1) // bsize
                            state["exp_avg_slow"] = torch.zeros_like(p, dtype=torch.int8)
                            state["exp_avg_slow_scales"] = torch.full((n_blks_s, 1), 1e-12 / 127.0, dtype=torch.float32, device=p.device)
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
                        if state.get("is_factored_v", False):
                            rows = p.shape[0]
                            cols = p.numel() // rows
                            state["exp_avg_sq_row"] = torch.zeros(rows, dtype=torch.float32, device=p.device)
                            state["exp_avg_sq_col"] = torch.zeros(cols, dtype=torch.float32, device=p.device)
                        elif state.get("is_8bit", False):
                            bsize = state.get("block_size_8bit", 2048)
                            n_blks_v = (p.numel() + bsize - 1) // bsize
                            if state.get("is_quad_v", False):
                                state["exp_avg_sq"] = torch.zeros((p.numel() // 4,), dtype=torch.uint8, device=p.device)
                            else:
                                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.uint8)
                            state["exp_avg_sq_scales"] = torch.full((n_blks_v, 1), 1e-12, dtype=torch.float32, device=p.device)
                        elif state.get("is_quad_v", False):
                            state["exp_avg_sq"] = torch.zeros((p.numel() // 4,), dtype=state_dtype_v, device=p.device)
                        else:
                            state["exp_avg_sq"] = torch.zeros_like(
                                p, dtype=state_dtype_v, memory_format=torch.preserve_format
                            )
                        if group["amsgrad"]:
                            # For 8bit/ultra, v is stored quantized (uint8) but the amsgrad max
                            # comparison is done on the dequantized float32 v_val.
                            # max_exp_avg_sq must therefore be float32, not uint8.
                            _is_quantized_v = state.get("is_8bit", False) or state.get("is_ultra", False)
                            max_v_dtype = torch.float32 if _is_quantized_v else state_dtype_v
                            state["max_exp_avg_sq"] = torch.zeros(
                                p.numel(), dtype=max_v_dtype, device=p.device
                            ).reshape(p.shape)

                    # FP16 dynamic range scaling — only meaningful for float16 variance states.
                    # 8-bit and ultra modes carry their own per-block scales (exp_avg_scales /
                    # exp_avg_sq_scales) and must NOT receive a stale scale_v / sqrt_scale_v
                    # that could confuse downstream dispatch or pollute state_dict exports.
                    _is_xla = (p.device.type == "xla" or group.get("backend") == "xla")
                    _is_quantized = state.get("is_8bit", False) or state.get("is_ultra", False)
                    if not _is_xla and not _is_quantized:
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
                    _is_quantized_v = state.get("is_8bit", False) or state.get("is_ultra", False)
                    max_v_dtype = torch.float32 if _is_quantized_v else state_dtype_v
                    state["max_exp_avg_sq"] = torch.zeros(
                        p.numel(), dtype=max_v_dtype, device=p.device
                    ).reshape(p.shape)
                _is_xla = (p.device.type == "xla" or group.get("backend") == "xla")
                _is_quantized = state.get("is_8bit", False) or state.get("is_ultra", False)
                if not _is_xla and not _is_quantized and "scale_v" not in state:
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
                    exp_avg_sqs.append(
                        state["exp_avg_sq"] if "exp_avg_sq" in state else state["exp_avg_sq_row"]
                    )
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

    def _build_cuda_8bit_partition(
        self,
        params: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
    ) -> dict:
        """Partition an 8-bit parameter group into quantized CUDA-8bit, factored-v, and micro tensors."""
        _supported_dtypes = (torch.float32, torch.bfloat16, torch.float16)
        invalid = {"eligible": False}
        if not params or not params[0].is_cuda:
            return invalid
        ref_device = params[0].device

        q_idxs: List[int] = []
        q_params: List[Tensor] = []
        q_ms: List[Tensor] = []
        q_m_scales: List[Tensor] = []
        q_vs: List[Tensor] = []
        q_v_scales: List[Tensor] = []
        q_bsizes: List[int] = []

        factored_idxs: List[int] = []
        factored_params: List[Tensor] = []

        micro_idxs: List[int] = []
        micro_params: List[Tensor] = []
        micro_ms: List[Tensor] = []
        micro_vs: List[Tensor] = []
        micro_can_fused = True

        for i, (p, m, v) in enumerate(zip(params, exp_avgs, exp_avg_sqs)):
            if p.numel() == 0:
                continue
            if p.device != ref_device or m.device != ref_device or v.device != ref_device:
                return invalid
            st = self.state[p]
            m_scale = st.get("exp_avg_scales", None)
            v_scale = st.get("exp_avg_sq_scales", None)

            if st.get("is_factored_v", False):
                rows = p.shape[0] if p.dim() >= 2 else 0
                cols = (p.numel() // rows) if rows > 0 else 0
                r_t = st.get("exp_avg_sq_row", None)
                c_t = st.get("exp_avg_sq_col", None)
                bsize = int(st.get("block_size_8bit", 0) or 0)
                is_6bit_m = bool(st.get("is_6bit_m", False))
                if is_6bit_m:
                    pad_m = (bsize - (p.numel() % bsize)) % bsize
                    padded_m = p.numel() + pad_m
                    if padded_m % 4 != 0:
                        padded_m += (4 - (padded_m % 4)) % 4
                    expected_m_elems = (padded_m // 4) * 3
                    valid_m_dt = (m.dtype == torch.uint8 and m.numel() == expected_m_elems)
                else:
                    valid_m_dt = (m.dtype == torch.int8 and m.numel() == p.numel())
                valid_factored = (
                    p.dim() >= 2
                    and ((rows >= 64 and cols >= 64) or (min(rows, cols) >= 8 and max(rows, cols) >= 128))
                    and valid_m_dt
                    and m_scale is not None
                    and r_t is not None
                    and c_t is not None
                    and r_t.device == ref_device
                    and c_t.device == ref_device
                    and m_scale.device == ref_device
                    and r_t.dtype == torch.float32
                    and c_t.dtype == torch.float32
                    and m_scale.dtype == torch.float32
                    and r_t.numel() == rows
                    and c_t.numel() == cols
                    and bsize in (64, 128, 256, 512, 1024, 2048)
                    and (not is_6bit_m or bsize % 4 == 0)
                    and p.dtype in _supported_dtypes
                    and "exp_avg_slow" not in st
                )
                if not valid_factored:
                    return invalid
                factored_idxs.append(i)
                factored_params.append(p)
                continue

            has_quant_state = (
                st.get("is_8bit", False)
                or st.get("is_ultra", False)
                or m.dtype in (torch.int8, torch.uint8)
                or v.dtype == torch.uint8
                or m_scale is not None
                or v_scale is not None
            )
            if has_quant_state:
                bsize = int(st.get("block_size_8bit", st.get("ultra_block_size", 0)) or 0)
                is_4bit_m = (m.dtype == torch.uint8)
                is_quad_v = is_4bit_m and (p.numel() >= 4) and (p.numel() % 4 == 0) and (v.numel() * 4 == p.numel())
                expected_m_numel = (p.numel() + 1) // 2 if is_4bit_m else p.numel()
                expected_v_numel = (p.numel() // 4) if is_quad_v else p.numel()
                valid_q = (
                    v.dtype == torch.uint8
                    and m.dtype in (torch.int8, torch.uint8)
                    and m_scale is not None
                    and v_scale is not None
                    and m_scale.device == ref_device
                    and v_scale.device == ref_device
                    and m_scale.dtype == torch.float32
                    and v_scale.dtype == torch.float32
                    and bsize in (64, 128, 256, 512, 1024, 2048)
                    and p.dtype in _supported_dtypes
                    and v.numel() == expected_v_numel
                    and m.numel() == expected_m_numel
                    and p.is_contiguous()
                    and m.is_contiguous()
                    and v.is_contiguous()
                    and m_scale.is_contiguous()
                    and v_scale.is_contiguous()
                    and "exp_avg_slow" not in st
                )
                if not valid_q:
                    return invalid
                q_idxs.append(i)
                q_params.append(p)
                q_ms.append(m)
                q_m_scales.append(m_scale)
                q_vs.append(v)
                q_v_scales.append(v_scale)
                q_bsizes.append(bsize)
            else:
                if (
                    p.dtype not in _supported_dtypes
                    or m.dtype not in _supported_dtypes
                    or v.dtype not in _supported_dtypes
                    or p.numel() != m.numel()
                    or p.numel() != v.numel()
                    or "exp_avg_slow" in st
                ):
                    return invalid
                micro_idxs.append(i)
                micro_params.append(p)
                micro_ms.append(m)
                micro_vs.append(v)
                if not (
                    p.is_contiguous()
                    and m.is_contiguous()
                    and v.is_contiguous()
                    and (p.dtype, m.dtype, v.dtype) in _CUDA_DTYPE_TRIOS
                ):
                    micro_can_fused = False

        if not q_params and not factored_params:
            return invalid

        return {
            "eligible": True,
            "ref_device": ref_device,
            "q_idxs": q_idxs,
            "q_params": q_params,
            "q_ms": q_ms,
            "q_m_scales": q_m_scales,
            "q_vs": q_vs,
            "q_v_scales": q_v_scales,
            "q_bsizes": q_bsizes,
            "factored_idxs": factored_idxs,
            "factored_params": factored_params,
            "micro_idxs": micro_idxs,
            "micro_params": micro_params,
            "micro_ms": micro_ms,
            "micro_vs": micro_vs,
            "micro_can_fused": micro_can_fused,
        }

    @staticmethod
    @torch.no_grad()
    def _execute_micro_adamw(
        micro_params: List[Tensor],
        micro_grads: List[Tensor],
        micro_ms: List[Tensor],
        micro_vs: List[Tensor],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
        decay_norm_only: bool,
        inv_scale: float,
        cautious: bool,
    ) -> None:
        """Update unquantized sub-64-element micro tensors (biases/norms) alongside fused 8-bit CUDA kernels."""
        bc1 = 1.0 - (beta1 ** step)
        bc2_sqrt = math.sqrt(1.0 - (beta2 ** step))
        step_size = lr * bc2_sqrt / bc1
        for p, g, m, v in zip(micro_params, micro_grads, micro_ms, micro_vs):
            g_f = g.float()
            if inv_scale != 1.0:
                g_f.mul_(inv_scale)
            g_f.nan_to_num_(nan=0.0, posinf=65504.0, neginf=-65504.0)

            m_f = m.float()
            m_f.lerp_(g_f, 1.0 - beta1)
            if m_f is not m:
                m.copy_(m_f.to(m.dtype))

            v_f = v.float()
            v_f.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
            v_f.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if v_f is not v:
                v.copy_(v_f.to(v.dtype))

            eff_eps = eps
            denom = v_f.sqrt().add_(eff_eps * bc2_sqrt).clamp_min_(eff_eps)
            num = m_f
            if cautious:
                mask = (num * g_f >= 0).to(num.dtype)
                mask_mean = mask.mean().clamp(min=1e-3)
                num = num * (mask / mask_mean)
            update = num / denom

            p_f = p.float()
            if weight_decay != 0.0 and (not decay_norm_only or p.dim() >= 2):
                p_f.mul_(max(1.0 - lr * weight_decay, 0.0))
            p_f.add_(update, alpha=-step_size)
            if p.dtype == torch.float16:
                p_f.clamp_(-65504.0, 65504.0)
            if p_f is not p:
                p.copy_(p_f.to(p.dtype))

    @torch.no_grad()
    def _execute_cuda_8bit_partition(
        self,
        partition: dict,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
        decay_norm_only: bool = False,
        inv_scale: float = 1.0,
        stochastic: bool = True,
        cautious: bool = False,
    ) -> bool:
        """Execute partitioned 8-bit/4-bit CUDA update with zero partial mutation on failure."""
        if not partition or not partition.get("eligible", False):
            return False
        q_params = partition.get("q_params", [])
        factored_idxs = partition.get("factored_idxs", [])
        factored_params = partition.get("factored_params", [])
        if not q_params and not factored_params:
            return False

        q_grads: List[Tensor] = []
        for idx, p in zip(partition["q_idxs"], q_params):
            g = grads[idx]
            if g.device != p.device or g.numel() != p.numel():
                return False
            g_c = g.to(p.dtype) if g.dtype != p.dtype else g
            q_grads.append(g_c if g_c.is_contiguous() else g_c.contiguous())

        micro_idxs = partition["micro_idxs"]
        micro_grads: List[Tensor] = []
        if micro_idxs:
            for idx, p in zip(micro_idxs, partition["micro_params"]):
                g = grads[idx]
                if g.device != p.device or g.numel() != p.numel():
                    return False
                micro_grads.append(g)

        factored_grads: List[Tensor] = []
        if factored_idxs:
            for idx, p in zip(factored_idxs, factored_params):
                g = grads[idx]
                if g.device != p.device or g.numel() != p.numel():
                    return False
                factored_grads.append(g)

        if q_params:
            ok = step_fused_8bit(
                q_params,
                q_grads,
                partition["q_ms"],
                partition["q_m_scales"],
                partition["q_vs"],
                partition["q_v_scales"],
                partition["q_bsizes"],
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                step=step,
                decay_norm_only=decay_norm_only,
                inv_scale=inv_scale,
                stochastic=stochastic,
                cautious=cautious,
            )
            if not ok:
                return False

        if micro_idxs:
            micro_params = partition["micro_params"]
            micro_ms = partition["micro_ms"]
            micro_vs = partition["micro_vs"]
            used_fused_micro = False
            if (
                partition["micro_can_fused"]
                and inv_scale == 1.0
                and not cautious
                and all(g.dtype == p.dtype and g.is_contiguous() for g, p in zip(micro_grads, micro_params))
            ):
                used_fused_micro = bool(
                    step_fused(
                        micro_params,
                        micro_grads,
                        micro_ms,
                        micro_vs,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay,
                        step=step,
                        decay_norm_only=decay_norm_only,
                    )
                )

            if not used_fused_micro:
                self._execute_micro_adamw(
                    micro_params,
                    micro_grads,
                    micro_ms,
                    micro_vs,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    step=step,
                    decay_norm_only=decay_norm_only,
                    inv_scale=inv_scale,
                    cautious=cautious,
                )

        if factored_params:
            unhandled_factored: List[Tensor] = []
            unhandled_grads: List[Tensor] = []
            if is_triton_available():
                for p, g in zip(factored_params, factored_grads):
                    st = self.state[p]
                    apply_wd = weight_decay != 0.0 and (not decay_norm_only or p.dim() >= 2)
                    handled = triton_factored_v_step(
                        p,
                        g,
                        st["exp_avg"],
                        st["exp_avg_sq_row"],
                        st["exp_avg_sq_col"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay,
                        step=step,
                        apply_wd=apply_wd,
                        inv_scale=inv_scale,
                        cautious=cautious,
                        stochastic=st.get("stochastic_rounding", stochastic),
                        is_6bit_m=st.get("is_6bit_m", False),
                        exp_avg_scales=st.get("exp_avg_scales", None),
                        block_size=st.get("block_size_8bit", 256),
                    )
                    if not handled:
                        unhandled_factored.append(p)
                        unhandled_grads.append(g)
            else:
                unhandled_factored = factored_params
                unhandled_grads = factored_grads

            if unhandled_factored:
                _execute_factored_v_batch(
                    unhandled_factored,
                    unhandled_grads,
                    self.state,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    step=step,
                    decay_norm_only=decay_norm_only,
                    inv_scale=inv_scale,
                    cautious=cautious,
                    stochastic=stochastic,
                )

        return True

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

        global _AMP_SCALED_THIS_BACKWARD
        self._has_stepped = True
        if getattr(self, "_auto_hook_queue", None):
            self._flush_auto_8bit_hook_queue()
        if getattr(self, "_auto_hook_stepped_count", 0) > 0:
            if not getattr(self, "_grad_accum_steps_known", False) and getattr(self, "_anchor_backward_count", 0) > 1:
                self._grad_accum_steps = self._anchor_backward_count
                self._grad_accum_steps_known = True
            self._auto_hook_stepped_count = 0
            self._anchor_backward_count = 0
        _AMP_SCALED_THIS_BACKWARD = False

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
                    found_inf = None
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

            needs_host_sanitize = (
                group.get("precision", "compact") not in ("8bit", "ultra")
                or group.get("clip_grad_norm", None) is not None
                or group.get("grad_centralization", False)
            )
            grads, backend_grad_scale = _prepare_grads(
                grads, group, grad_scale, sanitize=needs_host_sanitize
            )

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

            # Google TPU Global Graph Compilation (PyTorch XLA)
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

            # PATH 2 / PATH 3 viability evaluation
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
                and not (precision in ("8bit", "ultra") and beta3 is not None)
            )
            if can_use_triton and precision in ("8bit", "ultra"):
                for p in params_with_grad:
                    b = self.state[p].get("block_size_8bit", self.state[p].get("ultra_block_size", None))
                    if b is not None and (b <= 0 or (b & (b - 1)) != 0):
                        can_use_triton = False
                        break

            has_8bit_quantized = (
                precision in ("8bit", "ultra")
                and not lion
                and bool(exp_avg_sqs)
                and any(
                    self.state[p].get("is_8bit", False) or exp_avg_sqs[i].dtype == torch.uint8
                    for i, p in enumerate(params_with_grad)
                )
            )
            can_use_cuda_fused_8bit = (
                precision in ("8bit", "ultra")
                and backend in (None, "auto", "cuda")
                and is_cuda
                and not nesterov
                and beta3 is None
                and not lion
                and not amsgrad
                and not capturable
                and not compiling
                and not maximize
                and not has_complex
                and (backend_grad_scale is None or (not is_capturing and not compiling))
                and (found_inf is None or (not is_capturing and not compiling))
                and has_8bit_quantized
                and is_cuda_kernel_available()
            )

            steps_are_uniform = len(state_steps) <= 1 or all(
                _get_value(s) == _get_value(state_steps[0]) for s in state_steps[1:]
            )

            # Ultra / 8-Bit (Quantized Optimizer State Execution)
            # If precision is 8-bit or ultra and neither CUDA 8-bit nor Triton GPU
            # kernel is viable (e.g. CPU, MPS, or custom backend), route to _step_ultra.
            if precision in ("8bit", "ultra") and not (
                (can_use_triton or can_use_cuda_fused_8bit) and steps_are_uniform
            ):
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
                        path_type="8bit" if precision == "8bit" else "ultra",
                        params=params_with_grad,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        state_steps=state_steps,
                        exp_avgs_slow=exp_avgs_slow if beta3 is not None else None,
                        max_exp_avg_sqs=max_exp_avg_sqs if amsgrad else None,
                        decay_norm_only=decay_norm_only,
                        extra={"group": group, "precision": precision},
                    )
                continue

            inv_scale = 1.0
            if backend_grad_scale is not None and not is_capturing and not compiling:
                inv_scale = float(backend_grad_scale.reciprocal().item()) if hasattr(backend_grad_scale, "item") else (1.0 / float(backend_grad_scale))

            # Triton and custom CUDA kernels require moment buffers to share the same storage dtype.
            # When ApexW uses split precision (e.g. m in bf16 and v in fp32 for extreme beta2),
            # we route through the multi-tensor PyTorch foreach engine.
            _states_have_mixed_moment_dtypes = any(
                self.state[p].get("state_dtype_v", self.state[p].get("state_dtype", p.dtype))
                != self.state[p].get("state_dtype", p.dtype)
                for p in params_with_grad
                if p in self.state
            )
            _has_factored_v = any(
                self.state[p].get("is_factored_v", False)
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
                and not _has_factored_v
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
                and not _dynamic_scaling_active
                and all(
                    p.device == params_with_grad[0].device
                    and g.device == p.device and m.device == p.device and v.device == p.device
                    and p.is_contiguous() and g.is_contiguous()
                    and m.is_contiguous() and v.is_contiguous()
                    and p.numel() == g.numel() == m.numel()
                    and (v.numel() == p.numel() or (p.numel() >= 4 and p.numel() % 4 == 0 and v.numel() * 4 == p.numel()))
                    and g.dtype == p.dtype
                    and (p.dtype, m.dtype, v.dtype) in _CUDA_DTYPE_TRIOS
                    for p, g, m, v in zip(params_with_grad, grads, exp_avgs, exp_avg_sqs)
                )
                and is_cuda_kernel_available()
            )
            if can_use_cuda_fused:
                can_use_triton = False

            # PyTorch Native C++/CUDA Fused Multi-Tensor Fast Path
            # Used for full-precision CUDA/CPU when custom CUDA/Triton kernels are not active,
            # or when fused=True is explicitly requested.
            all_fp32 = all(p.dtype == torch.float32 for p in params_with_grad)
            states_fp32 = not _dynamic_scaling_active and not _states_have_mixed_moment_dtypes and all(
                exp_avgs[i].dtype == torch.float32 and exp_avgs[i].numel() == exp_avg_sqs[i].numel()
                for i in range(len(exp_avgs))
            )
            states_homogeneous = (
                not _states_have_mixed_moment_dtypes
                and len(exp_avgs) == len(exp_avg_sqs)
                and all(
                    exp_avgs[i].dtype == exp_avg_sqs[i].dtype
                    and exp_avgs[i].dtype == params_with_grad[i].dtype
                    and exp_avgs[i].numel() == exp_avg_sqs[i].numel()
                    for i in range(len(exp_avgs))
                )
            )
            # _fused_adamw_ requires param dtype == state dtype on the same device.
            # 'full' precision stores states as float32 regardless of param dtype.
            # When params are FP16/BF16 with float32 states, _fused_adamw_ crashes.
            # Such groups must fall through to _multi_tensor_step which handles mixed precision.
            _fused_safe = (
                (all_fp32 and states_fp32)   # fp32 params + fp32 states → safe
                or states_homogeneous        # states match param dtype → safe
            )
            _homogeneous_fast_path = _fused_safe

            # Custom kernel preference: don't let PyTorch's native _fused_adamw_ preempt
            # custom CUDA or Triton implementations when custom kernels are requested or viable,
            # unless the user explicitly requested fused=True or tensors are natively homogeneous
            # (where native _fused_adamw_ achieves peak C++/CUDA SIMD throughput).
            custom_kernel_viable = (can_use_cuda_fused or can_use_cuda_fused_8bit or can_use_triton)
            prefers_custom = (backend in ("cuda", "triton")) or (
                backend in (None, "auto")
                and custom_kernel_viable
                and fused is not True
                and not _homogeneous_fast_path
            )

            can_native_fused = (
                not prefers_custom
                and not _has_factored_v
                and _fused_safe  # param dtype must be compatible with state dtype for _fused_adamw_
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
                        extra={"group": group, "precision": precision},
                    )
                continue

            if can_use_cuda_fused_8bit and steps_are_uniform:
                cuda8_partition = self._build_cuda_8bit_partition(
                    params_with_grad, exp_avgs, exp_avg_sqs
                )
                if cuda8_partition.get("eligible", False):
                    step_val = int(_get_value(state_steps[0])) + 1
                    torch._foreach_add_(state_steps, 1)
                    success = self._execute_cuda_8bit_partition(
                        cuda8_partition,
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
                        inv_scale=inv_scale,
                        stochastic=group.get("stochastic_rounding", True),
                        cautious=cautious,
                    )
                    if success:
                        if not is_capturing and not compiling:
                            plan = FastExecutionPlan(
                                path_type="cuda_8bit",
                                params=params_with_grad,
                                exp_avgs=exp_avgs,
                                exp_avg_sqs=exp_avg_sqs,
                                state_steps=state_steps,
                                decay_norm_only=decay_norm_only,
                                extra={"group": group, "precision": precision},
                            )
                            plan._cuda_8bit_partition = cuda8_partition
                            self._execution_plans[group_id] = plan
                        continue
                    torch._foreach_sub_(state_steps, 1)

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

                bsize_triton = (
                    [
                        self.state[p].get("block_size_8bit", self.state[p].get("ultra_block_size", None))
                        for p in params_with_grad
                    ]
                    if (m_scales is not None or v_scales is not None)
                    else None
                )
                success, _ = _dispatch_triton_with_factored(
                    state=self.state,
                    params=params_with_grad,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    exp_avgs_slow=exp_avgs_slow if beta3 is not None else None,
                    m_scales=m_scales,
                    v_scales=v_scales,
                    bsize=bsize_triton,
                    scales_v=triton_scales_v,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=wd,
                    step=step_val,
                    decay_norm_only=decay_norm_only,
                    inv_scale=inv_scale,
                    beta3=beta3,
                    alpha=alpha,
                    cautious=cautious,
                    stochastic=group.get("stochastic_rounding", True),
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
                            extra={"group": group, "precision": precision},
                        )
                    continue
                # If Triton execution did not succeed, roll back step counter for fallback
                torch._foreach_sub_(state_steps, 1)

            # 8-bit / ultra precision: CUDA 8-bit and Triton are the only accelerated paths.
            # If neither succeeds (e.g. CPU fallback, non-contiguous tensors, or missing kernel),
            # always fall back to _step_ultra — never let 8bit/ultra fall through to PATH 3
            # (FP32/BF16 C++ fused CUDA) or PATH 4 (_multi_tensor_step), which expect float moments.
            if precision in ("8bit", "ultra"):
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
                        path_type="8bit" if precision == "8bit" else "ultra",
                        params=params_with_grad,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        state_steps=state_steps,
                        exp_avgs_slow=exp_avgs_slow if beta3 is not None else None,
                        max_exp_avg_sqs=max_exp_avg_sqs if amsgrad else None,
                        decay_norm_only=decay_norm_only,
                        extra={"group": group, "precision": precision},
                    )
                continue


            # Custom C++/CUDA 128-bit Vectorized Fused Kernel
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
                            extra={"group": group, "precision": precision},
                        )
                    continue
                # If CUDA execution did not succeed, roll back step counter for fallback
                torch._foreach_sub_(state_steps, 1)

            # Apple Silicon (MPS) & PyTorch Native Multi-Tensor
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
                        stochastic=group.get("stochastic_rounding", True),
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
                            extra={
                                "has_complex": has_complex,
                                "group": group,
                                "factored_mask": [self.state[p].get("is_factored_v", False) for p in params_with_grad] if _has_factored_v else None,
                            },
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
                        extra={"has_complex": has_complex, "group": group, "precision": precision},
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

        # Update step counters in-place on device (zero host barrier)
        for st in state_steps:
            st.add_(1)

        # Compute shared device tensors for bias corrections (per-step, inside param loop)
        beta1_t = torch.tensor(beta1, dtype=torch.float32, device=device)
        beta2_t = torch.tensor(beta2, dtype=torch.float32, device=device)
        beta3_t = torch.tensor(beta3, dtype=torch.float32, device=device) if beta3 is not None else None

        # Fused tensor updates
        for i, p in enumerate(params):
            # use per-parameter step_t, not always state_steps[0]
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

            g = torch.nan_to_num(grads[i], nan=0.0, posinf=65504.0, neginf=-65504.0)
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
            p_state = self.state.get(p, {})
            is_8bit = p_state.get("is_8bit", False) or m.dtype in (torch.int8, torch.uint8)
            is_ultra = p_state.get("is_ultra", False)

            if p_state.get("is_factored_v", False):
                _apply_factored_v_int8_update(
                    p,
                    g.float(),
                    p_state,
                    beta1=beta1,
                    beta2=beta2,
                    step_size=step_size,
                    eps_scaled=eps_scaled,
                    effective_eps=eps,
                    nesterov=nesterov,
                    cautious=cautious,
                    stochastic=p_state.get("stochastic_rounding", True),
                    block_size=p_state.get("block_size_8bit", 256),
                )
                continue

            g_f = g.float()
            if is_8bit or is_ultra:
                bs = p_state.get("block_size_8bit", p_state.get("ultra_block_size", 256))
                stoch = p_state.get("stochastic_rounding", True)
                is_4bit_m = p_state.get("is_4bit_m", False)
                is_quad_v = bool(p_state.get("is_quad_v", False)) or (is_4bit_m and p.numel() >= 4 and p.numel() % 4 == 0 and v.numel() * 4 == p.numel())
                if is_4bit_m:
                    m_deq = _dequantize_blockwise_int4(m, p_state["exp_avg_scales"], numel=p.numel(), block_size=bs).view(p.shape)
                else:
                    m_deq = _dequantize_blockwise_int8(m, p_state["exp_avg_scales"], block_size=bs)
                m_deq.lerp_(g_f, 1.0 - beta1)

                m_slow_deq = None
                if m_slow is not None and "exp_avg_slow_scales" in p_state:
                    m_slow_deq = _dequantize_blockwise_int8(m_slow, p_state["exp_avg_slow_scales"], block_size=bs)
                    m_slow_deq.lerp_(g_f, 1.0 - beta3)

                if is_8bit and "exp_avg_sq_scales" in p_state:
                    if is_quad_v:
                        v_bs = max(1, bs // 4)
                        v_deq = _dequantize_blockwise_uint8(v, p_state["exp_avg_sq_scales"], block_size=v_bs)
                        g_sq_q = g_f.reshape(-1, 4).square().mean(dim=1)
                        v_deq.mul_(beta2).add_(g_sq_q, alpha=1.0 - beta2)
                        v_deq.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        denom = v_deq.sqrt().add_(eps_scaled).clamp_min_(eps).unsqueeze(1).expand(-1, 4).reshape(p.shape)
                        _quantize_blockwise_uint8(v_deq, block_size=v_bs, stochastic=stoch, out_q=v, out_scales=p_state["exp_avg_sq_scales"], old_scales=p_state["exp_avg_sq_scales"])
                    else:
                        v_deq = _dequantize_blockwise_uint8(v, p_state["exp_avg_sq_scales"], block_size=bs)
                        v_deq.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                        v_deq.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        if amsgrad:
                            torch.maximum(max_v, v_deq, out=max_v)
                            v_target = max_v
                        else:
                            v_target = v_deq
                        denom = v_target.sqrt().add_(eps_scaled).clamp_min_(eps)
                        _quantize_blockwise_uint8(v_deq, block_size=bs, stochastic=stoch, out_q=v, out_scales=p_state["exp_avg_sq_scales"], old_scales=p_state["exp_avg_sq_scales"])
                else:
                    v_f = v.float().mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                    v.copy_(v_f.to(v.dtype))
                    denom = v_f.sqrt().add_(eps_scaled)

                if m_slow_deq is not None:
                    num = m_slow_deq.add(m_deq, alpha=slow_ratio)
                elif nesterov:
                    num = torch.lerp(g_f, m_deq, beta1)
                elif is_8bit and not cautious:
                    m_look = torch.lerp(g_f, m_deq, beta1)
                    gain = torch.where(m_deq * g_f >= 0.0, 1.0 + 0.20 * beta1, 1.0 - 0.35 * beta1)
                    num = m_look.mul_(gain)
                else:
                    num = m_deq

                if cautious:
                    mask = (g_f * num >= 0).to(p.dtype)
                    mask_mean = mask.mean().clamp(min=1e-3)
                    num = num * (mask / mask_mean)

                if p.dtype == torch.float32:
                    p.addcdiv_(num, denom, value=-step_size)
                else:
                    p.sub_((num / denom).mul_(step_size).to(p.dtype))

                if is_4bit_m:
                    _quantize_blockwise_int4(m_deq, block_size=bs, stochastic=stoch, out_q=m, out_scales=p_state["exp_avg_scales"], old_scales=p_state["exp_avg_scales"])
                else:
                    _quantize_blockwise_int8(m_deq, block_size=bs, stochastic=stoch, out_q=m, out_scales=p_state["exp_avg_scales"], old_scales=p_state["exp_avg_scales"])
                continue

            # Lion mode
            if lion:
                g_comp = g if g.dtype == m.dtype else g.to(m.dtype)
                update = torch.lerp(g_comp, m, beta1).sign()
                p.add_(update.to(p.dtype), alpha=-lr)
                m.lerp_(g_comp, 1.0 - beta2)
                continue

            # Floating-point AdamW mode (Compact BF16/FP16, Full FP32)
            m_f = m.float().lerp_(g_f, 1.0 - beta1)
            m.copy_(m_f.to(m.dtype))
            if m_slow is not None:
                m_slow_f = m_slow.float().lerp_(g_f, 1.0 - beta3)
                m_slow.copy_(m_slow_f.to(m_slow.dtype))

            is_quad_v_c = (p.numel() >= 4 and p.numel() % 4 == 0 and v.numel() * 4 == p.numel())
            if is_quad_v_c:
                g_sq_q = g_f.reshape(-1, 4).square().mean(dim=1)
                v_f = v.float().mul_(beta2).add_(g_sq_q, alpha=1.0 - beta2)
                v.copy_(v_f.to(v.dtype))
                denom = v_f.sqrt().add_(eps_scaled).unsqueeze(1).expand(-1, 4).reshape(p.shape)
            else:
                v_f = v.float().mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                v.copy_(v_f.to(v.dtype))
                if amsgrad:
                    torch.maximum(max_v, v_f, out=max_v)
                    v_target = max_v
                else:
                    v_target = v_f
                denom = v_target.sqrt().add_(eps_scaled)

            if m_slow is not None:
                num = m_f.add(m_slow_f, alpha=slow_ratio)
            elif nesterov:
                num = torch.lerp(g_f, m_f, beta1)
            else:
                num = m_f

            if cautious:
                mask = (g_f * num >= 0).to(p.dtype)
                mask_mean = mask.mean().clamp(min=1e-3)
                num = num * (mask / mask_mean)

            if p.dtype == torch.float32:
                p.addcdiv_(num, denom, value=-step_size)
            else:
                update_val = (num / denom).mul_(step_size)
                _copy_fp32_to_16bit_sr_(p, p.float().sub_(update_val), stochastic=True)

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
        """Quantized optimizer state execution path.

        Handles execution for both quantized precision modes:
        1. precision="8bit": Sub-1-Byte NF4 INT4 momentum (0.50 B/p) + Quad 24-octave Log-UINT8 variance (0.25 B/p)
           with Fused Nesterov-SNR Lookahead and dual stochastic dithering (~0.75 - 2.0 bytes/parameter).
        2. precision="ultra": Blockwise INT8 momentum + 16-bit variance (3 bytes/parameter).
        Small 1D/normalization tensors (< 64 elements) are updated in compact 16-bit storage.
        """
        compiling = is_compiling()
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if found_inf is not None and not is_capturing and not compiling:
            if found_inf.item() > 0:
                return

        inv_scale = None
        if grad_scale is not None:
            inv_scale = grad_scale.reciprocal() if hasattr(grad_scale, "reciprocal") else (1.0 / float(grad_scale))

        decay_factor = max(1.0 - lr * weight_decay, 0.0) if weight_decay != 0 else 1.0
        gain_hi = 1.0 + 0.20 * beta1
        gain_lo = 1.0 - 0.35 * beta1

        sq_idx = 0
        for i, p in enumerate(params):
            step_t = state_steps[i]
            step_t.add_(1)
            step = _get_value(step_t)
            bc1 = 1.0 - (beta1 ** step)
            bc2 = 1.0 - (beta2 ** step)
            bc2_sqrt = math.sqrt(bc2)
            step_size = lr * bc2_sqrt / bc1
            effective_eps = eps
            eps_scaled = effective_eps * bc2_sqrt

            p_state = self.state[p]
            is_8bit = p_state.get("is_8bit", False)
            is_ultra = p_state.get("is_ultra", False)
            block_size = p_state.get("block_size_8bit", p_state.get("ultra_block_size", 256))
            stochastic = p_state.get("stochastic_rounding", True)
            apply_wd = weight_decay != 0 and (not decay_norm_only or p.dim() >= 2)

            if (not lion) and p_state.get("is_factored_v", False):
                sq_idx += 1
                _apply_factored_v_int8_update(
                    p,
                    grads[i],
                    p_state,
                    beta1=beta1,
                    beta2=beta2,
                    step_size=step_size,
                    eps_scaled=eps_scaled,
                    effective_eps=effective_eps,
                    nesterov=nesterov,
                    cautious=cautious,
                    stochastic=stochastic,
                    block_size=block_size,
                    wd_factor=decay_factor if apply_wd else None,
                    inv_scale=inv_scale,
                    maximize=maximize,
                    sanitize_g=True,
                )
                continue

            if (
                (not lion)
                and is_8bit
                and not p_state.get("is_6bit_m", False)
                and not amsgrad
                and beta3 is None
                and not cautious
                and p.numel() > _FACTORED_CHUNK_ELEMS
                and p.is_contiguous()
                and p_state["exp_avg"].is_contiguous()
                and exp_avg_sqs[sq_idx].is_contiguous()
            ):
                v = exp_avg_sqs[sq_idx]
                sq_idx += 1
                p_flat = p.reshape(-1)
                g_flat = grads[i].reshape(-1) if grads[i].is_contiguous() else grads[i].contiguous().reshape(-1)
                m_buf_flat = p_state["exp_avg"].reshape(-1)
                v_buf_flat = v.reshape(-1)
                m_scales = p_state["exp_avg_scales"]
                v_scales = p_state["exp_avg_sq_scales"]
                numel = p.numel()
                is_4bit_m_c = p_state.get("is_4bit_m", False)
                is_quad_v_c = bool(p_state.get("is_quad_v", False)) or (is_4bit_m_c and numel >= 4 and numel % 4 == 0 and v.numel() * 4 == numel)
                v_bs_c = max(1, block_size // 4) if is_quad_v_c else block_size
                chunk_elems = ((_FACTORED_CHUNK_ELEMS // block_size) or 1) * block_size
                for off in range(0, numel, chunk_elems):
                    end = min(off + chunk_elems, numel)
                    chunk_len = end - off
                    blk0 = off // block_size
                    is_last = (end >= numel)
                    sc_m = m_scales[blk0:] if is_last else m_scales[blk0 : blk0 + chunk_len // block_size]
                    sc_v = v_scales[blk0:] if is_last else v_scales[blk0 : blk0 + chunk_len // block_size]
                    g_c = _prep_grad_chunk_fp32(
                        g_flat[off:end], sanitize_g=True, maximize=maximize, inv_scale=inv_scale
                    )
                    if is_4bit_m_c:
                        m_sl = m_buf_flat[off // 2 : (end + 1) // 2]
                        m_c = _dequantize_blockwise_int4(m_sl, sc_m, numel=chunk_len, block_size=block_size)
                    else:
                        m_sl = m_buf_flat[off:end]
                        m_c = _dequantize_blockwise_int8(m_sl, sc_m, block_size=block_size)
                    m_c.lerp_(g_c, 1.0 - beta1)

                    if nesterov:
                        num_c = torch.lerp(g_c, m_c, beta1)
                    else:
                        m_look = torch.lerp(g_c, m_c, beta1)
                        gain = torch.where(m_c * g_c >= 0.0, gain_hi, gain_lo)
                        num_c = m_look.mul_(gain)

                    if is_quad_v_c:
                        v_sl = v_buf_flat[off // 4 : end // 4]
                        v_c = _dequantize_blockwise_uint8(v_sl, sc_v, block_size=v_bs_c)
                        g_sq_q = g_c.view(-1, 4).square().mean(dim=1)
                        v_c.mul_(beta2).add_(g_sq_q, alpha=1.0 - beta2)
                        v_c.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        denom_q = v_c.sqrt().add_(eps_scaled).clamp_min_(effective_eps)
                        num_c.view(-1, 4).div_(denom_q.unsqueeze(1))
                        num_c.mul_(step_size)
                        denom_c = num_c
                    else:
                        v_sl = v_buf_flat[off:end]
                        v_c = _dequantize_blockwise_uint8(v_sl, sc_v, block_size=block_size)
                        v_c.mul_(beta2).addcmul_(g_c, g_c, value=1.0 - beta2)
                        v_c.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        denom_c = v_c.sqrt().add_(eps_scaled).clamp_min_(effective_eps)
                        torch.div(num_c, denom_c, out=denom_c)
                        denom_c.mul_(step_size)

                    p_sl = p_flat[off:end]
                    if apply_wd:
                        if p.dtype == torch.float32:
                            p_sl.mul_(decay_factor)
                            p_sl.sub_(denom_c)
                        else:
                            pf = p_sl.float().mul_(decay_factor).sub_(denom_c)
                            _copy_fp32_to_16bit_sr_(p_sl, pf, stochastic=stochastic)
                    else:
                        if p.dtype == torch.float32:
                            p_sl.sub_(denom_c)
                        else:
                            pf = p_sl.float().sub_(denom_c)
                            _copy_fp32_to_16bit_sr_(p_sl, pf, stochastic=stochastic)
                    if is_4bit_m_c:
                        _quantize_blockwise_int4(
                            m_c, block_size=block_size, stochastic=stochastic,
                            out_q=m_sl, out_scales=sc_m, old_scales=sc_m,
                        )
                    else:
                        _quantize_blockwise_int8(
                            m_c, block_size=block_size, stochastic=stochastic,
                            out_q=m_sl, out_scales=sc_m, old_scales=sc_m,
                        )
                    _quantize_blockwise_uint8(
                        v_c, block_size=v_bs_c, stochastic=stochastic,
                        out_q=v_sl, out_scales=sc_v, old_scales=sc_v,
                    )
                continue

            g_f = torch.nan_to_num(grads[i], nan=0.0, posinf=65504.0, neginf=-65504.0).float()
            if maximize:
                g_f.neg_()
            if inv_scale is not None:
                g_f.mul_(inv_scale)

            # Decoupled weight decay
            if apply_wd:
                p.mul_(decay_factor)

            # Lion mode: only tracks first-order momentum
            if lion:
                if is_8bit or is_ultra:
                    m = _dequantize_blockwise_int8(p_state["exp_avg"], p_state["exp_avg_scales"], block_size=block_size)
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
                    g_c = g_f.to(m.dtype)
                    update = torch.lerp(g_c, m, beta1).sign()
                    p.add_(update.to(p.dtype), alpha=-lr)
                    m.lerp_(g_c, 1.0 - beta2)
                continue

            # AdamW / AdEMAMix path
            v = exp_avg_sqs[sq_idx]
            max_v = max_exp_avg_sqs[sq_idx] if amsgrad else None
            sq_idx += 1

            if is_8bit:
                is_4bit_m = p_state.get("is_4bit_m", False)
                m_numel = p_state.get("m_numel", p.numel())
                is_quad_v = bool(p_state.get("is_quad_v", False)) or (is_4bit_m and m_numel >= 4 and m_numel % 4 == 0 and p_state["exp_avg_sq"].numel() * 4 == m_numel)
                if is_4bit_m:
                    m = _dequantize_blockwise_int4(
                        p_state["exp_avg"], p_state["exp_avg_scales"], numel=m_numel, block_size=block_size
                    ).view(p.shape)  # reshape flat output to match parameter shape
                else:
                    m = _dequantize_blockwise_int8(p_state["exp_avg"], p_state["exp_avg_scales"], block_size=block_size)
                m.lerp_(g_f, 1.0 - beta1)

                m_slow = None
                if beta3 is not None and "exp_avg_slow_scales" in p_state:
                    m_slow = _dequantize_blockwise_int8(
                        p_state["exp_avg_slow"], p_state["exp_avg_slow_scales"], block_size=block_size
                    )
                    m_slow.lerp_(g_f, 1.0 - beta3)

                if is_quad_v:
                    v_bs = max(1, block_size // 4)
                    v_val = _dequantize_blockwise_uint8(p_state["exp_avg_sq"], p_state["exp_avg_sq_scales"], block_size=v_bs)
                    g_sq_q = g_f.reshape(-1, 4).square().mean(dim=1)
                    v_val.mul_(beta2).add_(g_sq_q, alpha=1.0 - beta2)
                    v_val.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    denom = v_val.sqrt().add_(eps_scaled).clamp_min_(effective_eps).unsqueeze(1).expand(-1, 4).reshape(p.shape)
                else:
                    v_bs = block_size
                    v_val = _dequantize_blockwise_uint8(p_state["exp_avg_sq"], p_state["exp_avg_sq_scales"], block_size=block_size)
                    v_val.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                    v_val.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    if amsgrad:
                        torch.maximum(max_v, v_val, out=max_v)
                        v_active = max_v
                    else:
                        v_active = v_val
                    denom = v_active.sqrt().add_(eps_scaled).clamp_min_(effective_eps)

                if m_slow is not None:
                    slow_ratio = alpha * (bc1 / (1.0 - (beta3 ** step)))
                    num = m.add(m_slow, alpha=slow_ratio)
                elif nesterov:
                    num = torch.lerp(g_f, m, beta1)
                elif not cautious:
                    m_look = torch.lerp(g_f, m, beta1)
                    gain = torch.where(m * g_f >= 0.0, gain_hi, gain_lo)
                    num = m_look.mul_(gain)
                else:
                    num = m

                if cautious:
                    mask = (num * g_f >= 0).to(torch.float32)
                    mask_mean = mask.mean().clamp(min=1e-3)
                    num = num * (mask / mask_mean)

                if p.dtype == torch.float32:
                    p.addcdiv_(num, denom, value=-step_size)
                else:
                    update = (num / denom).mul_(step_size)
                    if is_quad_v and stochastic:
                        _copy_fp32_to_16bit_sr_(p, p.float().sub_(update), stochastic=True)
                    else:
                        p.sub_(update.to(p.dtype))

                # Requantize both moments back to quantized storage in-place (EMA-MAX scale smoothing)
                if is_4bit_m:
                    # INT4 requant: pack 2 nibbles per byte into the existing uint8 buffer
                    _quantize_blockwise_int4(
                        m,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=p_state["exp_avg"],
                        out_scales=p_state["exp_avg_scales"],
                        old_scales=p_state["exp_avg_scales"],
                    )
                else:
                    _quantize_blockwise_int8(
                        m,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=p_state["exp_avg"],
                        out_scales=p_state["exp_avg_scales"],
                        old_scales=p_state["exp_avg_scales"],
                    )
                if m_slow is not None:
                    _quantize_blockwise_int8(
                        m_slow,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=p_state["exp_avg_slow"],
                        out_scales=p_state["exp_avg_slow_scales"],
                        old_scales=p_state["exp_avg_slow_scales"],
                    )

                _quantize_blockwise_uint8(
                    v_val,
                    block_size=v_bs,
                    stochastic=stochastic,
                    out_q=p_state["exp_avg_sq"],
                    out_scales=p_state["exp_avg_sq_scales"],
                    old_scales=p_state["exp_avg_sq_scales"],
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
                    mask = (num * g_f >= 0).to(torch.float32)
                    mask_mean = mask.mean().clamp(min=1e-3)
                    num = num * (mask / mask_mean)

                update = (num / denom).mul_(step_size)
                p.sub_(update.to(p.dtype))

                # Requantize updated momentum back to INT8 storage in-place (EMA-MAX scale smoothing)
                _quantize_blockwise_int8(
                    m,
                    block_size=block_size,
                    stochastic=stochastic,
                    out_q=p_state["exp_avg"],
                    out_scales=p_state["exp_avg_scales"],
                    old_scales=p_state["exp_avg_scales"],
                )
                if m_slow is not None:
                    _quantize_blockwise_int8(
                        m_slow,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=p_state["exp_avg_slow"],
                        out_scales=p_state["exp_avg_slow_scales"],
                        old_scales=p_state["exp_avg_slow_scales"],
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
                    mask = (num * g_f >= 0).to(torch.float32)
                    mask_mean = mask.mean().clamp(min=1e-3)
                    num = num * (mask / mask_mean)

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

        grads = _sanitize_gradients_fast(grads)

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
            inv_scale = grad_scale.reciprocal() if hasattr(grad_scale, "reciprocal") else (1.0 / float(grad_scale))

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
        grads = _sanitize_gradients_fast(grads)

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

            effective_eps = eps

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
            stochastic = p_state.get("stochastic_rounding", True)
            if p_state.get("is_factored_v", False) and not lion and not (capturable or compiling):
                _apply_factored_v_int8_update(
                    p,
                    g,
                    p_state,
                    beta1=beta1,
                    beta2=beta2,
                    step_size=step_size,
                    eps_scaled=eps_scaled,
                    effective_eps=effective_eps,
                    nesterov=nesterov,
                    cautious=cautious,
                    stochastic=stochastic,
                    block_size=p_state.get("block_size_8bit", 256),
                    wd_factor=None,
                    sanitize_g=False,
                )
                continue
            if is_8bit or is_ultra or m.dtype == torch.int8:
                block_size = p_state.get("block_size_8bit", p_state.get("ultra_block_size", 256))
                g_f = g.float()
                is_4bit_m = p_state.get("is_4bit_m", False)
                _m_numel = p_state.get("m_numel", m.numel() * 2 if is_4bit_m else m.numel())
                if is_4bit_m:
                    m_deq = _dequantize_blockwise_int4(m, p_state["exp_avg_scales"], numel=_m_numel, block_size=block_size).view(p.shape)
                else:
                    m_deq = _dequantize_blockwise_int8(m, p_state["exp_avg_scales"], block_size=block_size)
                m_deq.lerp_(g_f, 1.0 - beta1)

                m_slow_deq = None
                if m_slow is not None and "exp_avg_slow_scales" in p_state:
                    m_slow_deq = _dequantize_blockwise_int8(m_slow, p_state["exp_avg_slow_scales"], block_size=block_size)
                    m_slow_deq.lerp_(g_f, 1.0 - beta3)

                if p_state.get("is_factored_v", False):
                    denom = _compute_factored_v_denom(
                        g_f,
                        p_state["exp_avg_sq_row"],
                        p_state["exp_avg_sq_col"],
                        beta2=beta2,
                        eps_scaled=eps_scaled,
                        effective_eps=effective_eps,
                        out_shape=p.shape,
                    )
                elif is_8bit or (v is not None and v.dtype == torch.uint8):
                    is_quad_v_s = bool(p_state.get("is_quad_v", False)) or (
                        is_4bit_m and _m_numel >= 4 and _m_numel % 4 == 0 and v.numel() * 4 == _m_numel
                    )
                    v_bs_s = max(1, block_size // 4) if is_quad_v_s else block_size
                    v_deq = _dequantize_blockwise_uint8(v, p_state["exp_avg_sq_scales"], block_size=v_bs_s)
                    if is_quad_v_s:
                        g_sq_q = g_f.reshape(-1, 4).square().mean(dim=1)
                        v_deq.mul_(beta2).add_(g_sq_q, alpha=1.0 - beta2)
                        denom = v_deq.sqrt().add_(eps_scaled).clamp_min_(effective_eps).unsqueeze(1).expand(-1, 4).reshape(p.shape)
                    else:
                        v_deq.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                        if amsgrad:
                            torch.maximum(max_exp_avg_sqs[i], v_deq, out=max_exp_avg_sqs[i])
                            v_target = max_exp_avg_sqs[i]
                        else:
                            v_target = v_deq
                        denom = v_target.sqrt().add_(eps_scaled).clamp_min_(effective_eps)
                    _quantize_blockwise_uint8(
                        v_deq,
                        block_size=v_bs_s,
                        stochastic=stochastic,
                        out_q=v,
                        out_scales=p_state["exp_avg_sq_scales"],
                        old_scales=p_state["exp_avg_sq_scales"],
                    )
                else:
                    is_quad_v_s = False
                    v_f = v.float().mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
                    _copy_fp32_to_16bit_sr_(v, v_f, stochastic=stochastic)
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
                elif is_8bit and not cautious:
                    m_look = torch.lerp(g_f, m_deq, beta1)
                    gain = torch.where(m_deq * g_f >= 0.0, 1.0 + 0.20 * beta1, 1.0 - 0.35 * beta1)
                    num = m_look.mul_(gain)
                else:
                    num = m_deq

                if cautious:
                    mask = (num * g_f) > 0
                    mask_sum = mask.sum()
                    if mask_sum > 0:
                        scale = mask.numel() / mask_sum.clamp_min(1)
                        num = num * mask * scale

                # FP16 NaN guard: never cast denom to float16 before dividing.
                # float16 min subnormal ≈ 5.96e-8; any denom < that rounds to 0.0 → NaN.
                # Compute the entire update in float32 and only downcast at the final store.
                if p.dtype == torch.float32:
                    p.addcdiv_(num, denom, value=-step_size)
                else:
                    update = (num / denom).mul_(step_size)
                    if stochastic:
                        _copy_fp32_to_16bit_sr_(p, p.float().sub_(update), stochastic=True)
                    else:
                        p.sub_(update.to(p.dtype))

                if is_4bit_m:
                    _quantize_blockwise_int4(
                        m_deq,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=m,
                        out_scales=p_state["exp_avg_scales"],
                        old_scales=p_state["exp_avg_scales"],
                    )
                else:
                    _quantize_blockwise_int8(
                        m_deq,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=m,
                        out_scales=p_state["exp_avg_scales"],
                        old_scales=p_state["exp_avg_scales"],
                    )
                if m_slow_deq is not None:
                    _quantize_blockwise_int8(
                        m_slow_deq,
                        block_size=block_size,
                        stochastic=stochastic,
                        out_q=m_slow,
                        out_scales=p_state["exp_avg_slow_scales"],
                        old_scales=p_state["exp_avg_slow_scales"],
                    )
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
            is_quad_v_c = (
                v is not None
                and p.numel() >= 4
                and p.numel() % 4 == 0
                and v.numel() * 4 == p.numel()
            )
            if is_quad_v_c:
                g_f_q = g_f if state_dtype in (torch.bfloat16, torch.float16) else g.float()
                g_sq_q = g_f_q.reshape(-1, 4).square().mean(dim=1)
                v_f = v.float().mul_(beta2).add_(g_sq_q, alpha=(1.0 - beta2) * scale_v)
                if beta2 >= 0.995 and stochastic and v.dtype in (torch.bfloat16, torch.float16):
                    _copy_fp32_to_16bit_sr_(v, v_f, stochastic=True, in_place_src=True)
                else:
                    v.copy_(v_f)
            elif state_dtype in (torch.bfloat16, torch.float16):
                v_f = v.float().mul_(beta2).addcmul_(g_f, g_f, value=(1.0 - beta2) * scale_v)
                if beta2 >= 0.995 and stochastic:
                    _copy_fp32_to_16bit_sr_(v, v_f, stochastic=True, in_place_src=True)
                else:
                    v.copy_(v_f)
            else:
                v.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2) * scale_v)

            # Fast path for standard AdamW (zero extra allocations)
            if not amsgrad and not cautious and not nesterov and m_slow is None:
                if is_quad_v_c:
                    if capturable or compiling:
                        if scale_v != 1.0:
                            denom_q = (v_f.sqrt() / sqrt_scale_v / (bc2_sqrt * step_size_neg)).add_(effective_eps / step_size_neg)
                        else:
                            denom_q = (v_f.sqrt() / (bc2_sqrt * step_size_neg)).add_(effective_eps / step_size_neg)
                    else:
                        if scale_v != 1.0:
                            denom_q = v_f.sqrt().add_(eps_scaled * sqrt_scale_v).div_(sqrt_scale_v)
                        else:
                            denom_q = v_f.sqrt().add_(eps_scaled)
                    m_f_q = m_f if state_dtype in (torch.bfloat16, torch.float16) else m.float()
                    num_q = m_f_q
                    if capturable or compiling:
                        num_q.view(-1, 4).div_(denom_q.unsqueeze(1))
                        p.add_(num_q.to(p.dtype) if p.dtype != torch.float32 else num_q)
                    else:
                        step_delta = num_q.view(-1, 4).div_(denom_q.unsqueeze(1)).mul_(step_size).view(p.shape)
                        if p.dtype != torch.float32:
                            _copy_fp32_to_16bit_sr_(p, p.float().sub_(step_delta), stochastic=stochastic, in_place_src=True)
                        else:
                            p.sub_(step_delta)
                    continue
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
                    # FP16 NaN guard: compute update in float32, cast only the final result.
                    # p.addcdiv_(m, denom) would cast denom to p.dtype first, flushing tiny
                    # denominators to 0.0 in float16 → NaN on step 1 (eps_scaled ≈ 3e-10).
                    if p.dtype == torch.float32:
                        p.addcdiv_(m, denom, value=-step_size)
                    else:
                        m_f = m if m.dtype == torch.float32 else m.float()
                        update = (m_f / denom).mul_(step_size)
                        if stochastic:
                            _copy_fp32_to_16bit_sr_(p, p.float().sub_(update), stochastic=True, in_place_src=True)
                        else:
                            p.sub_(update.to(p.dtype))
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
            if is_quad_v_c:
                denom = denom.float().unsqueeze(1).expand(-1, 4).reshape(p.shape)

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

            # FP16 NaN guard: compute update in float32, only cast the final delta to p.dtype.
            # Both capturable and non-capturable paths must stay in float32 for the division.
            if capturable or compiling:
                p.addcdiv_(num, denom)
            elif p.dtype == torch.float32:
                p.addcdiv_(num, denom, value=-step_size)
            else:
                update = (num / denom).mul_(step_size)
                p.sub_(update.to(p.dtype))

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
        stochastic: bool = True,
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

        # Smart Grouping Dispatcher
        # Group tensors strictly by (device, param_dtype, state_dtype)
        # to separate homogenous subgroups from mixed-precision subgroups
        subgroups = defaultdict(lambda: {
            "params": [], "grads": [], "exp_avgs": [], "steps": [],
            "exp_avgs_slow": [], "exp_avg_sqs": [], "max_exp_avg_sqs": []
        })
        factored_indices: List[int] = []

        for i, p in enumerate(params):
            if self.state.get(p, {}).get("is_factored_v", False) and not lion and not (capturable or compiling):
                factored_indices.append(i)
                continue
            m = exp_avgs[i]
            # When not capturable and not compiling, partition also by current step count
            # so each subgroup is guaranteed uniform step counts. This ensures bias-correction
            # scalars are mathematically exact for every parameter, even with intermittent gradients.
            s_val = int(_get_value(state_steps[i])) if (not capturable and not compiling) else None
            is_quad_v = (
                not lion
                and p.numel() >= 4
                and p.numel() % 4 == 0
                and exp_avg_sqs[i].numel() * 4 == p.numel()
            )
            key = (p.device, p.dtype, m.dtype, s_val, is_quad_v)
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

        if factored_indices:
            skip_factored = False
            if found_inf is not None:
                is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
                if not is_capturing and not compiling and found_inf.item() > 0:
                    skip_factored = True
            if not skip_factored:
                f_inv_scales: Dict[torch.device, Tensor] = {}
                wd_factor_base = max(1.0 - lr * weight_decay, 0.0) if weight_decay != 0.0 else 1.0
                for idx in factored_indices:
                    p = params[idx]
                    g = grads[idx]
                    dev_p = p.device
                    step_t = state_steps[idx]
                    step_t.add_(1)
                    step = _get_value(step_t)
                    bc1 = 1.0 - (beta1 ** step)
                    bc2 = 1.0 - (beta2 ** step)
                    bc2_sqrt = math.sqrt(bc2)
                    step_size = lr * bc2_sqrt / bc1
                    effective_eps = eps
                    eps_scaled = effective_eps * bc2_sqrt
                    inv_sc = None
                    if grad_scale is not None:
                        if dev_p not in f_inv_scales:
                            if hasattr(grad_scale, "device"):
                                gs_dev = grad_scale if grad_scale.device == dev_p else grad_scale.to(dev_p)
                                f_inv_scales[dev_p] = gs_dev.reciprocal()
                            else:
                                f_inv_scales[dev_p] = 1.0 / float(grad_scale)
                        inv_sc = f_inv_scales[dev_p]
                    p_st = self.state[p]
                    apply_wd = weight_decay != 0.0 and (not decay_norm_only or p.dim() >= 2)
                    _apply_factored_v_int8_update(
                        p,
                        g,
                        p_st,
                        beta1=beta1,
                        beta2=beta2,
                        step_size=step_size,
                        eps_scaled=eps_scaled,
                        effective_eps=effective_eps,
                        nesterov=nesterov,
                        cautious=cautious,
                        stochastic=p_st.get("stochastic_rounding", True),
                        block_size=p_st.get("block_size_8bit", 256),
                        wd_factor=wd_factor_base if apply_wd else None,
                        inv_scale=inv_sc,
                        maximize=maximize,
                        sanitize_g=True,
                    )

        for (dev, param_dt, state_dt, _, is_quad_v), sg in subgroups.items():
            d_params = sg["params"]
            d_grads = sg["grads"]
            d_exp_avgs = sg["exp_avgs"]
            d_steps = sg["steps"]
            d_exp_avgs_slow = sg["exp_avgs_slow"]
            d_exp_avg_sqs = sg["exp_avg_sqs"]
            d_max_exp_avg_sqs = sg["max_exp_avg_sqs"]

            # Dispatch to custom multi-tensor batched CUDA kernel (1 launch for all tensors)
            # Guard: all params in this subgroup must have the same step counter.
            # When gradients are intermittent, step counts diverge and the kernel's
            # single-step bias correction would be wrong for the lagging parameters.
            v_dt = d_exp_avg_sqs[0].dtype if d_exp_avg_sqs else state_dt
            can_cuda_fused = (
                dev.type == "cuda"
                and (param_dt, state_dt, v_dt) in _CUDA_DTYPE_TRIOS
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
                and len(d_exp_avg_sqs) == len(d_params)
                and all(
                    p.is_contiguous()
                    and g.is_contiguous()
                    and m.is_contiguous()
                    and v.is_contiguous()
                    and p.numel() == g.numel() == m.numel()
                    and (v.numel() == p.numel() or (is_quad_v and v.numel() * 4 == p.numel()))
                    and g.dtype == p.dtype
                    and (p.dtype, m.dtype, v.dtype) in _CUDA_DTYPE_TRIOS
                    for p, g, m, v in zip(d_params, d_grads, d_exp_avgs, d_exp_avg_sqs)
                )
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

            d_grads = _sanitize_gradients_fast(d_grads)

            # Handle AMP grad_scale: in-place unscaling (zero auxiliary VRAM allocations)
            if grad_scale is not None:
                if hasattr(grad_scale, "device"):
                    device_grad_scale = grad_scale if (dev is None or grad_scale.device == dev) else grad_scale.to(dev)
                    inv_scale = device_grad_scale.reciprocal()
                else:
                    inv_scale = 1.0 / float(grad_scale)
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
                # slow_ratio must include bias-correction ratio bc1/bc3
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

            effective_eps = eps
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

                if is_quad_v:
                    c_grads_f = c_grads if c_grads[0].dtype == torch.float32 else [g.float() for g in c_grads]
                    torch._foreach_mul_(c_exp_avgs, beta1)
                    torch._foreach_add_(c_exp_avgs, c_grads_f, alpha=1.0 - beta1)
                    if beta3 is not None:
                        torch._foreach_mul_(c_exp_avgs_slow, beta3)
                        torch._foreach_add_(c_exp_avgs_slow, c_grads_f, alpha=1.0 - beta3)
                    g_sq_qs = [gf.reshape(-1, 4).square().mean(dim=1) for gf in c_grads_f]
                    if beta2 >= 0.995 and stochastic and c_exp_avg_sqs and c_exp_avg_sqs[0].dtype in (torch.bfloat16, torch.float16):
                        for v_i, gq in zip(c_exp_avg_sqs, g_sq_qs):
                            v_next = v_i.float().mul_(beta2).add_(gq, alpha=(1.0 - beta2) * scale_v)
                            _copy_fp32_to_16bit_sr_(v_i, v_next, stochastic=True, in_place_src=True)
                    else:
                        torch._foreach_mul_(c_exp_avg_sqs, beta2)
                        torch._foreach_add_(c_exp_avg_sqs, g_sq_qs, alpha=(1.0 - beta2) * scale_v)
                    del g_sq_qs
                    if amsgrad:
                        torch._foreach_maximum_(c_max_exp_avg_sqs, c_exp_avg_sqs)
                        v_sources = c_max_exp_avg_sqs
                    else:
                        v_sources = c_exp_avg_sqs
                    denoms_q = torch._foreach_sqrt(v_sources)
                    if use_tensor_steps:
                        if scale_v != 1.0:
                            torch._foreach_div_(denoms_q, c_bc2_sqrt_list)
                            torch._foreach_add_(denoms_q, effective_eps * sqrt_scale_v)
                            torch._foreach_div_(denoms_q, c_step_size_list)
                            torch._foreach_div_(denoms_q, sqrt_scale_v)
                        else:
                            torch._foreach_div_(denoms_q, c_bc2_sqrt_list)
                            torch._foreach_add_(denoms_q, effective_eps)
                            torch._foreach_div_(denoms_q, c_step_size_list)
                    else:
                        if scale_v != 1.0:
                            torch._foreach_add_(denoms_q, eps_scaled * sqrt_scale_v)
                            torch._foreach_div_(denoms_q, sqrt_scale_v)
                        else:
                            torch._foreach_add_(denoms_q, eps_scaled)
                    for idx_q, (p_i, g_f, m_i, denom_q) in enumerate(zip(c_params, c_grads_f, c_exp_avgs, denoms_q)):
                        m_f = m_i.float() if m_i.dtype != torch.float32 else m_i
                        if beta3 is not None:
                            ms_i = c_exp_avgs_slow[idx_q]
                            ms_f = ms_i.float() if ms_i.dtype != torch.float32 else ms_i
                            num = m_f.add(ms_f, alpha=slow_ratio)
                        elif cautious:
                            mask = (g_f * m_f >= 0).to(torch.float32)
                            mask_mean = mask.mean().clamp(min=1e-3)
                            num = m_f.mul(mask.div_(mask_mean))
                        elif nesterov:
                            num = torch.lerp(g_f, m_f, beta1)
                        else:
                            num = m_f
                        dq_f = denom_q.float() if denom_q.dtype != torch.float32 else denom_q
                        if use_tensor_steps:
                            step_delta = num.view(-1, 4).div_(dq_f.unsqueeze(1)).view(p_i.shape)
                            p_i.sub_(step_delta.to(p_i.dtype) if p_i.dtype != torch.float32 else step_delta)
                        else:
                            step_delta = num.view(-1, 4).div_(dq_f.unsqueeze(1)).mul_(step_size).view(p_i.shape)
                            if p_i.dtype != torch.float32:
                                _copy_fp32_to_16bit_sr_(p_i, p_i.float().sub_(step_delta), stochastic=stochastic, in_place_src=True)
                            else:
                                p_i.sub_(step_delta)
                    del denoms_q
                    if c_grads_comp is not None:
                        del c_grads_comp
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

                    # Update second-order momentum:
                    # When beta2 >= 0.995, BF16 round-to-nearest loses (1 - beta2)*g^2,
                    # causing v_t to stagnate. Stochastic rounding prevents stagnation.
                    if beta2 >= 0.995 and stochastic:
                        for v_i, g_i in zip(c_exp_avg_sqs, c_grads_f):
                            v_next = v_i.float().mul_(beta2).addcmul_(g_i, g_i, value=(1.0 - beta2) * scale_v)
                            _copy_fp32_to_16bit_sr_(v_i, v_next, stochastic=True, in_place_src=True)
                    else:
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
                        if c_params[0].dtype in (torch.bfloat16, torch.float16) and stochastic:
                            for p_i, m_i, denom_i in zip(c_params, c_exp_avgs, denoms):
                                step_delta = (m_i.float() / denom_i.float()) * step_size
                                _copy_fp32_to_16bit_sr_(p_i, p_i.float().sub_(step_delta), stochastic=True)
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
