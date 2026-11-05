"""
ApexW OpenAI Triton Fused Kernel Backend.

Compiles and executes high-performance fused AdamW kernels via OpenAI Triton.
Automatically compiled down to optimized GPU machine code for both NVIDIA GPUs
(via PTX/SASS) and AMD GPUs (via ROCm / HIP / GCN / CDNA ISA).
"""

import os
import math
from typing import List, Optional, Dict, Any, Union
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    triton = None
    tl = None


def is_rocm() -> bool:
    """Returns True if the PyTorch environment is running on AMD ROCm (HIP)."""
    return bool(hasattr(torch.version, "hip") and torch.version.hip is not None)


def is_triton_available() -> bool:
    """
    Returns True if OpenAI Triton is installed and a supported accelerator
    (NVIDIA CUDA or AMD ROCm) is available.
    """
    if not _HAS_TRITON:
        return False
    if os.environ.get("APEXW_DISABLE_TRITON", "0") == "1":
        return False
    # Check GPU availability (covers NVIDIA CUDA and AMD ROCm)
    if torch.cuda.is_available():
        return True
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return True
    return False


if _HAS_TRITON:
    @triton.jit
    def _adamw_kernel_triton_fp32(
        p_ptr,
        g_ptr,
        m_ptr,
        v_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps_scaled,
        weight_decay,
        step_size,
        inv_scale,
        apply_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Pure float32 (FP32) fast-path kernel.
        Bypasses quantization/dequantization scaling entirely to match or beat
        PyTorch's native AdamW(fused=True) throughput.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
        if inv_scale != 1.0:
            g = g * inv_scale
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

        # Decoupled weight decay
        if apply_wd:
            p = p * (1.0 - lr * weight_decay)

        # Update first-order momentum (EMA)
        m = m * beta1 + g * (1.0 - beta1)

        # Update second-order momentum without dynamic scaling overhead
        v = v * beta2 + (g * g) * (1.0 - beta2)

        # Denominator with scaled epsilon
        denom = tl.sqrt(v) + eps_scaled
        p = p - (m / denom) * step_size

        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(m_ptr + offsets, m, mask=mask)
        tl.store(v_ptr + offsets, v, mask=mask)

    @triton.jit
    def _adamw_kernel_triton_cross_precision(
        p_ptr,
        g_ptr,
        m_ptr,
        v_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step_size,
        eps_scaled,
        scale_v,
        sqrt_scale_v,
        inv_scale,
        apply_wd: tl.constexpr,
        has_scale_v: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        m_slow_ptr=None,
        beta3: float = 0.0,
        slow_ratio: float = 0.0,
        has_slow_momentum: tl.constexpr = False,
        cautious: tl.constexpr = False,
    ):
        """
        Cross-Precision Updates (precision='compact' with FP32 weights).
        Merges 16-bit to FP32 load casting, dynamic scaling, optimizer step calculation,
        and FP32 to 16-bit store casting into a single memory pass, eliminating temporary
        tensor allocations.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load parameter (FP32) and unscale gradient in registers
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if inv_scale != 1.0:
            g = g * inv_scale

        # Load 16-bit compact optimizer states and promote to FP32 in registers
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Decoupled weight decay
        if apply_wd:
            p = p * (1.0 - lr * weight_decay)

        # Update first-order momentum (EMA) in FP32 registers
        m = m * beta1 + g * (1.0 - beta1)

        # AdEMAMix dual-speed momentum
        if has_slow_momentum:
            m_slow = tl.load(m_slow_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            m_slow = m_slow * beta3 + g * (1.0 - beta3)
            tl.store(m_slow_ptr + offsets, m_slow.to(m_slow_ptr.dtype.element_ty), mask=mask)
            m_eff = m + slow_ratio * m_slow
        else:
            m_eff = m

        # Update second-order momentum (EMA) with merged dynamic scaling
        if has_scale_v:
            v = v * beta2 + (g * g) * ((1.0 - beta2) * scale_v)
            denom = tl.sqrt(v) + (eps_scaled * sqrt_scale_v)
            step_factor = step_size * sqrt_scale_v
        else:
            v = v * beta2 + (g * g) * (1.0 - beta2)
            denom = tl.sqrt(v) + eps_scaled
            step_factor = step_size

        update = (m_eff / denom) * step_factor

        # Cautious directional update masking
        if cautious:
            update = tl.where(g * update >= 0.0, update, 0.0)

        p = p - update

        # Store back in a single pass: FP32 param, casted 16-bit states
        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(m_ptr + offsets, m.to(m_ptr.dtype.element_ty), mask=mask)
        tl.store(v_ptr + offsets, v.to(v_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _adamw_kernel_triton(
        p_ptr,
        g_ptr,
        m_ptr,
        v_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step_size,
        eps_scaled,
        scale_v,
        sqrt_scale_v,
        inv_scale,
        apply_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        m_slow_ptr=None,
        beta3: float = 0.0,
        slow_ratio: float = 0.0,  # = alpha * (bc1/bc3): pre-bias-corrected AdEMAMix coefficient
        has_slow_momentum: tl.constexpr = False,
        cautious: tl.constexpr = False,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load parameter and optimizer states in float32 for maximum precision
        # Fuses AMP gradient unscaling directly in registers (zero auxiliary VRAM allocation)
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * inv_scale
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Decoupled weight decay
        if apply_wd:
            p = p * (1.0 - lr * weight_decay)

        # Update first-order momentum (EMA)
        m = m * beta1 + g * (1.0 - beta1)

        # Update second-order momentum with dynamic scale factor (EMA)
        v = v * beta2 + (g * g) * ((1.0 - beta2) * scale_v)

        # AdEMAMix dual-speed momentum (Pagliardini et al., 2024)
        # slow_ratio = alpha * (bc1/bc3) already incorporates bias corrections,
        # matching the foreach and single-tensor paths exactly.
        if has_slow_momentum:
            m_slow = tl.load(m_slow_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            m_slow = m_slow * beta3 + g * (1.0 - beta3)
            tl.store(m_slow_ptr + offsets, m_slow, mask=mask)
            m_eff = m + slow_ratio * m_slow
        else:
            m_eff = m

        # Denominator with scaled epsilon
        denom = tl.sqrt(v) + (eps_scaled * sqrt_scale_v)
        step_factor = step_size * sqrt_scale_v
        update = (m_eff / denom) * step_factor

        # Cautious directional update masking (Liang et al., 2024)
        if cautious:
            update = tl.where(g * update >= 0.0, update, 0.0)

        p = p - update

        # Store back into tensor buffers (Triton automatically casts back to target buffer dtypes)
        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(m_ptr + offsets, m, mask=mask)
        tl.store(v_ptr + offsets, v, mask=mask)
else:
    _adamw_kernel_triton = None
    _adamw_kernel_triton_fp32 = None
    _adamw_kernel_triton_cross_precision = None


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 0:
        return 1
    if _HAS_TRITON and hasattr(triton, "next_power_of_2"):
        try:
            return triton.next_power_of_2(n)
        except Exception:
            pass
    return 1 << (n - 1).bit_length()


def _triton_preflight_check(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    exp_avg_slow: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """
    Validates tensor layout, device, and dtype before any Triton kernel is launched.
    Returns None if all checks pass, or a human-readable reason string if not supported.
    Pre-flight validation ensures unsupported configurations safely return False before
    any tensor data is modified in-place.
    """
    _SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}

    if not param.is_cuda:
        return f"param must be a CUDA tensor (got device={param.device})"
    if not grad.is_cuda:
        return f"grad must be a CUDA tensor (got device={grad.device})"
    if param.device != grad.device:
        return f"param and grad must be on the same device ({param.device} vs {grad.device})"
    if param.device != exp_avg.device or param.device != exp_avg_sq.device:
        return "param, exp_avg, exp_avg_sq must all be on the same device"
    if exp_avg_slow is not None and exp_avg_slow.device != param.device:
        return "exp_avg_slow must be on the same device as param"
    if param.numel() != grad.numel():
        return f"param and grad have different element counts ({param.numel()} vs {grad.numel()})"
    if param.dtype not in _SUPPORTED_DTYPES:
        return f"param dtype {param.dtype} not supported by Triton kernel"
    if grad.dtype not in _SUPPORTED_DTYPES:
        return f"grad dtype {grad.dtype} not supported by Triton kernel"
    if exp_avg.dtype not in _SUPPORTED_DTYPES:
        return f"exp_avg dtype {exp_avg.dtype} not supported by Triton kernel"
    if exp_avg_sq.dtype not in _SUPPORTED_DTYPES:
        return f"exp_avg_sq dtype {exp_avg_sq.dtype} not supported by Triton kernel"
    if exp_avg_slow is not None and exp_avg_slow.dtype not in _SUPPORTED_DTYPES:
        return f"exp_avg_slow dtype {exp_avg_slow.dtype} not supported by Triton kernel"
    return None  # all checks passed


def triton_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step_size: float,
    eps_scaled: float,
    scale_v: float = 1.0,
    sqrt_scale_v: float = 1.0,
    inv_scale: float = 1.0,
    apply_wd: bool = True,
    block_size: Optional[int] = None,
    skip_preflight: bool = False,
    exp_avg_slow: Optional[torch.Tensor] = None,
    beta3: Optional[float] = None,
    slow_ratio: float = 0.0,  # = alpha * (bc1/bc3): pre-bias-corrected AdEMAMix coefficient
    cautious: bool = False,
) -> bool:
    """
    Executes the Triton fused AdamW kernel on a single tensor.

    All pre-flight checks (device, dtype, layout, shape) are performed before kernel
    launch. If the kernel fails after launch, the exception propagates to prevent
    partial double-updates across fallback paths.

    ``slow_ratio`` must be the pre-bias-corrected AdEMAMix blending coefficient
    ``alpha * (1 - beta1^t) / (1 - beta3^t)`` computed by the caller. Passing raw
    ``alpha`` here would produce incorrect results during the warmup phase.
    """
    if not is_triton_available():
        return False

    n_elements = param.numel()
    if n_elements == 0:
        return True

    # Pre-flight validation before any mutation
    if not skip_preflight:
        reason = _triton_preflight_check(param, grad, exp_avg, exp_avg_sq, exp_avg_slow)
        if reason is not None:
            if os.environ.get("APEXW_DEBUG", "0") == "1":
                print(f"[ApexW] Triton pre-flight rejected: {reason}")
            return False  # Safe: no kernel has run yet.

    if block_size is None:
        block_size = min(max(64, _next_power_of_2(n_elements)), 1024)

    # Ensure tensors are contiguous
    if not param.is_contiguous():
        param.data = param.contiguous()
    if not grad.is_contiguous():
        grad.data = grad.contiguous()
    if not exp_avg.is_contiguous():
        exp_avg.data = exp_avg.contiguous()
    if not exp_avg_sq.is_contiguous():
        exp_avg_sq.data = exp_avg_sq.contiguous()
    if exp_avg_slow is not None and not exp_avg_slow.is_contiguous():
        exp_avg_slow.data = exp_avg_slow.contiguous()

    has_slow = exp_avg_slow is not None and beta3 is not None
    num_blocks = triton.cdiv(n_elements, block_size) if (triton is not None and hasattr(triton, "cdiv")) else (n_elements + block_size - 1) // block_size

    # 1. Pure FP32 Fast Path:
    # If parameters and states are both float32, bypass quantization/dequantization scaling entirely
    # to match or beat PyTorch's native AdamW(fused=True) throughput.
    is_pure_fp32 = (
        param.dtype == torch.float32
        and grad.dtype == torch.float32
        and exp_avg.dtype == torch.float32
        and exp_avg_sq.dtype == torch.float32
        and scale_v == 1.0
        and sqrt_scale_v == 1.0
        and not has_slow
        and not cautious
    )

    if is_pure_fp32 and _adamw_kernel_triton_fp32 is not None:
        _adamw_kernel_triton_fp32[(num_blocks,)](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            n_elements,
            float(lr),
            float(beta1),
            float(beta2),
            float(eps_scaled),
            float(weight_decay),
            float(step_size),
            float(inv_scale),
            apply_wd=bool(apply_wd),
            BLOCK_SIZE=block_size,
        )
        return True

    # 2. Cross-Precision Fast Path (precision='compact' with FP32 weights):
    # Merges 16-bit to FP32 load casting, dynamic scaling, optimizer step calculation,
    # and FP32 to 16-bit store casting into a single memory pass, eliminating temporary tensor allocations.
    is_cross_precision = (
        param.dtype == torch.float32
        and (exp_avg.dtype in (torch.bfloat16, torch.float16) or exp_avg_sq.dtype in (torch.bfloat16, torch.float16))
    )

    if is_cross_precision and _adamw_kernel_triton_cross_precision is not None:
        has_scale_v = bool(scale_v != 1.0 or sqrt_scale_v != 1.0)
        _adamw_kernel_triton_cross_precision[(num_blocks,)](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            n_elements,
            float(lr),
            float(beta1),
            float(beta2),
            float(eps),
            float(weight_decay),
            float(step_size),
            float(eps_scaled),
            float(scale_v),
            float(sqrt_scale_v),
            float(inv_scale),
            apply_wd=bool(apply_wd),
            has_scale_v=has_scale_v,
            BLOCK_SIZE=block_size,
            m_slow_ptr=exp_avg_slow if has_slow else param,
            beta3=float(beta3) if beta3 is not None else 0.0,
            slow_ratio=float(slow_ratio),
            has_slow_momentum=bool(has_slow),
            cautious=bool(cautious),
        )
        return True

    # 3. Standard Unified Triton Kernel (all other cases)
    if _adamw_kernel_triton is not None:
        _adamw_kernel_triton[(num_blocks,)](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            n_elements,
            float(lr),
            float(beta1),
            float(beta2),
            float(eps),
            float(weight_decay),
            float(step_size),
            float(eps_scaled),
            float(scale_v),
            float(sqrt_scale_v),
            float(inv_scale),
            apply_wd=bool(apply_wd),
            BLOCK_SIZE=block_size,
            m_slow_ptr=exp_avg_slow if has_slow else param,
            beta3=float(beta3) if beta3 is not None else 0.0,
            slow_ratio=float(slow_ratio),
            has_slow_momentum=bool(has_slow),
            cautious=bool(cautious),
        )
        return True
    return False


def triton_multi_tensor_adamw_step(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    decay_norm_only: bool = True,
    scales_v: Optional[Union[Dict[int, float], List[float]]] = None,
    inv_scale: float = 1.0,
    block_size: Optional[int] = None,
    exp_avgs_slow: Optional[List[torch.Tensor]] = None,
    beta3: Optional[float] = None,
    alpha: float = 5.0,
    cautious: bool = False,
    skip_preflight: bool = False,
) -> bool:
    """
    Executes the Triton fused AdamW kernel across multiple tensors.
    Works natively on both NVIDIA CUDA and AMD ROCm GPUs with zero C++ compilation.
    Fuses gradient unscaling, AdEMAMix dual-speed momentum, and cautious directional masking.
    """
    if not is_triton_available():
        return False
    if len(params) == 0:
        return True

    # Validate the entire batch before launching any kernel (only if skip_preflight is False)
    has_slow_list = exp_avgs_slow is not None and len(exp_avgs_slow) == len(params)
    if not skip_preflight:
        for i, p in enumerate(params):
            slow_i = exp_avgs_slow[i] if has_slow_list else None
            reason = _triton_preflight_check(p, grads[i], exp_avgs[i], exp_avg_sqs[i], slow_i)
            if reason is not None:
                if os.environ.get("APEXW_DEBUG", "0") == "1":
                    print(f"[ApexW] Triton pre-flight rejected param[{i}]: {reason}")
                return False  # Safe: no kernel has run yet.

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    bc2_sqrt = math.sqrt(bc2)
    step_size = lr * bc2_sqrt / bc1
    eps_scaled = eps * bc2_sqrt

    # Pre-compute bias-corrected AdEMAMix blending coefficient (matching foreach/single paths).
    # slow_ratio = alpha * (1 - beta1^t) / (1 - beta3^t) correctly weights m_slow
    # relative to fast momentum during warmup. Using raw alpha produces over-weighted
    # slow momentum in early steps (step=1: bc1≈0.1, bc3≈0.0001 → 1000× error).
    slow_ratio = 0.0
    if beta3 is not None and has_slow_list:
        bc3 = 1.0 - (beta3 ** step)
        slow_ratio = alpha * (bc1 / bc3) if bc3 > 0.0 else 0.0

    is_scales_v_list = isinstance(scales_v, list)

    for i, p in enumerate(params):
        g = grads[i]
        m = exp_avgs[i]
        v = exp_avg_sqs[i]
        m_slow = exp_avgs_slow[i] if has_slow_list else None

        apply_wd = (weight_decay != 0.0)
        if apply_wd and decay_norm_only and p.dim() < 2:
            apply_wd = False

        scale_v = 1.0
        sqrt_scale_v = 1.0
        if scales_v is not None:
            if is_scales_v_list:
                scale_v = scales_v[i]
            elif id(p) in scales_v:
                scale_v = scales_v[id(p)]
            if scale_v != 1.0:
                sqrt_scale_v = math.sqrt(scale_v)

        # triton_adamw_step no longer catches exceptions after the kernel launches,
        # so any failure here correctly aborts the step.
        triton_adamw_step(
            param=p,
            grad=g,
            exp_avg=m,
            exp_avg_sq=v,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            step_size=step_size,
            eps_scaled=eps_scaled,
            scale_v=scale_v,
            sqrt_scale_v=sqrt_scale_v,
            inv_scale=inv_scale,
            apply_wd=apply_wd,
            block_size=block_size,
            skip_preflight=True,
            exp_avg_slow=m_slow,
            beta3=beta3,
            slow_ratio=slow_ratio,
            cautious=cautious,
        )

    return True


def triton_adamw_fp32_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step_size: float,
    eps_scaled: float,
    inv_scale: float = 1.0,
    apply_wd: bool = True,
    block_size: Optional[int] = None,
    skip_preflight: bool = False,
) -> bool:
    """
    Explicit fast-path bypass for pure float32 (FP32) parameters and states.
    Bypasses dynamic scaling, unpacking, and packing states entirely to match
    native AdamW(fused=True) throughput.
    """
    return triton_adamw_step(
        param=param,
        grad=grad,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        step_size=step_size,
        eps_scaled=eps_scaled,
        scale_v=1.0,
        sqrt_scale_v=1.0,
        inv_scale=inv_scale,
        apply_wd=apply_wd,
        block_size=block_size,
        skip_preflight=skip_preflight,
    )


def triton_adamw_cross_precision_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step_size: float,
    eps_scaled: float,
    scale_v: float = 1.0,
    sqrt_scale_v: float = 1.0,
    inv_scale: float = 1.0,
    apply_wd: bool = True,
    block_size: Optional[int] = None,
    skip_preflight: bool = False,
    exp_avg_slow: Optional[torch.Tensor] = None,
    beta3: Optional[float] = None,
    slow_ratio: float = 0.0,
    cautious: bool = False,
) -> bool:
    """
    Explicit fused single-pass cross-precision step for FP32 weights and 16-bit states
    (precision='compact'). Merges casting, scaling, and optimizer update into a single
    memory pass without temporary allocations.
    """
    return triton_adamw_step(
        param=param,
        grad=grad,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        step_size=step_size,
        eps_scaled=eps_scaled,
        scale_v=scale_v,
        sqrt_scale_v=sqrt_scale_v,
        inv_scale=inv_scale,
        apply_wd=apply_wd,
        block_size=block_size,
        skip_preflight=skip_preflight,
        exp_avg_slow=exp_avg_slow,
        beta3=beta3,
        slow_ratio=slow_ratio,
        cautious=cautious,
    )
