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

    @triton.jit
    def _adamw_kernel_triton_8bit(
        p_ptr,
        g_ptr,
        m_ptr,
        m_scales_ptr,
        v_ptr,
        v_scales_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step_size,
        eps_scaled,
        inv_scale,
        apply_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        cautious: tl.constexpr = False,
        step: int = 1,
    ):
        """
        Fused 8-bit AdamW kernel with dynamic block-wise scaling (2 bytes per param).
        m is stored in INT8 (with FP32 scale per block) and v in UINT8 (with FP32 scale per block).
        Dequantization, EMA updates, weight decay, bias-corrected step,
        and intra-block dynamic re-quantization evaluate entirely in FP32 SRAM registers.
        Includes fast integer-hash stochastic rounding on m to eliminate momentum drift.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load scales for this block
        scale_m = tl.load(m_scales_ptr + pid)
        scale_v = tl.load(v_scales_ptr + pid)

        # Load parameters and gradients in float32
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if inv_scale != 1.0:
            g = g * inv_scale

        # Load 8-bit momentum and variance, dequantize to float32
        q_m = tl.load(m_ptr + offsets, mask=mask, other=0).to(tl.float32)
        q_v = tl.load(v_ptr + offsets, mask=mask, other=0).to(tl.float32)
        m = q_m * scale_m
        # Hardware-native 32-octave logarithmic dequantization for variance
        r_v = tl.where(q_v > 0.0, tl.exp2((q_v - 1.0) * (32.0 / 254.0) - 32.0), 0.0)
        v = r_v * scale_v

        # Decoupled weight decay
        if apply_wd:
            p = p * (1.0 - lr * weight_decay)

        # Update momentum in float32 registers
        m = m * beta1 + g * (1.0 - beta1)

        # Update variance in float32 registers
        v = v * beta2 + (g * g) * (1.0 - beta2)

        # Compute denominator and parameter update
        denom = tl.sqrt(v) + eps_scaled
        update = (m / denom) * step_size

        # Cautious directional update masking (Liang et al., 2024)
        if cautious:
            update = tl.where(g * update >= 0.0, update, 0.0)

        p = p - update

        # Store updated parameter back to global memory
        tl.store(p_ptr + offsets, p.to(p_ptr.dtype.element_ty), mask=mask)

        # Dynamic INT8 requantization for m (block-wise signed with unbiased stochastic rounding)
        m_abs = tl.where(mask, tl.abs(m), 0.0)
        max_m = tl.max(m_abs, axis=0)
        new_scale_m = tl.maximum(max_m / 127.0, 1e-12)
        scaled_m = m / new_scale_m

        # Fast 32-bit integer hash for uniform pseudo-random noise in [-0.5, 0.5)
        seed = (offsets.to(tl.int32) ^ (pid * 1973)) + (step * 9277)
        seed = seed ^ (seed >> 16)
        seed = seed * 0x45d9f3b
        seed = seed ^ (seed >> 16)
        noise = ((seed & 0x7fffffff).to(tl.float32) * (1.0 / 2147483648.0)) - 0.5

        rounded_m = tl.floor(scaled_m + noise + 0.5)
        new_q_m = tl.clamp(rounded_m, -128.0, 127.0).to(tl.int8)

        tl.store(m_ptr + offsets, new_q_m, mask=mask)
        tl.store(m_scales_ptr + pid, new_scale_m)

        # Dynamic 32-octave Log-scale UINT8 requantization for v (over 9 orders of magnitude)
        v_pos = tl.where(mask, v, 0.0)
        max_v = tl.max(v_pos, axis=0)
        new_scale_v = tl.maximum(max_v, 1e-12)
        r_new = tl.maximum(v / new_scale_v, 2.3283064e-10)  # 2^-32 minimum ratio
        log_r = tl.log2(r_new)
        q_v_calc = tl.where(
            v > 0.0,
            tl.clamp(tl.floor(1.0 + (log_r + 32.0) * (254.0 / 32.0) + 0.5), 1.0, 255.0),
            0.0,
        )
        new_q_v = q_v_calc.to(tl.uint8)

        tl.store(v_ptr + offsets, new_q_v, mask=mask)
        tl.store(v_scales_ptr + pid, new_scale_v)

    @triton.jit
    def _adamw_kernel_triton_8bit_m_16bit_v(
        p_ptr,
        g_ptr,
        m_ptr,
        m_scales_ptr,
        v_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step_size,
        eps_scaled,
        inv_scale,
        apply_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        cautious: tl.constexpr = False,
    ):
        """
        Fused 3-byte hybrid AdamW kernel (precision='ultra').
        m is stored in INT8 (1 byte with scale) and v in BF16/FP16 (2 bytes).
        Total 3 bytes per parameter with zero variance quantization error.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        scale_m = tl.load(m_scales_ptr + pid)

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if inv_scale != 1.0:
            g = g * inv_scale

        q_m = tl.load(m_ptr + offsets, mask=mask, other=0).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = q_m * scale_m

        if apply_wd:
            p = p * (1.0 - lr * weight_decay)

        m = m * beta1 + g * (1.0 - beta1)
        v = v * beta2 + (g * g) * (1.0 - beta2)

        denom = tl.sqrt(v) + eps_scaled
        update = (m / denom) * step_size

        if cautious:
            update = tl.where(g * update >= 0.0, update, 0.0)

        p = p - update

        tl.store(p_ptr + offsets, p.to(p_ptr.dtype.element_ty), mask=mask)
        tl.store(v_ptr + offsets, v.to(v_ptr.dtype.element_ty), mask=mask)

        # Dynamic INT8 requantization for m
        m_abs = tl.where(mask, tl.abs(m), 0.0)
        max_m = tl.max(m_abs, axis=0)
        new_scale_m = tl.maximum(max_m / 127.0, 1e-12)
        scaled_m = m / new_scale_m
        rounded_m = tl.where(scaled_m >= 0.0, tl.floor(scaled_m + 0.5), tl.ceil(scaled_m - 0.5))
        new_q_m = tl.clamp(rounded_m, -128.0, 127.0).to(tl.int8)

        tl.store(m_ptr + offsets, new_q_m, mask=mask)
        tl.store(m_scales_ptr + pid, new_scale_m)
else:
    _adamw_kernel_triton = None
    _adamw_kernel_triton_fp32 = None
    _adamw_kernel_triton_cross_precision = None
    _adamw_kernel_triton_8bit = None
    _adamw_kernel_triton_8bit_m_16bit_v = None


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
    _SUPPORTED_PARAM_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
    _SUPPORTED_STATE_DTYPES = {torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8}

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
    if param.dtype not in _SUPPORTED_PARAM_DTYPES:
        return f"param dtype {param.dtype} not supported by Triton kernel"
    if grad.dtype not in _SUPPORTED_PARAM_DTYPES:
        return f"grad dtype {grad.dtype} not supported by Triton kernel"
    if exp_avg.dtype not in _SUPPORTED_STATE_DTYPES:
        return f"exp_avg dtype {exp_avg.dtype} not supported by Triton kernel"
    if exp_avg_sq.dtype not in _SUPPORTED_STATE_DTYPES:
        return f"exp_avg_sq dtype {exp_avg_sq.dtype} not supported by Triton kernel"
    if exp_avg_slow is not None and exp_avg_slow.dtype not in _SUPPORTED_STATE_DTYPES:
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
    exp_avg_scales: Optional[torch.Tensor] = None,
    exp_avg_sq_scales: Optional[torch.Tensor] = None,
    step: int = 1,
) -> bool:
    """
    Executes the Triton fused AdamW kernel on a single tensor.

    Supports pure FP32, cross-precision 16-bit, fused 8-bit (INT8 m + UINT8 v),
    and hybrid 3-byte (INT8 m + 16-bit v) execution paths.
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

    if block_size is None:
        if exp_avg.dtype == torch.int8 and exp_avg_sq.dtype == torch.uint8:
            block_size = 256
        elif exp_avg.dtype == torch.int8 and exp_avg_sq.dtype in (torch.bfloat16, torch.float16, torch.float32):
            block_size = 2048 if n_elements >= 2048 else 256
        else:
            block_size = min(max(64, _next_power_of_2(n_elements)), 1024)

    has_slow = exp_avg_slow is not None and beta3 is not None
    num_blocks = triton.cdiv(n_elements, block_size) if (triton is not None and hasattr(triton, "cdiv")) else (n_elements + block_size - 1) // block_size

    # 0. Fused 8-Bit and 3-Byte Fast Paths (precision='8bit' / 'ultra')
    if exp_avg.dtype == torch.int8 and exp_avg_scales is not None:
        if exp_avg_sq.dtype == torch.uint8 and exp_avg_sq_scales is not None and _adamw_kernel_triton_8bit is not None:
            _adamw_kernel_triton_8bit[(num_blocks,)](
                param,
                grad,
                exp_avg,
                exp_avg_scales,
                exp_avg_sq,
                exp_avg_sq_scales,
                n_elements,
                float(lr),
                float(beta1),
                float(beta2),
                float(eps),
                float(weight_decay),
                float(step_size),
                float(eps_scaled),
                float(inv_scale),
                apply_wd=bool(apply_wd),
                BLOCK_SIZE=block_size,
                cautious=bool(cautious),
                step=int(step),
            )
            return True
        elif exp_avg_sq.dtype in (torch.bfloat16, torch.float16, torch.float32) and _adamw_kernel_triton_8bit_m_16bit_v is not None:
            _adamw_kernel_triton_8bit_m_16bit_v[(num_blocks,)](
                param,
                grad,
                exp_avg,
                exp_avg_scales,
                exp_avg_sq,
                n_elements,
                float(lr),
                float(beta1),
                float(beta2),
                float(eps),
                float(weight_decay),
                float(step_size),
                float(eps_scaled),
                float(inv_scale),
                apply_wd=bool(apply_wd),
                BLOCK_SIZE=block_size,
                cautious=bool(cautious),
            )
            return True

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
    exp_avg_scales: Optional[Union[List[torch.Tensor], Dict[int, torch.Tensor]]] = None,
    exp_avg_sq_scales: Optional[Union[List[torch.Tensor], Dict[int, torch.Tensor]]] = None,
) -> bool:
    """
    Executes the Triton fused AdamW kernel across multiple tensors.
    Works natively on both NVIDIA CUDA and AMD ROCm GPUs with zero C++ compilation.
    Fuses gradient unscaling, AdEMAMix dual-speed momentum, cautious directional masking,
    and 8-bit dynamic block quantization.
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

    slow_ratio = 0.0
    if beta3 is not None and has_slow_list:
        bc3 = 1.0 - (beta3 ** step)
        slow_ratio = alpha * (bc1 / bc3) if bc3 > 0.0 else 0.0

    is_scales_v_list = isinstance(scales_v, list)
    is_m_scales_list = isinstance(exp_avg_scales, list)
    is_v_scales_list = isinstance(exp_avg_sq_scales, list)

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

        s_m = exp_avg_scales[i] if is_m_scales_list else (exp_avg_scales.get(id(p), None) if isinstance(exp_avg_scales, dict) else None)
        s_v = exp_avg_sq_scales[i] if is_v_scales_list else (exp_avg_sq_scales.get(id(p), None) if isinstance(exp_avg_sq_scales, dict) else None)

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
            exp_avg_scales=s_m,
            exp_avg_sq_scales=s_v,
            step=step,
        )

    return True


def triton_adamw_8bit_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_scales: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    exp_avg_sq_scales: Optional[torch.Tensor] = None,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    step_size: float = 1e-3,
    eps_scaled: float = 1e-8,
    inv_scale: float = 1.0,
    apply_wd: bool = True,
    block_size: Optional[int] = None,
    skip_preflight: bool = False,
    cautious: bool = False,
) -> bool:
    """Explicit fast-path for 8-bit AdamW with dynamic block scaling (2 bytes/param)."""
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
        inv_scale=inv_scale,
        apply_wd=apply_wd,
        block_size=block_size,
        skip_preflight=skip_preflight,
        cautious=cautious,
        exp_avg_scales=exp_avg_scales,
        exp_avg_sq_scales=exp_avg_sq_scales,
    )


def triton_multi_tensor_adamw_8bit_step(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_scales: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    exp_avg_sq_scales: Optional[List[torch.Tensor]] = None,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    step: int = 1,
    decay_norm_only: bool = True,
    inv_scale: float = 1.0,
    block_size: Optional[int] = None,
    cautious: bool = False,
    skip_preflight: bool = False,
) -> bool:
    """Multi-tensor fused 8-bit AdamW execution across multiple parameters."""
    return triton_multi_tensor_adamw_step(
        params=params,
        grads=grads,
        exp_avgs=exp_avgs,
        exp_avg_sqs=exp_avg_sqs,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        step=step,
        decay_norm_only=decay_norm_only,
        inv_scale=inv_scale,
        block_size=block_size,
        cautious=cautious,
        skip_preflight=skip_preflight,
        exp_avg_scales=exp_avg_scales,
        exp_avg_sq_scales=exp_avg_sq_scales,
    )


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
