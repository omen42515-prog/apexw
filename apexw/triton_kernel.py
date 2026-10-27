"""
ApexW OpenAI Triton Fused Kernel Backend.

Compiles and executes high-performance fused AdamW kernels via OpenAI Triton.
Automatically compiled down to optimized GPU machine code for both NVIDIA GPUs
(via PTX/SASS) and AMD GPUs (via ROCm / HIP / GCN / CDNA ISA).
"""

import os
import math
from typing import List, Optional, Dict, Any
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

        # Denominator with scaled epsilon
        denom = tl.sqrt(v) + (eps_scaled * sqrt_scale_v)
        step_factor = step_size * sqrt_scale_v
        update = (m / denom) * step_factor

        p = p - update

        # Store back into tensor buffers (Triton automatically casts back to target buffer dtypes)
        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(m_ptr + offsets, m, mask=mask)
        tl.store(v_ptr + offsets, v, mask=mask)


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
    block_size: int = 1024,
) -> bool:
    """Executes the Triton fused AdamW kernel on a single tensor."""
    if not is_triton_available():
        return False

    n_elements = param.numel()
    if n_elements == 0:
        return True

    # Ensure tensors are contiguous
    if not param.is_contiguous():
        param.data = param.contiguous()
    if not grad.is_contiguous():
        grad.data = grad.contiguous()
    if not exp_avg.is_contiguous():
        exp_avg.data = exp_avg.contiguous()
    if not exp_avg_sq.is_contiguous():
        exp_avg_sq.data = exp_avg_sq.contiguous()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    try:
        _adamw_kernel_triton[grid](
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
        )
        return True
    except Exception as e:
        if os.environ.get("APEXW_DEBUG", "0") == "1":
            print(f"[ApexW] Triton kernel execution failed: {e}")
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
    scales_v: Optional[Dict[int, float]] = None,
    inv_scale: float = 1.0,
) -> bool:
    """
    Executes the Triton fused AdamW kernel across multiple tensors.
    Works natively on both NVIDIA CUDA and AMD ROCm GPUs with zero C++ compilation.
    Fuses gradient unscaling in-kernel when running under PyTorch AMP GradScaler.
    """
    if not is_triton_available():
        return False
    if len(params) == 0:
        return True

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    bc2_sqrt = math.sqrt(bc2)
    step_size = lr * bc2_sqrt / bc1
    eps_scaled = eps * bc2_sqrt

    for i, p in enumerate(params):
        g = grads[i]
        m = exp_avgs[i]
        v = exp_avg_sqs[i]

        apply_wd = (weight_decay != 0.0)
        if apply_wd and decay_norm_only and p.dim() < 2:
            apply_wd = False

        scale_v = 1.0
        sqrt_scale_v = 1.0
        if scales_v is not None and id(p) in scales_v:
            scale_v = scales_v[id(p)]
            sqrt_scale_v = math.sqrt(scale_v)

        success = triton_adamw_step(
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
        )
        if not success:
            return False

    return True
