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
) -> Optional[str]:
    """
    Validates tensor layout, device, and dtype before any Triton kernel is launched.
    Returns None if all checks pass, or a human-readable reason string if not supported.
    This is the Fix 4 pre-flight: distinguishes 'unsupported config' (safe to return False)
    from 'failure after kernel already mutated state' (must raise, not silently retry).
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
) -> bool:
    """
    Executes the Triton fused AdamW kernel on a single tensor.

    Fix 4: All preflight checks (device, dtype, layout, shape) are performed BEFORE
    the kernel is launched. Returns False only when no mutations have been made.
    If the kernel is launched and fails, the exception propagates (does not return False),
    preventing the caller from silently re-running the same parameter with a fallback.
    """
    if not is_triton_available():
        return False

    n_elements = param.numel()
    if n_elements == 0:
        return True

    # Fix 4: Pre-flight validation — all checks before any mutation.
    reason = _triton_preflight_check(param, grad, exp_avg, exp_avg_sq)
    if reason is not None:
        if os.environ.get("APEXW_DEBUG", "0") == "1":
            print(f"[ApexW] Triton pre-flight rejected: {reason}")
        return False  # Safe: no kernel has run yet.

    if block_size is None:
        block_size = min(max(64, _next_power_of_2(n_elements)), 1024)

    # Ensure tensors are contiguous — this is a mutation of metadata, not of values.
    if not param.is_contiguous():
        param.data = param.contiguous()
    if not grad.is_contiguous():
        grad.data = grad.contiguous()
    if not exp_avg.is_contiguous():
        exp_avg.data = exp_avg.contiguous()
    if not exp_avg_sq.is_contiguous():
        exp_avg_sq.data = exp_avg_sq.contiguous()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Fix 4: After this point, the kernel WILL mutate param, exp_avg, exp_avg_sq.
    # Any exception here must NOT be caught and silently converted to False —
    # the caller must know that the parameter has been partially updated.
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
    block_size: Optional[int] = None,
) -> bool:
    """
    Executes the Triton fused AdamW kernel across multiple tensors.
    Works natively on both NVIDIA CUDA and AMD ROCm GPUs with zero C++ compilation.
    Fuses gradient unscaling in-kernel when running under PyTorch AMP GradScaler.

    Fix 4: ALL tensors are pre-flight validated before any kernel is launched.
    If any tensor fails validation, returns False before touching any parameter.
    If a kernel succeeds then a later one raises, the exception propagates (not caught)
    so the caller cannot silently re-run the whole group through a fallback path.
    """
    if not is_triton_available():
        return False
    if len(params) == 0:
        return True

    # Fix 4: Validate the ENTIRE batch before launching ANY kernel.
    # This distinguishes "unsupported configuration" (safe False) from
    # "failure mid-batch after some params already mutated" (must raise).
    for i, p in enumerate(params):
        reason = _triton_preflight_check(p, grads[i], exp_avgs[i], exp_avg_sqs[i])
        if reason is not None:
            if os.environ.get("APEXW_DEBUG", "0") == "1":
                print(f"[ApexW] Triton pre-flight rejected param[{i}]: {reason}")
            return False  # Safe: no kernel has run yet.

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    bc2_sqrt = math.sqrt(bc2)
    step_size = lr * bc2_sqrt / bc1
    eps_scaled = eps * bc2_sqrt

    # Fix 4: After this loop begins, parameters are being mutated one by one.
    # Any exception after the first successful kernel call MUST propagate —
    # do not catch and return False, which would cause the caller to re-run
    # the entire group through a fallback path, double-updating already-modified params.
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
        )

    return True
