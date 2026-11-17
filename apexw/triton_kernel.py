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


_DEV_CAP_CACHE: Dict[int, tuple] = {}
_BATCHED_TRITON_ENABLED = True


def _get_device_capability_cached(device: torch.device) -> tuple:
    """Cache CUDA compute capability per device index to avoid per-tensor host queries."""
    idx = device.index if device.index is not None else 0
    cap = _DEV_CAP_CACHE.get(idx)
    if cap is None:
        if torch.cuda.is_available() and hasattr(torch.cuda, "get_device_capability"):
            try:
                cap = torch.cuda.get_device_capability(device)
            except Exception:
                cap = (7, 0)
        else:
            cap = (7, 0)
        _DEV_CAP_CACHE[idx] = cap
    return cap


def _get_num_stages(device: torch.device) -> int:
    return 3 if _get_device_capability_cached(device)[0] >= 8 else 2


def _dtype_to_code(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return -1


def _resolve_fp16_eps(eps: float, eps_scaled: float, is_fp16: bool) -> tuple:
    """Enforce FP16 normal-range epsilon floor (>= 6.1e-5) without perturbing FP32/BF16."""
    if is_fp16 and eps < 6.1e-5:
        ratio = (eps_scaled / eps) if eps > 0.0 else 1.0
        return 6.1e-5, max(eps_scaled, 6.1e-5 * ratio)
    return eps, eps_scaled


if _HAS_TRITON:
    @triton.jit
    def _sanitize_grad_triton(g):
        """Sanitize NaN/Inf gradients in FP32 registers without an extra global memory pass or CPU sync."""
        return tl.where(
            tl.abs(g) < 3.4028234e38,
            g,
            tl.where(g != g, 0.0, tl.where(g > 0.0, 65504.0, -65504.0)),
        )

    @triton.jit
    def _nf4_dequant_triton(q):
        """Dequantize 4-bit NF4 nibble (0..15) in registers matching _NF4_M_CODEBOOK_RAW."""
        q_i = q.to(tl.int32)
        a = tl.abs(q_i - 7)
        mag_01 = tl.where(a == 0, 0.0, 0.0911)
        mag_23 = tl.where(a == 2, 0.1848, 0.2844)
        mag_lo = tl.where(a < 2, mag_01, mag_23)
        mag_45 = tl.where(a == 4, 0.3949, 0.5251)
        mag_67 = tl.where(a == 6, 0.6962, 1.0000)
        mag_hi = tl.where(a < 6, mag_45, mag_67)
        mag = tl.where(a < 4, mag_lo, mag_hi)
        return tl.where(q_i < 7, -mag, mag)

    @triton.jit
    def _nf4_quant_triton(norm_val, u_rand, stochastic: tl.constexpr):
        """Quantize normalized [-1, 1] value to NF4 nibble (0..14) in registers."""
        is_neg = norm_val < 0.0
        ax = tl.abs(norm_val)

        # 3-level binary search across symmetric magnitudes {0.0, 0.0911, 0.1848, 0.2844, 0.3949, 0.5251, 0.6962, 1.0000}
        in_low = ax < 0.2844
        in_mid_low = ax < 0.0911
        in_mid_high = ax < 0.1848
        lo_low = tl.where(in_mid_low, 0, tl.where(in_mid_high, 1, 2))
        clo_low = tl.where(in_mid_low, 0.0, tl.where(in_mid_high, 0.0911, 0.1848))
        chi_low = tl.where(in_mid_low, 0.0911, tl.where(in_mid_high, 0.1848, 0.2844))

        in_upper_mid = ax < 0.5251
        in_u0 = ax < 0.3949
        in_u2 = ax < 0.6962
        lo_high = tl.where(in_upper_mid, tl.where(in_u0, 3, 4), tl.where(in_u2, 5, 6))
        clo_high = tl.where(in_upper_mid, tl.where(in_u0, 0.2844, 0.3949), tl.where(in_u2, 0.5251, 0.6962))
        chi_high = tl.where(in_upper_mid, tl.where(in_u0, 0.3949, 0.5251), tl.where(in_u2, 0.6962, 1.0000))

        lo = tl.where(in_low, lo_low, lo_high)
        c_lo = tl.where(in_low, clo_low, clo_high)
        c_hi = tl.where(in_low, chi_low, chi_high)

        if stochastic:
            prob_hi = (ax - c_lo) / (c_hi - c_lo)
            k = tl.where(u_rand < prob_hi, lo + 1, lo)
        else:
            k = tl.where((ax - c_lo) >= (c_hi - ax), lo + 1, lo)

        return tl.where(is_neg, 7 - k, 7 + k)

    @triton.jit
    def _xorshift32_seed(offsets, pid_val, step):
        step_m = (step & 0xffff) * 9277
        pid_m = (pid_val.to(tl.int32) & 0xffff) * 1973
        seed = (offsets.to(tl.int32) ^ pid_m) + step_m + 1
        seed = tl.where(seed == 0, 1, seed)
        seed = seed ^ (seed << 13)
        seed = seed ^ ((seed >> 17) & 0x7fff)
        seed = seed ^ (seed << 5)
        seed = seed ^ (seed << 13)
        seed = seed ^ ((seed >> 17) & 0x7fff)
        seed = seed ^ (seed << 5)
        return seed

    @triton.jit
    def _xorshift32_next(seed):
        s = seed ^ 0x6b43a9b5
        s = tl.where(s == 0, 1, s)
        s = s ^ (s << 13)
        s = s ^ ((s >> 17) & 0x7fff)
        s = s ^ (s << 5)
        return s

    @triton.jit
    def _load_f32_from_int64_ptr(base_addr, dtype_code: tl.constexpr, offsets, mask, other_val):
        if dtype_code == 0:
            ptr = base_addr.to(tl.pointer_type(tl.float32)) + offsets
            return tl.load(ptr, mask=mask, other=other_val).to(tl.float32)
        elif dtype_code == 1:
            ptr = base_addr.to(tl.pointer_type(tl.float16)) + offsets
            return tl.load(ptr, mask=mask, other=other_val).to(tl.float32)
        else:
            ptr = base_addr.to(tl.pointer_type(tl.bfloat16)) + offsets
            return tl.load(ptr, mask=mask, other=other_val).to(tl.float32)

    @triton.jit
    def _clamp_fp16_val(val):
        return tl.where(val == val, tl.clamp(val, -65504.0, 65504.0), val)

    @triton.jit
    def _store_f32_to_int64_ptr(base_addr, dtype_code: tl.constexpr, offsets, val, mask):
        if dtype_code == 0:
            ptr = base_addr.to(tl.pointer_type(tl.float32)) + offsets
            tl.store(ptr, val.to(tl.float32), mask=mask)
        elif dtype_code == 1:
            ptr = base_addr.to(tl.pointer_type(tl.float16)) + offsets
            tl.store(ptr, _clamp_fp16_val(val).to(tl.float16), mask=mask)
        else:
            ptr = base_addr.to(tl.pointer_type(tl.bfloat16)) + offsets
            tl.store(ptr, val.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _store_f32_to_int64_ptr_sr(base_addr, dtype_code: tl.constexpr, offsets, val, seed, stochastic: tl.constexpr, mask):
        if dtype_code == 0:
            ptr = base_addr.to(tl.pointer_type(tl.float32)) + offsets
            tl.store(ptr, val.to(tl.float32), mask=mask)
        elif dtype_code == 1:
            v_c = _clamp_fp16_val(val)
            if stochastic:
                u = v_c.to(tl.int32, bitcast=True)
                mag = (u & 0x7FFFFFFF) + (seed & 0x1FFF)
                u_sr = ((u & ~0x7FFFFFFF) | (mag & 0x7FFFFFFF)) & ~0x1FFF
                v_c = tl.where(tl.abs(v_c) >= 6.103515625e-5, u_sr.to(tl.float32, bitcast=True), v_c)
            ptr = base_addr.to(tl.pointer_type(tl.float16)) + offsets
            tl.store(ptr, v_c.to(tl.float16), mask=mask)
        else:
            v_b = val
            if stochastic:
                u = v_b.to(tl.int32, bitcast=True)
                mag = (u & 0x7FFFFFFF) + (seed & 0xFFFF)
                u_sr = ((u & ~0x7FFFFFFF) | (mag & 0x7FFFFFFF)) & ~0xFFFF
                v_b = tl.where((u & 0x7FFFFFFF) < 0x7F7F0000, u_sr.to(tl.float32, bitcast=True), v_b)
            ptr = base_addr.to(tl.pointer_type(tl.bfloat16)) + offsets
            tl.store(ptr, v_b.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _store_typed_clamped(ptr, offsets, val, mask):
        if ptr.dtype.element_ty == tl.float16:
            tl.store(ptr + offsets, _clamp_fp16_val(val).to(tl.float16), mask=mask)
        else:
            tl.store(ptr + offsets, val.to(ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _store_typed_clamped_sr(ptr, offsets, val, seed, stochastic: tl.constexpr, mask):
        if ptr.dtype.element_ty == tl.float16:
            v_c = _clamp_fp16_val(val)
            if stochastic:
                u = v_c.to(tl.int32, bitcast=True)
                mag = (u & 0x7FFFFFFF) + (seed & 0x1FFF)
                u_sr = ((u & ~0x7FFFFFFF) | (mag & 0x7FFFFFFF)) & ~0x1FFF
                v_c = tl.where(tl.abs(v_c) >= 6.103515625e-5, u_sr.to(tl.float32, bitcast=True), v_c)
            tl.store(ptr + offsets, v_c.to(tl.float16), mask=mask)
        elif ptr.dtype.element_ty == tl.bfloat16:
            v_b = val
            if stochastic:
                u = v_b.to(tl.int32, bitcast=True)
                mag = (u & 0x7FFFFFFF) + (seed & 0xFFFF)
                u_sr = ((u & ~0x7FFFFFFF) | (mag & 0x7FFFFFFF)) & ~0xFFFF
                v_b = tl.where((u & 0x7FFFFFFF) < 0x7F7F0000, u_sr.to(tl.float32, bitcast=True), v_b)
            tl.store(ptr + offsets, v_b.to(tl.bfloat16), mask=mask)
        else:
            tl.store(ptr + offsets, val.to(ptr.dtype.element_ty), mask=mask)

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
        pid = tl.program_id(axis=0).to(tl.int64)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        g = _sanitize_grad_triton(tl.load(g_ptr + offsets, mask=mask, other=0.0))
        if inv_scale != 1.0:
            g = g * inv_scale
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

        # Decoupled weight decay (clamped >= 0 to prevent polarity inversion)
        if apply_wd:
            p = p * tl.maximum(1.0 - lr * weight_decay, 0.0)

        # Update first-order momentum (EMA)
        m = m * beta1 + g * (1.0 - beta1)

        # Update second-order momentum without dynamic scaling overhead
        v = v * beta2 + (g * g) * (1.0 - beta2)
        v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)

        # Denominator with scaled epsilon
        denom = tl.maximum(tl.sqrt(tl.maximum(v, 0.0)) + eps_scaled, eps_scaled)
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
        QUAD_BLOCK: tl.constexpr = 1,
        IS_QUAD_V: tl.constexpr = False,
    ):
        """
        Cross-Precision Updates (precision='compact' with FP32 weights).
        Merges 16-bit to FP32 load casting, dynamic scaling, optimizer step calculation,
        and FP32 to 16-bit store casting into a single memory pass, eliminating temporary
        tensor allocations.
        """
        pid = tl.program_id(axis=0).to(tl.int64)
        gain_hi = 1.0 + 0.20 * beta1
        gain_lo = 1.0 - 0.35 * beta1

        if IS_QUAD_V:
            q_lane = tl.arange(0, QUAD_BLOCK)
            quad_off = pid * QUAD_BLOCK + q_lane
            mask_q = quad_off < (n_elements >> 2)
            off0 = pid * BLOCK_SIZE + q_lane * 4
            off1 = off0 + 1
            off2 = off0 + 2
            off3 = off0 + 3
            mk0 = off0 < n_elements
            mk1 = off1 < n_elements
            mk2 = off2 < n_elements
            mk3 = off3 < n_elements

            p0 = tl.load(p_ptr + off0, mask=mk0, other=0.0).to(tl.float32)
            p1 = tl.load(p_ptr + off1, mask=mk1, other=0.0).to(tl.float32)
            p2 = tl.load(p_ptr + off2, mask=mk2, other=0.0).to(tl.float32)
            p3 = tl.load(p_ptr + off3, mask=mk3, other=0.0).to(tl.float32)
            g0 = _sanitize_grad_triton(tl.load(g_ptr + off0, mask=mk0, other=0.0).to(tl.float32))
            g1 = _sanitize_grad_triton(tl.load(g_ptr + off1, mask=mk1, other=0.0).to(tl.float32))
            g2 = _sanitize_grad_triton(tl.load(g_ptr + off2, mask=mk2, other=0.0).to(tl.float32))
            g3 = _sanitize_grad_triton(tl.load(g_ptr + off3, mask=mk3, other=0.0).to(tl.float32))
            if inv_scale != 1.0:
                g0 = g0 * inv_scale
                g1 = g1 * inv_scale
                g2 = g2 * inv_scale
                g3 = g3 * inv_scale

            m0 = tl.load(m_ptr + off0, mask=mk0, other=0.0).to(tl.float32)
            m1 = tl.load(m_ptr + off1, mask=mk1, other=0.0).to(tl.float32)
            m2 = tl.load(m_ptr + off2, mask=mk2, other=0.0).to(tl.float32)
            m3 = tl.load(m_ptr + off3, mask=mk3, other=0.0).to(tl.float32)
            v_q = tl.load(v_ptr + quad_off, mask=mask_q, other=0.0).to(tl.float32)

            if apply_wd:
                wd_f = tl.maximum(1.0 - lr * weight_decay, 0.0)
                p0 = p0 * wd_f
                p1 = p1 * wd_f
                p2 = p2 * wd_f
                p3 = p3 * wd_f

            m0 = m0 * beta1 + g0 * (1.0 - beta1)
            m1 = m1 * beta1 + g1 * (1.0 - beta1)
            m2 = m2 * beta1 + g2 * (1.0 - beta1)
            m3 = m3 * beta1 + g3 * (1.0 - beta1)
            g_sq_mean = 0.25 * (g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)

            if has_scale_v:
                v_q = v_q * beta2 + g_sq_mean * ((1.0 - beta2) * scale_v)
                v_q = tl.where(tl.abs(v_q) < 3.4028234e38, v_q, 0.0)
                inv_denom = (step_size * sqrt_scale_v) / tl.maximum(tl.sqrt(tl.maximum(v_q, 0.0)) + (eps_scaled * sqrt_scale_v), eps * sqrt_scale_v)
            else:
                v_q = v_q * beta2 + g_sq_mean * (1.0 - beta2)
                v_q = tl.where(tl.abs(v_q) < 3.4028234e38, v_q, 0.0)
                inv_denom = step_size / tl.maximum(tl.sqrt(tl.maximum(v_q, 0.0)) + eps_scaled, eps)

            ml0 = m0 * beta1 + g0 * (1.0 - beta1)
            ml1 = m1 * beta1 + g1 * (1.0 - beta1)
            ml2 = m2 * beta1 + g2 * (1.0 - beta1)
            ml3 = m3 * beta1 + g3 * (1.0 - beta1)
            u0 = (ml0 * tl.where(m0 * g0 >= 0.0, gain_hi, gain_lo)) * inv_denom
            u1 = (ml1 * tl.where(m1 * g1 >= 0.0, gain_hi, gain_lo)) * inv_denom
            u2 = (ml2 * tl.where(m2 * g2 >= 0.0, gain_hi, gain_lo)) * inv_denom
            u3 = (ml3 * tl.where(m3 * g3 >= 0.0, gain_hi, gain_lo)) * inv_denom

            tl.store(p_ptr + off0, p0 - u0, mask=mk0)
            tl.store(p_ptr + off1, p1 - u1, mask=mk1)
            tl.store(p_ptr + off2, p2 - u2, mask=mk2)
            tl.store(p_ptr + off3, p3 - u3, mask=mk3)
            _store_typed_clamped(m_ptr + off0, 0, m0, mk0)
            _store_typed_clamped(m_ptr + off1, 0, m1, mk1)
            _store_typed_clamped(m_ptr + off2, 0, m2, mk2)
            _store_typed_clamped(m_ptr + off3, 0, m3, mk3)
            _store_typed_clamped(v_ptr, quad_off, v_q, mask_q)
            return

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load parameter (FP32) and unscale gradient in registers
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = _sanitize_grad_triton(tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        if inv_scale != 1.0:
            g = g * inv_scale

        # Load 16-bit compact optimizer states and promote to FP32 in registers
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Decoupled weight decay
        if apply_wd:
            p = p * tl.maximum(1.0 - lr * weight_decay, 0.0)

        # Update first-order momentum (EMA) in FP32 registers
        m = m * beta1 + g * (1.0 - beta1)

        # AdEMAMix dual-speed momentum
        if has_slow_momentum:
            m_slow = tl.load(m_slow_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            m_slow = m_slow * beta3 + g * (1.0 - beta3)
            _store_typed_clamped(m_slow_ptr, offsets, m_slow, mask)
            m_eff = m + slow_ratio * m_slow
        else:
            m_eff = m

        # Update second-order momentum (EMA) with merged dynamic scaling
        if has_scale_v:
            v = v * beta2 + (g * g) * ((1.0 - beta2) * scale_v)
            v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)
            denom = tl.maximum(tl.sqrt(tl.maximum(v, 0.0)) + (eps_scaled * sqrt_scale_v), eps * sqrt_scale_v)
            step_factor = step_size * sqrt_scale_v
        else:
            v = v * beta2 + (g * g) * (1.0 - beta2)
            v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)
            denom = tl.maximum(tl.sqrt(tl.maximum(v, 0.0)) + eps_scaled, eps)
            step_factor = step_size

        update = (m_eff / denom) * step_factor

        # Cautious directional update masking (block-normalized to preserve magnitude)
        if cautious:
            c_mask = (g * m_eff >= 0.0).to(tl.float32)
            valid = mask.to(tl.float32)
            n_valid = tl.maximum(tl.sum(valid, axis=0), 1.0)
            c_mean = tl.maximum(tl.sum(c_mask * valid, axis=0) / n_valid, 1e-3)
            update = update * (c_mask / c_mean)

        p = p - update

        # Store back in a single pass: FP32 param, clamped/casted 16-bit states
        tl.store(p_ptr + offsets, p, mask=mask)
        _store_typed_clamped(m_ptr, offsets, m, mask)
        _store_typed_clamped(v_ptr, offsets, v, mask)

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
        QUAD_BLOCK: tl.constexpr = 1,
        IS_QUAD_V: tl.constexpr = False,
    ):
        pid = tl.program_id(axis=0).to(tl.int64)
        gain_hi = 1.0 + 0.20 * beta1
        gain_lo = 1.0 - 0.35 * beta1

        if IS_QUAD_V:
            q_lane = tl.arange(0, QUAD_BLOCK)
            quad_off = pid * QUAD_BLOCK + q_lane
            mask_q = quad_off < (n_elements >> 2)
            off0 = pid * BLOCK_SIZE + q_lane * 4
            off1 = off0 + 1
            off2 = off0 + 2
            off3 = off0 + 3
            mk0 = off0 < n_elements
            mk1 = off1 < n_elements
            mk2 = off2 < n_elements
            mk3 = off3 < n_elements

            p0 = tl.load(p_ptr + off0, mask=mk0, other=0.0).to(tl.float32)
            p1 = tl.load(p_ptr + off1, mask=mk1, other=0.0).to(tl.float32)
            p2 = tl.load(p_ptr + off2, mask=mk2, other=0.0).to(tl.float32)
            p3 = tl.load(p_ptr + off3, mask=mk3, other=0.0).to(tl.float32)
            g0 = _sanitize_grad_triton(tl.load(g_ptr + off0, mask=mk0, other=0.0).to(tl.float32))
            g1 = _sanitize_grad_triton(tl.load(g_ptr + off1, mask=mk1, other=0.0).to(tl.float32))
            g2 = _sanitize_grad_triton(tl.load(g_ptr + off2, mask=mk2, other=0.0).to(tl.float32))
            g3 = _sanitize_grad_triton(tl.load(g_ptr + off3, mask=mk3, other=0.0).to(tl.float32))
            if inv_scale != 1.0:
                g0 = g0 * inv_scale
                g1 = g1 * inv_scale
                g2 = g2 * inv_scale
                g3 = g3 * inv_scale

            m0 = tl.load(m_ptr + off0, mask=mk0, other=0.0).to(tl.float32)
            m1 = tl.load(m_ptr + off1, mask=mk1, other=0.0).to(tl.float32)
            m2 = tl.load(m_ptr + off2, mask=mk2, other=0.0).to(tl.float32)
            m3 = tl.load(m_ptr + off3, mask=mk3, other=0.0).to(tl.float32)
            v_q = tl.load(v_ptr + quad_off, mask=mask_q, other=0.0).to(tl.float32)

            if apply_wd:
                wd_f = tl.maximum(1.0 - lr * weight_decay, 0.0)
                p0 = p0 * wd_f
                p1 = p1 * wd_f
                p2 = p2 * wd_f
                p3 = p3 * wd_f

            m0 = m0 * beta1 + g0 * (1.0 - beta1)
            m1 = m1 * beta1 + g1 * (1.0 - beta1)
            m2 = m2 * beta1 + g2 * (1.0 - beta1)
            m3 = m3 * beta1 + g3 * (1.0 - beta1)
            g_sq_mean = 0.25 * (g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)
            v_q = v_q * beta2 + g_sq_mean * ((1.0 - beta2) * scale_v)
            v_q = tl.where(tl.abs(v_q) < 3.4028234e38, v_q, 0.0)

            inv_denom = (step_size * sqrt_scale_v) / tl.maximum(tl.sqrt(tl.maximum(v_q, 0.0)) + (eps_scaled * sqrt_scale_v), eps * sqrt_scale_v)
            ml0 = m0 * beta1 + g0 * (1.0 - beta1)
            ml1 = m1 * beta1 + g1 * (1.0 - beta1)
            ml2 = m2 * beta1 + g2 * (1.0 - beta1)
            ml3 = m3 * beta1 + g3 * (1.0 - beta1)
            u0 = (ml0 * tl.where(m0 * g0 >= 0.0, gain_hi, gain_lo)) * inv_denom
            u1 = (ml1 * tl.where(m1 * g1 >= 0.0, gain_hi, gain_lo)) * inv_denom
            u2 = (ml2 * tl.where(m2 * g2 >= 0.0, gain_hi, gain_lo)) * inv_denom
            u3 = (ml3 * tl.where(m3 * g3 >= 0.0, gain_hi, gain_lo)) * inv_denom

            _store_typed_clamped(p_ptr + off0, 0, p0 - u0, mk0)
            _store_typed_clamped(p_ptr + off1, 0, p1 - u1, mk1)
            _store_typed_clamped(p_ptr + off2, 0, p2 - u2, mk2)
            _store_typed_clamped(p_ptr + off3, 0, p3 - u3, mk3)
            _store_typed_clamped(m_ptr + off0, 0, m0, mk0)
            _store_typed_clamped(m_ptr + off1, 0, m1, mk1)
            _store_typed_clamped(m_ptr + off2, 0, m2, mk2)
            _store_typed_clamped(m_ptr + off3, 0, m3, mk3)
            _store_typed_clamped(v_ptr, quad_off, v_q, mask_q)
            return

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load parameter and optimizer states in float32 for maximum precision
        # Fuses AMP gradient unscaling directly in registers (zero auxiliary VRAM allocation)
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = _sanitize_grad_triton(tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        if inv_scale != 1.0:
            g = g * inv_scale
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Decoupled weight decay
        if apply_wd:
            p = p * tl.maximum(1.0 - lr * weight_decay, 0.0)

        # Update first-order momentum (EMA)
        m = m * beta1 + g * (1.0 - beta1)

        # Update second-order momentum with dynamic scale factor (EMA)
        v = v * beta2 + (g * g) * ((1.0 - beta2) * scale_v)
        v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)

        # AdEMAMix dual-speed momentum (Pagliardini et al., 2024)
        # slow_ratio = alpha * (bc1/bc3) already incorporates bias corrections,
        # matching the foreach and single-tensor paths exactly.
        if has_slow_momentum:
            m_slow = tl.load(m_slow_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            m_slow = m_slow * beta3 + g * (1.0 - beta3)
            _store_typed_clamped(m_slow_ptr, offsets, m_slow, mask)
            m_eff = m + slow_ratio * m_slow
        else:
            m_eff = m

        # Denominator with scaled epsilon
        denom = tl.maximum(tl.sqrt(tl.maximum(v, 0.0)) + (eps_scaled * sqrt_scale_v), eps * sqrt_scale_v)
        step_factor = step_size * sqrt_scale_v
        update = (m_eff / denom) * step_factor

        # Cautious directional update masking (Liang et al., 2024)
        if cautious:
            c_mask = (g * m_eff >= 0.0).to(tl.float32)
            valid = mask.to(tl.float32)
            n_valid = tl.maximum(tl.sum(valid, axis=0), 1.0)
            c_mean = tl.maximum(tl.sum(c_mask * valid, axis=0) / n_valid, 1e-3)
            update = update * (c_mask / c_mean)

        p = p - update

        # Store back into tensor buffers with FP16 overflow protection
        _store_typed_clamped(p_ptr, offsets, p, mask)
        _store_typed_clamped(m_ptr, offsets, m, mask)
        _store_typed_clamped(v_ptr, offsets, v, mask)

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
        is_unscaled: tl.constexpr = False,
        stochastic: tl.constexpr = True,
    ):
        """
        Fused 8-bit AdamW kernel with dynamic block-wise scaling (2 bytes per param).
        m is stored in INT8 (with FP32 scale per block) and v in UINT8 (with FP32 scale per block).
        Dequantization, EMA updates, weight decay, bias-corrected step,
        and intra-block dynamic re-quantization evaluate entirely in FP32 SRAM registers.
        Includes high-throughput XorShift32 stochastic dithering on m and v to eliminate drift.
        """
        pid = tl.program_id(axis=0).to(tl.int64)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load scales for this block
        scale_m = tl.load(m_scales_ptr + pid)
        scale_v = tl.load(v_scales_ptr + pid)

        # Load parameters and gradients in float32
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = _sanitize_grad_triton(tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        if not is_unscaled:
            if inv_scale != 1.0:
                g = g * inv_scale

        # Load 8-bit momentum and variance, dequantize to float32
        q_m = tl.load(m_ptr + offsets, mask=mask, other=0).to(tl.float32)
        q_v = tl.load(v_ptr + offsets, mask=mask, other=0).to(tl.float32)
        m = q_m * scale_m
        # Hardware-native 24-octave logarithmic dequantization for variance
        r_v = tl.where(q_v > 0.0, tl.exp2((q_v - 1.0) * (24.0 / 254.0) - 24.0), 0.0)
        v = r_v * scale_v

        # Decoupled weight decay (strictly non-negative to guard against polarity inversion on large lr/wd)
        if apply_wd:
            p = p * tl.maximum(1.0 - lr * weight_decay, 0.0)

        # Update momentum in float32 registers
        m = m * beta1 + g * (1.0 - beta1)

        # Update variance in float32 registers and guard non-finite overflow
        v = v * beta2 + (g * g) * (1.0 - beta2)
        v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)

        # Compute denominator and parameter update with Fused Nesterov-SNR Lookahead
        denom = tl.maximum(tl.sqrt(tl.maximum(v, 0.0)) + eps_scaled, eps)
        if cautious:
            update = (m / denom) * step_size
            c_mask = (g * m >= 0.0).to(tl.float32)
            valid = mask.to(tl.float32)
            n_valid = tl.maximum(tl.sum(valid, axis=0), 1.0)
            c_mean = tl.maximum(tl.sum(c_mask * valid, axis=0) / n_valid, 1e-3)
            update = update * (c_mask / c_mean)
        else:
            m_look = m * beta1 + g * (1.0 - beta1)
            gain = tl.where(m * g >= 0.0, 1.0 + 0.20 * beta1, 1.0 - 0.35 * beta1)
            update = ((m_look * gain) / denom) * step_size

        p = p - update

        # Store updated parameter back to global memory with FP16 overflow clamp
        _store_typed_clamped(p_ptr, offsets, p, mask)

        # Dynamic INT8 requantization for m (block-wise signed with optional unbiased stochastic rounding)
        m_abs = tl.where(mask, tl.abs(m), 0.0)
        max_m = tl.max(m_abs, axis=0)
        target_scale_m = max_m / 127.0
        # EMA-MAX: scale jumps up instantly but decays at most 10%/step.
        new_scale_m = tl.maximum(tl.maximum(target_scale_m, scale_m * 0.9), 1e-12)
        inv_scale_m = 1.0 / new_scale_m
        scaled_m = m * inv_scale_m

        if stochastic:
            seed = _xorshift32_seed(offsets, pid, step)
            noise = (((seed >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            noise_v = ((seed & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            rounded_m = tl.floor(scaled_m + noise + 0.5)
        else:
            noise_v = 0.0
            rounded_m = tl.where(scaled_m >= 0.0, tl.floor(scaled_m + 0.5), tl.ceil(scaled_m - 0.5))

        new_q_m = tl.clamp(rounded_m, -127.0, 127.0).to(tl.int8)

        tl.store(m_ptr + offsets, new_q_m, mask=mask)
        tl.store(m_scales_ptr + pid, new_scale_m)

        # Dynamic 24-octave Log-scale UINT8 requantization for v
        v_pos = tl.where(mask, v, 0.0)
        max_v = tl.max(v_pos, axis=0)
        new_scale_v = tl.maximum(tl.maximum(max_v, scale_v * 0.95), 1e-12)
        inv_scale_v = 1.0 / new_scale_v
        r_new = tl.maximum(v * inv_scale_v, 5.9604645e-8)  # 2^-24 minimum ratio
        log_r = tl.log2(r_new)

        q_v_calc = tl.where(
            v > 0.0,
            tl.clamp(tl.floor(log_r * (254.0 / 24.0) + (255.5 + noise_v)), 1.0, 255.0),
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
        pid = tl.program_id(axis=0).to(tl.int64)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        scale_m = tl.load(m_scales_ptr + pid)

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = _sanitize_grad_triton(tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        if inv_scale != 1.0:
            g = g * inv_scale

        q_m = tl.load(m_ptr + offsets, mask=mask, other=0).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = q_m * scale_m

        if apply_wd:
            p = p * tl.maximum(1.0 - lr * weight_decay, 0.0)

        m = m * beta1 + g * (1.0 - beta1)
        v = v * beta2 + (g * g) * (1.0 - beta2)
        v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)

        denom = tl.maximum(tl.sqrt(tl.maximum(v, 0.0)) + eps_scaled, eps)
        update = (m / denom) * step_size

        if cautious:
            c_mask = (g * m >= 0.0).to(tl.float32)
            valid = mask.to(tl.float32)
            n_valid = tl.maximum(tl.sum(valid, axis=0), 1.0)
            c_mean = tl.maximum(tl.sum(c_mask * valid, axis=0) / n_valid, 1e-3)
            update = update * (c_mask / c_mean)

        p = p - update

        _store_typed_clamped(p_ptr, offsets, p, mask)
        _store_typed_clamped(v_ptr, offsets, v, mask)

        # Dynamic INT8 requantization for m
        m_abs = tl.where(mask, tl.abs(m), 0.0)
        max_m = tl.max(m_abs, axis=0)
        new_scale_m = tl.maximum(max_m / 127.0, 1e-12)
        scaled_m = m / new_scale_m
        rounded_m = tl.where(scaled_m >= 0.0, tl.floor(scaled_m + 0.5), tl.ceil(scaled_m - 0.5))
        new_q_m = tl.clamp(rounded_m, -127.0, 127.0).to(tl.int8)

        tl.store(m_ptr + offsets, new_q_m, mask=mask)
        tl.store(m_scales_ptr + pid, new_scale_m)

    @triton.jit
    def _adamw_kernel_triton_8bit_int4(
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
        HALF_BLOCK: tl.constexpr,
        cautious: tl.constexpr = False,
        step: int = 1,
        is_unscaled: tl.constexpr = False,
        stochastic: tl.constexpr = True,
        QUAD_BLOCK: tl.constexpr = 1,
        IS_QUAD_V: tl.constexpr = False,
    ):
        """
        Fused Sub-Byte NF4 INT4 momentum + 24-octave Log-UINT8 (or Quad-V 0.25B) variance Triton kernel.
        """
        pid = tl.program_id(axis=0).to(tl.int64)
        scale_m = tl.load(m_scales_ptr + pid)
        scale_v = tl.load(v_scales_ptr + pid)
        gain_hi = 1.0 + 0.20 * beta1
        gain_lo = 1.0 - 0.35 * beta1

        if IS_QUAD_V:
            q_lane = tl.arange(0, QUAD_BLOCK)
            quad_off = pid * QUAD_BLOCK + q_lane
            n_quads = n_elements >> 2
            mask_q = quad_off < n_quads

            byte0 = pid * HALF_BLOCK + q_lane * 2
            byte1 = byte0 + 1
            n_bytes = (n_elements + 1) >> 1
            mask_b0 = byte0 < n_bytes
            mask_b1 = byte1 < n_bytes

            off0 = pid * BLOCK_SIZE + q_lane * 4
            off1 = off0 + 1
            off2 = off0 + 2
            off3 = off0 + 3
            mk0 = off0 < n_elements
            mk1 = off1 < n_elements
            mk2 = off2 < n_elements
            mk3 = off3 < n_elements

            pk0 = tl.load(m_ptr + byte0, mask=mask_b0, other=0x77).to(tl.int32)
            pk1 = tl.load(m_ptr + byte1, mask=mask_b1, other=0x77).to(tl.int32)
            m0 = _nf4_dequant_triton((pk0 >> 4) & 0xF) * scale_m
            m1 = _nf4_dequant_triton(pk0 & 0xF) * scale_m
            m2 = _nf4_dequant_triton((pk1 >> 4) & 0xF) * scale_m
            m3 = _nf4_dequant_triton(pk1 & 0xF) * scale_m

            qv = tl.load(v_ptr + quad_off, mask=mask_q, other=0).to(tl.float32)
            v_q = tl.where(qv > 0.0, tl.exp2((qv - 1.0) * (24.0 / 254.0) - 24.0), 0.0) * scale_v

            p0 = tl.load(p_ptr + off0, mask=mk0, other=0.0).to(tl.float32)
            p1 = tl.load(p_ptr + off1, mask=mk1, other=0.0).to(tl.float32)
            p2 = tl.load(p_ptr + off2, mask=mk2, other=0.0).to(tl.float32)
            p3 = tl.load(p_ptr + off3, mask=mk3, other=0.0).to(tl.float32)

            g0 = _sanitize_grad_triton(tl.load(g_ptr + off0, mask=mk0, other=0.0).to(tl.float32))
            g1 = _sanitize_grad_triton(tl.load(g_ptr + off1, mask=mk1, other=0.0).to(tl.float32))
            g2 = _sanitize_grad_triton(tl.load(g_ptr + off2, mask=mk2, other=0.0).to(tl.float32))
            g3 = _sanitize_grad_triton(tl.load(g_ptr + off3, mask=mk3, other=0.0).to(tl.float32))
            if not is_unscaled:
                if inv_scale != 1.0:
                    g0 = g0 * inv_scale
                    g1 = g1 * inv_scale
                    g2 = g2 * inv_scale
                    g3 = g3 * inv_scale

            if apply_wd:
                wd_factor = tl.maximum(1.0 - lr * weight_decay, 0.0)
                p0 = p0 * wd_factor
                p1 = p1 * wd_factor
                p2 = p2 * wd_factor
                p3 = p3 * wd_factor

            m0 = m0 * beta1 + g0 * (1.0 - beta1)
            m1 = m1 * beta1 + g1 * (1.0 - beta1)
            m2 = m2 * beta1 + g2 * (1.0 - beta1)
            m3 = m3 * beta1 + g3 * (1.0 - beta1)

            g_sq_mean = 0.25 * (g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)
            v_q = v_q * beta2 + g_sq_mean * (1.0 - beta2)
            v_q = tl.where(tl.abs(v_q) < 3.4028234e38, v_q, 0.0)

            true_eps = tl.where(eps > 1e-6, 1e-8, eps)
            true_eps_scaled = eps_scaled * (true_eps / tl.maximum(eps, 1e-12))
            inv_denom = step_size / tl.maximum(tl.sqrt(tl.maximum(v_q, 0.0)) + true_eps_scaled, true_eps)

            if cautious:
                u0 = m0 * inv_denom
                u1 = m1 * inv_denom
                u2 = m2 * inv_denom
                u3 = m3 * inv_denom
                cm0 = tl.where(mk0 & (g0 * m0 >= 0.0), 1.0, 0.0)
                cm1 = tl.where(mk1 & (g1 * m1 >= 0.0), 1.0, 0.0)
                cm2 = tl.where(mk2 & (g2 * m2 >= 0.0), 1.0, 0.0)
                cm3 = tl.where(mk3 & (g3 * m3 >= 0.0), 1.0, 0.0)
                n_valid = tl.maximum(
                    tl.sum(mk0.to(tl.float32), axis=0)
                    + tl.sum(mk1.to(tl.float32), axis=0)
                    + tl.sum(mk2.to(tl.float32), axis=0)
                    + tl.sum(mk3.to(tl.float32), axis=0),
                    1.0,
                )
                c_mean = tl.maximum(
                    (tl.sum(cm0, axis=0) + tl.sum(cm1, axis=0) + tl.sum(cm2, axis=0) + tl.sum(cm3, axis=0)) / n_valid,
                    1e-3,
                )
                u0 = u0 * (cm0 / c_mean)
                u1 = u1 * (cm1 / c_mean)
                u2 = u2 * (cm2 / c_mean)
                u3 = u3 * (cm3 / c_mean)
            else:
                ml0 = m0 * beta1 + g0 * (1.0 - beta1)
                ml1 = m1 * beta1 + g1 * (1.0 - beta1)
                ml2 = m2 * beta1 + g2 * (1.0 - beta1)
                ml3 = m3 * beta1 + g3 * (1.0 - beta1)
                u0 = (ml0 * tl.where(m0 * g0 >= 0.0, gain_hi, gain_lo)) * inv_denom
                u1 = (ml1 * tl.where(m1 * g1 >= 0.0, gain_hi, gain_lo)) * inv_denom
                u2 = (ml2 * tl.where(m2 * g2 >= 0.0, gain_hi, gain_lo)) * inv_denom
                u3 = (ml3 * tl.where(m3 * g3 >= 0.0, gain_hi, gain_lo)) * inv_denom

            if stochastic:
                seed0 = _xorshift32_seed(off0, pid, step)
                seed1 = _xorshift32_next(seed0)
                u_m0 = (((seed0 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                u_m1 = ((seed0 & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                u_m2 = (((seed1 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                u_m3 = ((seed1 & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                seed2 = _xorshift32_next(seed1)
                noise_v = ((seed2 & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            else:
                seed0 = 0
                seed1 = 0
                seed2 = 0
                u_m0 = 0.0
                u_m1 = 0.0
                u_m2 = 0.0
                u_m3 = 0.0
                noise_v = 0.0

            _store_typed_clamped_sr(p_ptr, off0, p0 - u0, seed0, stochastic, mk0)
            _store_typed_clamped_sr(p_ptr, off1, p1 - u1, seed1, stochastic, mk1)
            _store_typed_clamped_sr(p_ptr, off2, p2 - u2, seed2, stochastic, mk2)
            _store_typed_clamped_sr(p_ptr, off3, p3 - u3, seed0 ^ seed1, stochastic, mk3)

            max_m = tl.maximum(
                tl.maximum(tl.max(tl.where(mk0, tl.abs(m0), 0.0), axis=0), tl.max(tl.where(mk1, tl.abs(m1), 0.0), axis=0)),
                tl.maximum(tl.max(tl.where(mk2, tl.abs(m2), 0.0), axis=0), tl.max(tl.where(mk3, tl.abs(m3), 0.0), axis=0)),
            )
            new_scale_m = tl.maximum(tl.maximum(max_m, scale_m * 0.9), 1e-12)
            inv_scale_m = 1.0 / new_scale_m

            nq0 = _nf4_quant_triton(tl.clamp(m0 * inv_scale_m, -1.0, 1.0), u_m0, stochastic)
            nq1 = tl.where(mk1, _nf4_quant_triton(tl.clamp(m1 * inv_scale_m, -1.0, 1.0), u_m1, stochastic), 7)
            nq2 = _nf4_quant_triton(tl.clamp(m2 * inv_scale_m, -1.0, 1.0), u_m2, stochastic)
            nq3 = tl.where(mk3, _nf4_quant_triton(tl.clamp(m3 * inv_scale_m, -1.0, 1.0), u_m3, stochastic), 7)

            tl.store(m_ptr + byte0, ((nq0 << 4) | nq1).to(tl.uint8), mask=mask_b0)
            tl.store(m_ptr + byte1, ((nq2 << 4) | nq3).to(tl.uint8), mask=mask_b1)
            tl.store(m_scales_ptr + pid, new_scale_m)

            max_v = tl.max(tl.where(mask_q, v_q, 0.0), axis=0)
            new_scale_v = tl.maximum(tl.maximum(max_v, scale_v * 0.95), 1e-12)
            rq = tl.maximum(v_q / new_scale_v, 5.9604645e-8)
            nq_v = tl.where(
                v_q > 0.0,
                tl.clamp(tl.floor(tl.log2(rq) * (254.0 / 24.0) + (255.5 + noise_v)), 1.0, 255.0),
                0.0,
            ).to(tl.uint8)
            tl.store(v_ptr + quad_off, nq_v, mask=mask_q)
            tl.store(v_scales_ptr + pid, new_scale_v)
            return

        lane = tl.arange(0, HALF_BLOCK)
        byte_offsets = pid * HALF_BLOCK + lane
        n_bytes = (n_elements + 1) >> 1
        mask_byte = byte_offsets < n_bytes

        even_off = pid * BLOCK_SIZE + lane * 2
        odd_off = even_off + 1
        mask_0 = even_off < n_elements
        mask_1 = odd_off < n_elements

        packed_m = tl.load(m_ptr + byte_offsets, mask=mask_byte, other=0x77).to(tl.int32)
        q_m0 = (packed_m >> 4) & 0xF
        q_m1 = packed_m & 0xF

        p0 = tl.load(p_ptr + even_off, mask=mask_0, other=0.0).to(tl.float32)
        p1 = tl.load(p_ptr + odd_off, mask=mask_1, other=0.0).to(tl.float32)
        g0 = _sanitize_grad_triton(tl.load(g_ptr + even_off, mask=mask_0, other=0.0).to(tl.float32))
        g1 = _sanitize_grad_triton(tl.load(g_ptr + odd_off, mask=mask_1, other=0.0).to(tl.float32))
        if not is_unscaled:
            if inv_scale != 1.0:
                g0 = g0 * inv_scale
                g1 = g1 * inv_scale

        qv0 = tl.load(v_ptr + even_off, mask=mask_0, other=0).to(tl.float32)
        qv1 = tl.load(v_ptr + odd_off, mask=mask_1, other=0).to(tl.float32)

        m0 = _nf4_dequant_triton(q_m0) * scale_m
        m1 = _nf4_dequant_triton(q_m1) * scale_m
        rv0 = tl.where(qv0 > 0.0, tl.exp2((qv0 - 1.0) * (24.0 / 254.0) - 24.0), 0.0)
        rv1 = tl.where(qv1 > 0.0, tl.exp2((qv1 - 1.0) * (24.0 / 254.0) - 24.0), 0.0)
        v0 = rv0 * scale_v
        v1 = rv1 * scale_v

        if apply_wd:
            wd_factor = tl.maximum(1.0 - lr * weight_decay, 0.0)
            p0 = p0 * wd_factor
            p1 = p1 * wd_factor

        m0 = m0 * beta1 + g0 * (1.0 - beta1)
        m1 = m1 * beta1 + g1 * (1.0 - beta1)
        v0 = v0 * beta2 + (g0 * g0) * (1.0 - beta2)
        v1 = v1 * beta2 + (g1 * g1) * (1.0 - beta2)
        v0 = tl.where(tl.abs(v0) < 3.4028234e38, v0, 0.0)
        v1 = tl.where(tl.abs(v1) < 3.4028234e38, v1, 0.0)

        denom0 = tl.maximum(tl.sqrt(v0) + eps_scaled, eps)
        denom1 = tl.maximum(tl.sqrt(v1) + eps_scaled, eps)

        if cautious:
            u0 = (m0 / denom0) * step_size
            u1 = (m1 / denom1) * step_size
            cm0 = tl.where(mask_0 & (g0 * m0 >= 0.0), 1.0, 0.0)
            cm1 = tl.where(mask_1 & (g1 * m1 >= 0.0), 1.0, 0.0)
            n_valid = tl.maximum(
                tl.sum(mask_0.to(tl.float32), axis=0) + tl.sum(mask_1.to(tl.float32), axis=0),
                1.0,
            )
            c_mean = tl.maximum((tl.sum(cm0, axis=0) + tl.sum(cm1, axis=0)) / n_valid, 1e-3)
            u0 = u0 * (cm0 / c_mean)
            u1 = u1 * (cm1 / c_mean)
        else:
            ml0 = m0 * beta1 + g0 * (1.0 - beta1)
            ml1 = m1 * beta1 + g1 * (1.0 - beta1)
            u0 = ((ml0 * tl.where(m0 * g0 >= 0.0, gain_hi, gain_lo)) / denom0) * step_size
            u1 = ((ml1 * tl.where(m1 * g1 >= 0.0, gain_hi, gain_lo)) / denom1) * step_size

        p0 = p0 - u0
        p1 = p1 - u1
        _store_typed_clamped(p_ptr, even_off, p0, mask_0)
        _store_typed_clamped(p_ptr, odd_off, p1, mask_1)

        # NF4 block-wise requantization for m (EMA-MAX scale smoothing: 0.9x decay floor)
        m0_abs = tl.where(mask_0, tl.abs(m0), 0.0)
        m1_abs = tl.where(mask_1, tl.abs(m1), 0.0)
        max_m = tl.maximum(tl.max(m0_abs, axis=0), tl.max(m1_abs, axis=0))
        new_scale_m = tl.maximum(tl.maximum(max_m, scale_m * 0.9), 1e-12)
        inv_scale_m = 1.0 / new_scale_m
        norm0 = tl.clamp(m0 * inv_scale_m, -1.0, 1.0)
        norm1 = tl.clamp(m1 * inv_scale_m, -1.0, 1.0)

        if stochastic:
            seed0 = _xorshift32_seed(even_off, pid, step)
            seed1 = _xorshift32_next(seed0)
            u_m0 = (((seed0 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0))
            u_m1 = (((seed1 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0))
            noise_v0 = ((seed0 & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            noise_v1 = ((seed1 & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
        else:
            u_m0 = 0.0
            u_m1 = 0.0
            noise_v0 = 0.0
            noise_v1 = 0.0

        nq_m0 = _nf4_quant_triton(norm0, u_m0, stochastic)
        nq_m1 = tl.where(mask_1, _nf4_quant_triton(norm1, u_m1, stochastic), 7)
        new_packed_m = ((nq_m0 << 4) | nq_m1).to(tl.uint8)
        tl.store(m_ptr + byte_offsets, new_packed_m, mask=mask_byte)
        tl.store(m_scales_ptr + pid, new_scale_m)

        # Log-UINT8 requantization for v
        v0_pos = tl.where(mask_0, v0, 0.0)
        v1_pos = tl.where(mask_1, v1, 0.0)
        max_v = tl.maximum(tl.max(v0_pos, axis=0), tl.max(v1_pos, axis=0))
        new_scale_v = tl.maximum(tl.maximum(max_v, scale_v * 0.95), 1e-12)
        inv_scale_v = 1.0 / new_scale_v

        r0 = tl.maximum(v0 * inv_scale_v, 5.9604645e-8)
        r1 = tl.maximum(v1 * inv_scale_v, 5.9604645e-8)
        nq_v0 = tl.where(
            v0 > 0.0,
            tl.clamp(tl.floor(tl.log2(r0) * (254.0 / 24.0) + (255.5 + noise_v0)), 1.0, 255.0),
            0.0,
        ).to(tl.uint8)
        nq_v1 = tl.where(
            v1 > 0.0,
            tl.clamp(tl.floor(tl.log2(r1) * (254.0 / 24.0) + (255.5 + noise_v1)), 1.0, 255.0),
            0.0,
        ).to(tl.uint8)
        tl.store(v_ptr + even_off, nq_v0, mask=mask_0)
        tl.store(v_ptr + odd_off, nq_v1, mask=mask_1)
        tl.store(v_scales_ptr + pid, new_scale_v)

    @triton.jit
    def _adamw_kernel_triton_8bit_batched(
        p_ptrs,
        g_ptrs,
        m_ptrs,
        ms_ptrs,
        v_ptrs,
        vs_ptrs,
        numels_ptr,
        wd_flags_ptr,
        blk_to_tensor_ptr,
        blk_local_id_ptr,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step_size,
        eps_scaled,
        inv_scale,
        BLOCK_SIZE: tl.constexpr,
        HALF_BLOCK: tl.constexpr,
        P_CODE: tl.constexpr,
        G_CODE: tl.constexpr,
        IS_INT4_M: tl.constexpr,
        cautious: tl.constexpr = False,
        step: int = 1,
        is_unscaled: tl.constexpr = False,
        stochastic: tl.constexpr = True,
        QUAD_BLOCK: tl.constexpr = 1,
        IS_QUAD_V: tl.constexpr = False,
    ):
        """
        Single-launch pointer-array batched 8-bit/INT4/Quad-V AdamW kernel across all homogeneous
        tensors in a bucket, eliminating per-tensor Python kernel launch overhead.
        """
        pid = tl.program_id(axis=0).to(tl.int64)
        t_idx = tl.load(blk_to_tensor_ptr + pid).to(tl.int64)
        local_pid = tl.load(blk_local_id_ptr + pid).to(tl.int64)

        n_elements = tl.load(numels_ptr + t_idx)
        apply_wd_t = tl.load(wd_flags_ptr + t_idx) != 0

        p_addr = tl.load(p_ptrs + t_idx)
        g_addr = tl.load(g_ptrs + t_idx)
        m_addr = tl.load(m_ptrs + t_idx)
        ms_addr = tl.load(ms_ptrs + t_idx)
        v_addr = tl.load(v_ptrs + t_idx)
        vs_addr = tl.load(vs_ptrs + t_idx)

        ms_ptr = ms_addr.to(tl.pointer_type(tl.float32)) + local_pid
        vs_ptr = vs_addr.to(tl.pointer_type(tl.float32)) + local_pid
        scale_m = tl.load(ms_ptr)
        scale_v = tl.load(vs_ptr)
        gain_hi = 1.0 + 0.20 * beta1
        gain_lo = 1.0 - 0.35 * beta1

        if IS_INT4_M:
            if IS_QUAD_V:
                q_lane = tl.arange(0, QUAD_BLOCK)
                quad_off = local_pid * QUAD_BLOCK + q_lane
                mask_q = quad_off < (n_elements >> 2)

                byte0 = local_pid * HALF_BLOCK + q_lane * 2
                byte1 = byte0 + 1
                n_bytes = (n_elements + 1) >> 1
                mask_b0 = byte0 < n_bytes
                mask_b1 = byte1 < n_bytes

                off0 = local_pid * BLOCK_SIZE + q_lane * 4
                off1 = off0 + 1
                off2 = off0 + 2
                off3 = off0 + 3
                mk0 = off0 < n_elements
                mk1 = off1 < n_elements
                mk2 = off2 < n_elements
                mk3 = off3 < n_elements

                m_base = m_addr.to(tl.pointer_type(tl.uint8))
                v_base = v_addr.to(tl.pointer_type(tl.uint8))
                pk0 = tl.load(m_base + byte0, mask=mask_b0, other=0x77).to(tl.int32)
                pk1 = tl.load(m_base + byte1, mask=mask_b1, other=0x77).to(tl.int32)
                m0 = _nf4_dequant_triton((pk0 >> 4) & 0xF) * scale_m
                m1 = _nf4_dequant_triton(pk0 & 0xF) * scale_m
                m2 = _nf4_dequant_triton((pk1 >> 4) & 0xF) * scale_m
                m3 = _nf4_dequant_triton(pk1 & 0xF) * scale_m

                qv = tl.load(v_base + quad_off, mask=mask_q, other=0).to(tl.float32)
                v_q = tl.where(qv > 0.0, tl.exp2((qv - 1.0) * (24.0 / 254.0) - 24.0), 0.0) * scale_v

                p0 = _load_f32_from_int64_ptr(p_addr, P_CODE, off0, mk0, 0.0)
                p1 = _load_f32_from_int64_ptr(p_addr, P_CODE, off1, mk1, 0.0)
                p2 = _load_f32_from_int64_ptr(p_addr, P_CODE, off2, mk2, 0.0)
                p3 = _load_f32_from_int64_ptr(p_addr, P_CODE, off3, mk3, 0.0)
                g0 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, off0, mk0, 0.0))
                g1 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, off1, mk1, 0.0))
                g2 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, off2, mk2, 0.0))
                g3 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, off3, mk3, 0.0))
                if not is_unscaled:
                    if inv_scale != 1.0:
                        g0 = g0 * inv_scale
                        g1 = g1 * inv_scale
                        g2 = g2 * inv_scale
                        g3 = g3 * inv_scale

                if apply_wd_t:
                    wd_factor = tl.maximum(1.0 - lr * weight_decay, 0.0)
                    p0 = p0 * wd_factor
                    p1 = p1 * wd_factor
                    p2 = p2 * wd_factor
                    p3 = p3 * wd_factor

                m0 = m0 * beta1 + g0 * (1.0 - beta1)
                m1 = m1 * beta1 + g1 * (1.0 - beta1)
                m2 = m2 * beta1 + g2 * (1.0 - beta1)
                m3 = m3 * beta1 + g3 * (1.0 - beta1)

                g_sq_mean = 0.25 * (g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)
                v_q = v_q * beta2 + g_sq_mean * (1.0 - beta2)
                v_q = tl.where(tl.abs(v_q) < 3.4028234e38, v_q, 0.0)

                true_eps = tl.where(eps > 1e-6, 1e-8, eps)
                true_eps_scaled = eps_scaled * (true_eps / tl.maximum(eps, 1e-12))
                inv_denom = step_size / tl.maximum(tl.sqrt(tl.maximum(v_q, 0.0)) + true_eps_scaled, true_eps)

                if cautious:
                    u0 = m0 * inv_denom
                    u1 = m1 * inv_denom
                    u2 = m2 * inv_denom
                    u3 = m3 * inv_denom
                    cm0 = tl.where(mk0 & (g0 * m0 >= 0.0), 1.0, 0.0)
                    cm1 = tl.where(mk1 & (g1 * m1 >= 0.0), 1.0, 0.0)
                    cm2 = tl.where(mk2 & (g2 * m2 >= 0.0), 1.0, 0.0)
                    cm3 = tl.where(mk3 & (g3 * m3 >= 0.0), 1.0, 0.0)
                    n_valid = tl.maximum(
                        tl.sum(mk0.to(tl.float32), axis=0)
                        + tl.sum(mk1.to(tl.float32), axis=0)
                        + tl.sum(mk2.to(tl.float32), axis=0)
                        + tl.sum(mk3.to(tl.float32), axis=0),
                        1.0,
                    )
                    c_mean = tl.maximum(
                        (tl.sum(cm0, axis=0) + tl.sum(cm1, axis=0) + tl.sum(cm2, axis=0) + tl.sum(cm3, axis=0)) / n_valid,
                        1e-3,
                    )
                    u0 = u0 * (cm0 / c_mean)
                    u1 = u1 * (cm1 / c_mean)
                    u2 = u2 * (cm2 / c_mean)
                    u3 = u3 * (cm3 / c_mean)
                else:
                    ml0 = m0 * beta1 + g0 * (1.0 - beta1)
                    ml1 = m1 * beta1 + g1 * (1.0 - beta1)
                    ml2 = m2 * beta1 + g2 * (1.0 - beta1)
                    ml3 = m3 * beta1 + g3 * (1.0 - beta1)
                    u0 = (ml0 * tl.where(m0 * g0 >= 0.0, gain_hi, gain_lo)) * inv_denom
                    u1 = (ml1 * tl.where(m1 * g1 >= 0.0, gain_hi, gain_lo)) * inv_denom
                    u2 = (ml2 * tl.where(m2 * g2 >= 0.0, gain_hi, gain_lo)) * inv_denom
                    u3 = (ml3 * tl.where(m3 * g3 >= 0.0, gain_hi, gain_lo)) * inv_denom

                if stochastic:
                    seed0 = _xorshift32_seed(off0, pid ^ t_idx, step)
                    seed1 = _xorshift32_next(seed0)
                    u_m0 = (((seed0 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                    u_m1 = ((seed0 & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                    u_m2 = (((seed1 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                    u_m3 = ((seed1 & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                    seed2 = _xorshift32_next(seed1)
                    noise_v = ((seed2 & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
                    s_p0 = seed0
                    s_p1 = seed0 >> 8
                    s_p2 = seed1
                    s_p3 = seed1 >> 8
                else:
                    u_m0 = 0.0
                    u_m1 = 0.0
                    u_m2 = 0.0
                    u_m3 = 0.0
                    noise_v = 0.0
                    s_p0 = 0
                    s_p1 = 0
                    s_p2 = 0
                    s_p3 = 0

                _store_f32_to_int64_ptr_sr(p_addr, P_CODE, off0, p0 - u0, s_p0, stochastic, mk0)
                _store_f32_to_int64_ptr_sr(p_addr, P_CODE, off1, p1 - u1, s_p1, stochastic, mk1)
                _store_f32_to_int64_ptr_sr(p_addr, P_CODE, off2, p2 - u2, s_p2, stochastic, mk2)
                _store_f32_to_int64_ptr_sr(p_addr, P_CODE, off3, p3 - u3, s_p3, stochastic, mk3)

                max_m = tl.maximum(
                    tl.maximum(tl.max(tl.where(mk0, tl.abs(m0), 0.0), axis=0), tl.max(tl.where(mk1, tl.abs(m1), 0.0), axis=0)),
                    tl.maximum(tl.max(tl.where(mk2, tl.abs(m2), 0.0), axis=0), tl.max(tl.where(mk3, tl.abs(m3), 0.0), axis=0)),
                )
                new_scale_m = tl.maximum(tl.maximum(max_m, scale_m * 0.9), 1e-12)
                inv_scale_m = 1.0 / new_scale_m

                nq0 = _nf4_quant_triton(tl.clamp(m0 * inv_scale_m, -1.0, 1.0), u_m0, stochastic)
                nq1 = tl.where(mk1, _nf4_quant_triton(tl.clamp(m1 * inv_scale_m, -1.0, 1.0), u_m1, stochastic), 7)
                nq2 = _nf4_quant_triton(tl.clamp(m2 * inv_scale_m, -1.0, 1.0), u_m2, stochastic)
                nq3 = tl.where(mk3, _nf4_quant_triton(tl.clamp(m3 * inv_scale_m, -1.0, 1.0), u_m3, stochastic), 7)

                tl.store(m_base + byte0, ((nq0 << 4) | nq1).to(tl.uint8), mask=mask_b0)
                tl.store(m_base + byte1, ((nq2 << 4) | nq3).to(tl.uint8), mask=mask_b1)
                tl.store(ms_ptr, new_scale_m)

                max_v = tl.max(tl.where(mask_q, v_q, 0.0), axis=0)
                new_scale_v = tl.maximum(tl.maximum(max_v, scale_v * 0.95), 1e-12)
                rq = tl.maximum(v_q / new_scale_v, 5.9604645e-8)
                nq_v = tl.where(
                    v_q > 0.0,
                    tl.clamp(tl.floor(tl.log2(rq) * (254.0 / 24.0) + (255.5 + noise_v)), 1.0, 255.0),
                    0.0,
                ).to(tl.uint8)
                tl.store(v_base + quad_off, nq_v, mask=mask_q)
                tl.store(vs_ptr, new_scale_v)
                return

            lane = tl.arange(0, HALF_BLOCK)
            byte_offsets = local_pid * HALF_BLOCK + lane
            mask_byte = byte_offsets < ((n_elements + 1) >> 1)
            even_off = local_pid * BLOCK_SIZE + lane * 2
            odd_off = even_off + 1
            mask_0 = even_off < n_elements
            mask_1 = odd_off < n_elements

            m_base = m_addr.to(tl.pointer_type(tl.uint8))
            v_base = v_addr.to(tl.pointer_type(tl.uint8))
            packed_m = tl.load(m_base + byte_offsets, mask=mask_byte, other=0x77).to(tl.int32)
            q_m0 = (packed_m >> 4) & 0xF
            q_m1 = packed_m & 0xF

            p0 = _load_f32_from_int64_ptr(p_addr, P_CODE, even_off, mask_0, 0.0)
            p1 = _load_f32_from_int64_ptr(p_addr, P_CODE, odd_off, mask_1, 0.0)
            g0 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, even_off, mask_0, 0.0))
            g1 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, odd_off, mask_1, 0.0))
            if not is_unscaled:
                if inv_scale != 1.0:
                    g0 = g0 * inv_scale
                    g1 = g1 * inv_scale

            qv0 = tl.load(v_base + even_off, mask=mask_0, other=0).to(tl.float32)
            qv1 = tl.load(v_base + odd_off, mask=mask_1, other=0).to(tl.float32)

            m0 = _nf4_dequant_triton(q_m0) * scale_m
            m1 = _nf4_dequant_triton(q_m1) * scale_m
            v0 = tl.where(qv0 > 0.0, tl.exp2((qv0 - 1.0) * (24.0 / 254.0) - 24.0), 0.0) * scale_v
            v1 = tl.where(qv1 > 0.0, tl.exp2((qv1 - 1.0) * (24.0 / 254.0) - 24.0), 0.0) * scale_v

            if apply_wd_t:
                wd_factor = tl.maximum(1.0 - lr * weight_decay, 0.0)
                p0 = p0 * wd_factor
                p1 = p1 * wd_factor

            m0 = m0 * beta1 + g0 * (1.0 - beta1)
            m1 = m1 * beta1 + g1 * (1.0 - beta1)
            v0 = tl.where(tl.abs(v0 * beta2 + (g0 * g0) * (1.0 - beta2)) < 3.4028234e38, v0 * beta2 + (g0 * g0) * (1.0 - beta2), 0.0)
            v1 = tl.where(tl.abs(v1 * beta2 + (g1 * g1) * (1.0 - beta2)) < 3.4028234e38, v1 * beta2 + (g1 * g1) * (1.0 - beta2), 0.0)

            denom0 = tl.maximum(tl.sqrt(v0) + eps_scaled, eps)
            denom1 = tl.maximum(tl.sqrt(v1) + eps_scaled, eps)
            if cautious:
                u0 = (m0 / denom0) * step_size
                u1 = (m1 / denom1) * step_size
                cm0 = tl.where(mask_0 & (g0 * m0 >= 0.0), 1.0, 0.0)
                cm1 = tl.where(mask_1 & (g1 * m1 >= 0.0), 1.0, 0.0)
                n_valid = tl.maximum(
                    tl.sum(mask_0.to(tl.float32), axis=0) + tl.sum(mask_1.to(tl.float32), axis=0),
                    1.0,
                )
                c_mean = tl.maximum((tl.sum(cm0, axis=0) + tl.sum(cm1, axis=0)) / n_valid, 1e-3)
                u0 = u0 * (cm0 / c_mean)
                u1 = u1 * (cm1 / c_mean)
            else:
                ml0 = m0 * beta1 + g0 * (1.0 - beta1)
                ml1 = m1 * beta1 + g1 * (1.0 - beta1)
                u0 = ((ml0 * tl.where(m0 * g0 >= 0.0, gain_hi, gain_lo)) / denom0) * step_size
                u1 = ((ml1 * tl.where(m1 * g1 >= 0.0, gain_hi, gain_lo)) / denom1) * step_size

            _store_f32_to_int64_ptr(p_addr, P_CODE, even_off, p0 - u0, mask_0)
            _store_f32_to_int64_ptr(p_addr, P_CODE, odd_off, p1 - u1, mask_1)

            max_m = tl.maximum(
                tl.max(tl.where(mask_0, tl.abs(m0), 0.0), axis=0),
                tl.max(tl.where(mask_1, tl.abs(m1), 0.0), axis=0),
            )
            new_scale_m = tl.maximum(tl.maximum(max_m, scale_m * 0.9), 1e-12)
            inv_scale_m = 1.0 / new_scale_m
            norm0 = tl.clamp(m0 * inv_scale_m, -1.0, 1.0)
            norm1 = tl.clamp(m1 * inv_scale_m, -1.0, 1.0)

            if stochastic:
                seed0 = _xorshift32_seed(even_off, pid ^ t_idx, step)
                seed1 = _xorshift32_next(seed0)
                u_m0 = (((seed0 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                u_m1 = (((seed1 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0))
                noise_v0 = ((seed0 & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
                noise_v1 = ((seed1 & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            else:
                u_m0 = 0.0
                u_m1 = 0.0
                noise_v0 = 0.0
                noise_v1 = 0.0

            nq_m0 = _nf4_quant_triton(norm0, u_m0, stochastic)
            nq_m1 = tl.where(mask_1, _nf4_quant_triton(norm1, u_m1, stochastic), 7)
            tl.store(m_base + byte_offsets, ((nq_m0 << 4) | nq_m1).to(tl.uint8), mask=mask_byte)
            tl.store(ms_ptr, new_scale_m)

            max_v = tl.maximum(
                tl.max(tl.where(mask_0, v0, 0.0), axis=0),
                tl.max(tl.where(mask_1, v1, 0.0), axis=0),
            )
            new_scale_v = tl.maximum(tl.maximum(max_v, scale_v * 0.95), 1e-12)
            inv_scale_v = 1.0 / new_scale_v
            r0 = tl.maximum(v0 * inv_scale_v, 5.9604645e-8)
            r1 = tl.maximum(v1 * inv_scale_v, 5.9604645e-8)
            nq_v0 = tl.where(
                v0 > 0.0,
                tl.clamp(tl.floor(tl.log2(r0) * (254.0 / 24.0) + (255.5 + noise_v0)), 1.0, 255.0),
                0.0,
            ).to(tl.uint8)
            nq_v1 = tl.where(
                v1 > 0.0,
                tl.clamp(tl.floor(tl.log2(r1) * (254.0 / 24.0) + (255.5 + noise_v1)), 1.0, 255.0),
                0.0,
            ).to(tl.uint8)
            tl.store(v_base + even_off, nq_v0, mask=mask_0)
            tl.store(v_base + odd_off, nq_v1, mask=mask_1)
            tl.store(vs_ptr, new_scale_v)
        else:
            offsets = local_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            m_base = m_addr.to(tl.pointer_type(tl.int8))
            v_base = v_addr.to(tl.pointer_type(tl.uint8))

            p = _load_f32_from_int64_ptr(p_addr, P_CODE, offsets, mask, 0.0)
            g = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, offsets, mask, 0.0))
            if not is_unscaled:
                if inv_scale != 1.0:
                    g = g * inv_scale

            q_m = tl.load(m_base + offsets, mask=mask, other=0).to(tl.float32)
            q_v = tl.load(v_base + offsets, mask=mask, other=0).to(tl.float32)
            m = q_m * scale_m
            v = tl.where(q_v > 0.0, tl.exp2((q_v - 1.0) * (24.0 / 254.0) - 24.0), 0.0) * scale_v

            if apply_wd_t:
                p = p * tl.maximum(1.0 - lr * weight_decay, 0.0)

            m = m * beta1 + g * (1.0 - beta1)
            v = v * beta2 + (g * g) * (1.0 - beta2)
            v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)

            denom = tl.maximum(tl.sqrt(v) + eps_scaled, eps)
            if cautious:
                update = (m / denom) * step_size
                c_mask = (g * m >= 0.0).to(tl.float32)
                valid = mask.to(tl.float32)
                n_valid = tl.maximum(tl.sum(valid, axis=0), 1.0)
                c_mean = tl.maximum(tl.sum(c_mask * valid, axis=0) / n_valid, 1e-3)
                update = update * (c_mask / c_mean)
            else:
                m_look = m * beta1 + g * (1.0 - beta1)
                gain = tl.where(m * g >= 0.0, gain_hi, gain_lo)
                update = ((m_look * gain) / denom) * step_size

            _store_f32_to_int64_ptr(p_addr, P_CODE, offsets, p - update, mask)

            max_m = tl.max(tl.where(mask, tl.abs(m), 0.0), axis=0)
            new_scale_m = tl.maximum(tl.maximum(max_m / 127.0, scale_m * 0.9), 1e-12)
            scaled_m = m / new_scale_m

            if stochastic:
                seed = _xorshift32_seed(offsets, pid ^ t_idx, step)
                noise = (((seed >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
                noise_v = ((seed & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
                rounded_m = tl.floor(scaled_m + noise + 0.5)
            else:
                noise_v = 0.0
                rounded_m = tl.where(scaled_m >= 0.0, tl.floor(scaled_m + 0.5), tl.ceil(scaled_m - 0.5))

            tl.store(m_base + offsets, tl.clamp(rounded_m, -127.0, 127.0).to(tl.int8), mask=mask)
            tl.store(ms_ptr, new_scale_m)

            max_v = tl.max(tl.where(mask, v, 0.0), axis=0)
            new_scale_v = tl.maximum(tl.maximum(max_v, scale_v * 0.95), 1e-12)
            r_new = tl.maximum(v / new_scale_v, 5.9604645e-8)
            q_v_calc = tl.where(
                v > 0.0,
                tl.clamp(tl.floor(tl.log2(r_new) * (254.0 / 24.0) + (255.5 + noise_v)), 1.0, 255.0),
                0.0,
            )
            tl.store(v_base + offsets, q_v_calc.to(tl.uint8), mask=mask)
            tl.store(vs_ptr, new_scale_v)

    @triton.jit
    def _adamw_kernel_triton_compact_batched(
        p_ptrs,
        g_ptrs,
        m_ptrs,
        v_ptrs,
        numels_ptr,
        wd_flags_ptr,
        scale_v_ptr,
        sqrt_scale_v_ptr,
        blk_to_tensor_ptr,
        blk_local_id_ptr,
        lr,
        beta1,
        beta2,
        weight_decay,
        step_size,
        eps_scaled,
        inv_scale,
        BLOCK_SIZE: tl.constexpr,
        P_CODE: tl.constexpr,
        G_CODE: tl.constexpr,
        M_CODE: tl.constexpr,
        V_CODE: tl.constexpr,
        has_scale_v: tl.constexpr = False,
        cautious: tl.constexpr = False,
        QUAD_BLOCK: tl.constexpr = 1,
        IS_QUAD_V: tl.constexpr = False,
    ):
        """
        Single-launch pointer-array batched kernel for Compact/Cross-precision/FP32 states,
        accelerating precision='compact' when custom CUDA C++ isn't used (e.g. split m/v dtypes or no nvcc).
        """
        pid = tl.program_id(axis=0).to(tl.int64)
        t_idx = tl.load(blk_to_tensor_ptr + pid).to(tl.int64)
        local_pid = tl.load(blk_local_id_ptr + pid).to(tl.int64)

        n_elements = tl.load(numels_ptr + t_idx)
        apply_wd_t = tl.load(wd_flags_ptr + t_idx) != 0
        p_addr = tl.load(p_ptrs + t_idx)
        g_addr = tl.load(g_ptrs + t_idx)
        m_addr = tl.load(m_ptrs + t_idx)
        v_addr = tl.load(v_ptrs + t_idx)
        gain_hi = 1.0 + 0.20 * beta1
        gain_lo = 1.0 - 0.35 * beta1

        if IS_QUAD_V:
            q_lane = tl.arange(0, QUAD_BLOCK)
            quad_off = local_pid * QUAD_BLOCK + q_lane
            mask_q = quad_off < (n_elements >> 2)
            off0 = local_pid * BLOCK_SIZE + q_lane * 4
            off1 = off0 + 1
            off2 = off0 + 2
            off3 = off0 + 3
            mk0 = off0 < n_elements
            mk1 = off1 < n_elements
            mk2 = off2 < n_elements
            mk3 = off3 < n_elements

            p0 = _load_f32_from_int64_ptr(p_addr, P_CODE, off0, mk0, 0.0)
            p1 = _load_f32_from_int64_ptr(p_addr, P_CODE, off1, mk1, 0.0)
            p2 = _load_f32_from_int64_ptr(p_addr, P_CODE, off2, mk2, 0.0)
            p3 = _load_f32_from_int64_ptr(p_addr, P_CODE, off3, mk3, 0.0)
            g0 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, off0, mk0, 0.0))
            g1 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, off1, mk1, 0.0))
            g2 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, off2, mk2, 0.0))
            g3 = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, off3, mk3, 0.0))
            if inv_scale != 1.0:
                g0 = g0 * inv_scale
                g1 = g1 * inv_scale
                g2 = g2 * inv_scale
                g3 = g3 * inv_scale

            m0 = _load_f32_from_int64_ptr(m_addr, M_CODE, off0, mk0, 0.0)
            m1 = _load_f32_from_int64_ptr(m_addr, M_CODE, off1, mk1, 0.0)
            m2 = _load_f32_from_int64_ptr(m_addr, M_CODE, off2, mk2, 0.0)
            m3 = _load_f32_from_int64_ptr(m_addr, M_CODE, off3, mk3, 0.0)
            v_q = _load_f32_from_int64_ptr(v_addr, V_CODE, quad_off, mask_q, 0.0)

            if apply_wd_t:
                wd_f = tl.maximum(1.0 - lr * weight_decay, 0.0)
                p0 = p0 * wd_f
                p1 = p1 * wd_f
                p2 = p2 * wd_f
                p3 = p3 * wd_f

            m0 = m0 * beta1 + g0 * (1.0 - beta1)
            m1 = m1 * beta1 + g1 * (1.0 - beta1)
            m2 = m2 * beta1 + g2 * (1.0 - beta1)
            m3 = m3 * beta1 + g3 * (1.0 - beta1)
            g_sq_mean = 0.25 * (g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)

            if has_scale_v:
                sv = tl.load(scale_v_ptr + t_idx)
                ssv = tl.load(sqrt_scale_v_ptr + t_idx)
                v_q = v_q * beta2 + g_sq_mean * ((1.0 - beta2) * sv)
                v_q = tl.where(tl.abs(v_q) < 3.4028234e38, v_q, 0.0)
                inv_denom = (step_size * ssv) / tl.maximum(tl.sqrt(tl.maximum(v_q, 0.0)) + (eps_scaled * ssv), tl.maximum(eps_scaled * ssv, 1e-30))
            else:
                v_q = v_q * beta2 + g_sq_mean * (1.0 - beta2)
                v_q = tl.where(tl.abs(v_q) < 3.4028234e38, v_q, 0.0)
                inv_denom = step_size / tl.maximum(tl.sqrt(tl.maximum(v_q, 0.0)) + eps_scaled, tl.maximum(eps_scaled, 1e-30))

            if cautious:
                u0 = m0 * inv_denom
                u1 = m1 * inv_denom
                u2 = m2 * inv_denom
                u3 = m3 * inv_denom
                cm0 = tl.where(mk0 & (g0 * m0 >= 0.0), 1.0, 0.0)
                cm1 = tl.where(mk1 & (g1 * m1 >= 0.0), 1.0, 0.0)
                cm2 = tl.where(mk2 & (g2 * m2 >= 0.0), 1.0, 0.0)
                cm3 = tl.where(mk3 & (g3 * m3 >= 0.0), 1.0, 0.0)
                n_valid = tl.maximum(
                    tl.sum(mk0.to(tl.float32), axis=0)
                    + tl.sum(mk1.to(tl.float32), axis=0)
                    + tl.sum(mk2.to(tl.float32), axis=0)
                    + tl.sum(mk3.to(tl.float32), axis=0),
                    1.0,
                )
                c_mean = tl.maximum(
                    (tl.sum(cm0, axis=0) + tl.sum(cm1, axis=0) + tl.sum(cm2, axis=0) + tl.sum(cm3, axis=0)) / n_valid,
                    1e-3,
                )
                u0 = u0 * (cm0 / c_mean)
                u1 = u1 * (cm1 / c_mean)
                u2 = u2 * (cm2 / c_mean)
                u3 = u3 * (cm3 / c_mean)
            else:
                ml0 = m0 * beta1 + g0 * (1.0 - beta1)
                ml1 = m1 * beta1 + g1 * (1.0 - beta1)
                ml2 = m2 * beta1 + g2 * (1.0 - beta1)
                ml3 = m3 * beta1 + g3 * (1.0 - beta1)
                u0 = (ml0 * tl.where(m0 * g0 >= 0.0, gain_hi, gain_lo)) * inv_denom
                u1 = (ml1 * tl.where(m1 * g1 >= 0.0, gain_hi, gain_lo)) * inv_denom
                u2 = (ml2 * tl.where(m2 * g2 >= 0.0, gain_hi, gain_lo)) * inv_denom
                u3 = (ml3 * tl.where(m3 * g3 >= 0.0, gain_hi, gain_lo)) * inv_denom

            _store_f32_to_int64_ptr(p_addr, P_CODE, off0, p0 - u0, mk0)
            _store_f32_to_int64_ptr(p_addr, P_CODE, off1, p1 - u1, mk1)
            _store_f32_to_int64_ptr(p_addr, P_CODE, off2, p2 - u2, mk2)
            _store_f32_to_int64_ptr(p_addr, P_CODE, off3, p3 - u3, mk3)
            _store_f32_to_int64_ptr(m_addr, M_CODE, off0, m0, mk0)
            _store_f32_to_int64_ptr(m_addr, M_CODE, off1, m1, mk1)
            _store_f32_to_int64_ptr(m_addr, M_CODE, off2, m2, mk2)
            _store_f32_to_int64_ptr(m_addr, M_CODE, off3, m3, mk3)
            _store_f32_to_int64_ptr(v_addr, V_CODE, quad_off, v_q, mask_q)
            return

        offsets = local_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = _load_f32_from_int64_ptr(p_addr, P_CODE, offsets, mask, 0.0)
        g = _sanitize_grad_triton(_load_f32_from_int64_ptr(g_addr, G_CODE, offsets, mask, 0.0))
        if inv_scale != 1.0:
            g = g * inv_scale
        m = _load_f32_from_int64_ptr(m_addr, M_CODE, offsets, mask, 0.0)
        v = _load_f32_from_int64_ptr(v_addr, V_CODE, offsets, mask, 0.0)

        if apply_wd_t:
            p = p * tl.maximum(1.0 - lr * weight_decay, 0.0)

        m = m * beta1 + g * (1.0 - beta1)
        if has_scale_v:
            sv = tl.load(scale_v_ptr + t_idx)
            ssv = tl.load(sqrt_scale_v_ptr + t_idx)
            v = v * beta2 + (g * g) * ((1.0 - beta2) * sv)
            v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)
            denom = tl.maximum(tl.sqrt(tl.maximum(v, 0.0)) + (eps_scaled * ssv), tl.maximum(eps_scaled * ssv, 1e-30))
            step_factor = step_size * ssv
        else:
            v = v * beta2 + (g * g) * (1.0 - beta2)
            v = tl.where(tl.abs(v) < 3.4028234e38, v, 0.0)
            denom = tl.maximum(tl.sqrt(tl.maximum(v, 0.0)) + eps_scaled, tl.maximum(eps_scaled, 1e-30))
            step_factor = step_size

        update = (m / denom) * step_factor
        if cautious:
            c_mask = (g * m >= 0.0).to(tl.float32)
            valid = mask.to(tl.float32)
            n_valid = tl.maximum(tl.sum(valid, axis=0), 1.0)
            c_mean = tl.maximum(tl.sum(c_mask * valid, axis=0) / n_valid, 1e-3)
            update = update * (c_mask / c_mean)

        _store_f32_to_int64_ptr(p_addr, P_CODE, offsets, p - update, mask)
        _store_f32_to_int64_ptr(m_addr, M_CODE, offsets, m, mask)
        _store_f32_to_int64_ptr(v_addr, V_CODE, offsets, v, mask)

    @triton.jit
    def _adamw_kernel_triton_factored_v(
        p_ptr,
        g_ptr,
        m_ptr,
        m_scales_ptr,
        row_norm_ptr,
        col_sqrt_ptr,
        n_elements,
        cols,
        lr,
        beta1,
        eps,
        weight_decay,
        step_size,
        eps_scaled,
        inv_scale,
        apply_wd: tl.constexpr,
        IS_INT8_M: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        cautious: tl.constexpr = False,
        step: int = 1,
        is_unscaled: tl.constexpr = True,
        stochastic: tl.constexpr = True,
    ):
        """Fused factored-v AdamW kernel supporting Compact (BF16/FP16/FP32) and INT8 m in SRAM."""
        pid = tl.program_id(axis=0).to(tl.int64)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = _sanitize_grad_triton(tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        if not is_unscaled:
            g = g * inv_scale

        if IS_INT8_M:
            scale_m = tl.load(m_scales_ptr + pid)
            m = tl.load(m_ptr + offsets, mask=mask, other=0).to(tl.float32) * scale_m
        else:
            m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        r_idx = offsets // cols
        c_idx = offsets - r_idx * cols
        rn = tl.load(row_norm_ptr + r_idx, mask=mask, other=1.0).to(tl.float32)
        cs = tl.load(col_sqrt_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
        denom = tl.maximum(rn * cs + eps_scaled, eps)

        if apply_wd:
            p = p * tl.maximum(1.0 - lr * weight_decay, 0.0)

        m = m * beta1 + g * (1.0 - beta1)
        num = m
        if cautious:
            c_mask = (g * num >= 0.0).to(tl.float32)
            valid = mask.to(tl.float32)
            n_valid = tl.maximum(tl.sum(valid, axis=0), 1.0)
            c_mean = tl.maximum(tl.sum(c_mask * valid, axis=0) / n_valid, 1e-3)
            num = num * (c_mask / c_mean)

        p = p - (num / denom) * step_size
        _store_typed_clamped(p_ptr, offsets, p, mask)

        if IS_INT8_M:
            m_abs = tl.where(mask, tl.abs(m), 0.0)
            max_m = tl.max(m_abs, axis=0)
            new_scale_m = tl.maximum(tl.maximum(max_m / 127.0, scale_m * 0.9), 1e-12)
            scaled_m = m / new_scale_m
            if stochastic:
                seed = _xorshift32_seed(offsets, pid, step)
                noise = (((seed >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
                rounded_m = tl.floor(scaled_m + noise + 0.5)
            else:
                rounded_m = tl.where(scaled_m >= 0.0, tl.floor(scaled_m + 0.5), tl.ceil(scaled_m - 0.5))
            tl.store(m_ptr + offsets, tl.clamp(rounded_m, -127.0, 127.0).to(tl.int8), mask=mask)
            tl.store(m_scales_ptr + pid, new_scale_m)
        else:
            _store_typed_clamped(m_ptr, offsets, m, mask)

    @triton.jit
    def _adamw_kernel_triton_factored_v_6bit(
        p_ptr,
        g_ptr,
        m_ptr,
        m_scales_ptr,
        row_norm_ptr,
        col_sqrt_ptr,
        n_elements,
        n_quads,
        cols,
        lr,
        beta1,
        eps,
        weight_decay,
        step_size,
        eps_scaled,
        inv_scale,
        apply_wd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        QUADS: tl.constexpr,
        cautious: tl.constexpr = False,
        step: int = 1,
        is_unscaled: tl.constexpr = True,
        stochastic: tl.constexpr = True,
    ):
        """Fused factored-v AdamW kernel with 6-bit packed m (4 values per 3 uint8 bytes)."""
        pid = tl.program_id(axis=0).to(tl.int64)
        q_offsets = pid * QUADS + tl.arange(0, QUADS)
        q_mask = q_offsets < n_quads
        scale_m = tl.load(m_scales_ptr + pid)

        b_base = q_offsets * 3
        b0 = tl.load(m_ptr + b_base, mask=q_mask, other=223).to(tl.int32)
        b1 = tl.load(m_ptr + b_base + 1, mask=q_mask, other=247).to(tl.int32)
        b2 = tl.load(m_ptr + b_base + 2, mask=q_mask, other=125).to(tl.int32)

        u0 = b0 & 0x3F
        u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
        u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
        u3 = (b2 >> 2) & 0x3F

        off0 = q_offsets * 4
        off1 = off0 + 1
        off2 = off0 + 2
        off3 = off0 + 3
        mk0 = off0 < n_elements
        mk1 = off1 < n_elements
        mk2 = off2 < n_elements
        mk3 = off3 < n_elements

        wd_f = tl.maximum(1.0 - lr * weight_decay, 0.0)
        one_minus_b1 = 1.0 - beta1

        # Lane 0
        p0 = tl.load(p_ptr + off0, mask=mk0, other=0.0).to(tl.float32)
        g0 = _sanitize_grad_triton(tl.load(g_ptr + off0, mask=mk0, other=0.0).to(tl.float32))
        if not is_unscaled:
            g0 = g0 * inv_scale
        r0 = off0 // cols
        c0 = off0 - r0 * cols
        d0 = tl.maximum(
            tl.load(row_norm_ptr + r0, mask=mk0, other=1.0).to(tl.float32)
            * tl.load(col_sqrt_ptr + c0, mask=mk0, other=0.0).to(tl.float32)
            + eps_scaled,
            eps,
        )
        m0 = (u0.to(tl.float32) - 31.0) * scale_m
        m0 = m0 * beta1 + g0 * one_minus_b1
        if apply_wd:
            p0 = p0 * wd_f
        _store_typed_clamped(p_ptr, off0, p0 - (m0 / d0) * step_size, mk0)

        # Lane 1
        p1 = tl.load(p_ptr + off1, mask=mk1, other=0.0).to(tl.float32)
        g1 = _sanitize_grad_triton(tl.load(g_ptr + off1, mask=mk1, other=0.0).to(tl.float32))
        if not is_unscaled:
            g1 = g1 * inv_scale
        r1 = off1 // cols
        c1 = off1 - r1 * cols
        d1 = tl.maximum(
            tl.load(row_norm_ptr + r1, mask=mk1, other=1.0).to(tl.float32)
            * tl.load(col_sqrt_ptr + c1, mask=mk1, other=0.0).to(tl.float32)
            + eps_scaled,
            eps,
        )
        m1 = (u1.to(tl.float32) - 31.0) * scale_m
        m1 = m1 * beta1 + g1 * one_minus_b1
        if apply_wd:
            p1 = p1 * wd_f
        _store_typed_clamped(p_ptr, off1, p1 - (m1 / d1) * step_size, mk1)

        # Lane 2
        p2 = tl.load(p_ptr + off2, mask=mk2, other=0.0).to(tl.float32)
        g2 = _sanitize_grad_triton(tl.load(g_ptr + off2, mask=mk2, other=0.0).to(tl.float32))
        if not is_unscaled:
            g2 = g2 * inv_scale
        r2 = off2 // cols
        c2 = off2 - r2 * cols
        d2 = tl.maximum(
            tl.load(row_norm_ptr + r2, mask=mk2, other=1.0).to(tl.float32)
            * tl.load(col_sqrt_ptr + c2, mask=mk2, other=0.0).to(tl.float32)
            + eps_scaled,
            eps,
        )
        m2 = (u2.to(tl.float32) - 31.0) * scale_m
        m2 = m2 * beta1 + g2 * one_minus_b1
        if apply_wd:
            p2 = p2 * wd_f
        _store_typed_clamped(p_ptr, off2, p2 - (m2 / d2) * step_size, mk2)

        # Lane 3
        p3 = tl.load(p_ptr + off3, mask=mk3, other=0.0).to(tl.float32)
        g3 = _sanitize_grad_triton(tl.load(g_ptr + off3, mask=mk3, other=0.0).to(tl.float32))
        if not is_unscaled:
            g3 = g3 * inv_scale
        r3 = off3 // cols
        c3 = off3 - r3 * cols
        d3 = tl.maximum(
            tl.load(row_norm_ptr + r3, mask=mk3, other=1.0).to(tl.float32)
            * tl.load(col_sqrt_ptr + c3, mask=mk3, other=0.0).to(tl.float32)
            + eps_scaled,
            eps,
        )
        m3 = (u3.to(tl.float32) - 31.0) * scale_m
        m3 = m3 * beta1 + g3 * one_minus_b1
        if apply_wd:
            p3 = p3 * wd_f
        _store_typed_clamped(p_ptr, off3, p3 - (m3 / d3) * step_size, mk3)

        # Shared per-block abs max & 6-bit requantization
        ax0 = tl.where(mk0, tl.abs(m0), 0.0)
        ax1 = tl.where(mk1, tl.abs(m1), 0.0)
        ax2 = tl.where(mk2, tl.abs(m2), 0.0)
        ax3 = tl.where(mk3, tl.abs(m3), 0.0)
        blk_max = tl.max(tl.maximum(tl.maximum(ax0, ax1), tl.maximum(ax2, ax3)), axis=0)
        new_scale = tl.maximum(tl.maximum(blk_max / 31.0, scale_m * 0.9), 1e-12)
        inv_s = 1.0 / new_scale

        s0 = m0 * inv_s
        s1 = m1 * inv_s
        s2 = m2 * inv_s
        s3 = m3 * inv_s
        if stochastic:
            sd0 = _xorshift32_seed(off0, pid, step)
            sd1 = _xorshift32_next(sd0)
            n0 = (((sd0 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            n1 = ((sd0 & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            n2 = (((sd1 >> 16) & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            n3 = ((sd1 & 0xffff).to(tl.float32) * (1.0 / 65536.0)) - 0.5
            s0 = tl.floor(s0 + n0 + 0.5)
            s1 = tl.floor(s1 + n1 + 0.5)
            s2 = tl.floor(s2 + n2 + 0.5)
            s3 = tl.floor(s3 + n3 + 0.5)
        else:
            s0 = tl.where(s0 >= 0.0, tl.floor(s0 + 0.5), tl.ceil(s0 - 0.5))
            s1 = tl.where(s1 >= 0.0, tl.floor(s1 + 0.5), tl.ceil(s1 - 0.5))
            s2 = tl.where(s2 >= 0.0, tl.floor(s2 + 0.5), tl.ceil(s2 - 0.5))
            s3 = tl.where(s3 >= 0.0, tl.floor(s3 + 0.5), tl.ceil(s3 - 0.5))

        q0 = (tl.clamp(s0, -31.0, 31.0) + 31.0).to(tl.int32)
        q1 = (tl.clamp(s1, -31.0, 31.0) + 31.0).to(tl.int32)
        q2 = (tl.clamp(s2, -31.0, 31.0) + 31.0).to(tl.int32)
        q3 = (tl.clamp(s3, -31.0, 31.0) + 31.0).to(tl.int32)

        nb0 = ((q0 & 0x3F) | ((q1 & 0x03) << 6)).to(tl.uint8)
        nb1 = (((q1 >> 2) & 0x0F) | ((q2 & 0x0F) << 4)).to(tl.uint8)
        nb2 = (((q2 >> 4) & 0x03) | ((q3 & 0x3F) << 2)).to(tl.uint8)

        tl.store(m_ptr + b_base, nb0, mask=q_mask)
        tl.store(m_ptr + b_base + 1, nb1, mask=q_mask)
        tl.store(m_ptr + b_base + 2, nb2, mask=q_mask)
        tl.store(m_scales_ptr + pid, new_scale)
else:
    _adamw_kernel_triton = None
    _adamw_kernel_triton_fp32 = None
    _adamw_kernel_triton_cross_precision = None
    _adamw_kernel_triton_8bit = None
    _adamw_kernel_triton_8bit_m_16bit_v = None
    _adamw_kernel_triton_8bit_int4 = None
    _adamw_kernel_triton_8bit_batched = None
    _adamw_kernel_triton_compact_batched = None
    _adamw_kernel_triton_factored_v = None
    _adamw_kernel_triton_factored_v_6bit = None


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
    exp_avg_scales: Optional[torch.Tensor] = None,
    exp_avg_sq_scales: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
) -> Optional[str]:
    """
    Validates tensor layout, device, and dtype before any Triton kernel is launched.
    Returns None if all checks pass, or a human-readable reason string if not supported.
    Pre-flight validation ensures unsupported configurations safely return False before
    any tensor data is modified in-place.
    """
    _SUPPORTED_PARAM_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
    _SUPPORTED_M_DTYPES = {torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8}
    _SUPPORTED_V_DTYPES = {torch.float32, torch.float16, torch.bfloat16, torch.uint8}
    _SUPPORTED_FLOAT_DTYPES = {torch.float32, torch.float16, torch.bfloat16}

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
    if exp_avg.dtype not in _SUPPORTED_M_DTYPES:
        return f"exp_avg dtype {exp_avg.dtype} not supported by Triton kernel"
    if exp_avg_sq.dtype not in _SUPPORTED_V_DTYPES:
        return f"exp_avg_sq dtype {exp_avg_sq.dtype} not supported by Triton kernel"
    if exp_avg.dtype == torch.uint8:
        if exp_avg_scales is None:
            return "NF4 INT4 exp_avg (uint8) requires exp_avg_scales for Triton kernel"
        if exp_avg_slow is not None:
            return "AdEMAMix slow momentum is not supported with INT4 exp_avg in Triton kernel"
        if exp_avg_sq.dtype != torch.uint8 or exp_avg_sq_scales is None:
            return "NF4 INT4 exp_avg (uint8) requires Log-UINT8 exp_avg_sq with scales for Triton kernel"
        expected_packed = (param.numel() + 1) // 2
        if exp_avg.numel() != expected_packed:
            return f"NF4 INT4 exp_avg (uint8) element count mismatch ({exp_avg.numel()} vs expected {expected_packed})"
        is_quad_v = (param.numel() >= 4 and param.numel() % 4 == 0 and exp_avg_sq.numel() * 4 == param.numel())
        if exp_avg_sq.numel() != param.numel() and not is_quad_v:
            return f"NF4 INT4 exp_avg_sq (uint8) element count mismatch ({exp_avg_sq.numel()})"
    if exp_avg.dtype == torch.int8:
        if exp_avg_scales is None:
            return "exp_avg INT8 requires exp_avg_scales for Triton kernel"
        if exp_avg_slow is not None:
            return "AdEMAMix slow momentum is not supported with INT8 exp_avg in Triton kernel"
    if exp_avg_sq.dtype == torch.uint8 and exp_avg_sq_scales is None:
        return "exp_avg_sq UINT8 requires exp_avg_sq_scales for Triton kernel"
    if exp_avg_slow is not None and exp_avg_slow.dtype not in _SUPPORTED_FLOAT_DTYPES:
        return f"exp_avg_slow dtype {exp_avg_slow.dtype} not supported by Triton kernel"
    if block_size is not None:
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0 or (block_size & (block_size - 1)) != 0:
            return f"block_size must be a positive power of 2 for Triton kernel (got {block_size})"
        if exp_avg.dtype == torch.uint8 and block_size < 2:
            return f"block_size must be >= 2 for NF4 packed INT4 exp_avg (got {block_size})"
    return None  # all checks passed


def _resolve_block_size(
    n_elements: int,
    m_dtype: torch.dtype,
    v_dtype: torch.dtype,
    block_size: Optional[int],
) -> Optional[int]:
    if block_size is None:
        if m_dtype in (torch.int8, torch.uint8) and v_dtype == torch.uint8:
            return 2048 if n_elements >= 2048 else (1024 if n_elements >= 1024 else (512 if n_elements >= 512 else (256 if n_elements >= 256 else (128 if n_elements >= 128 else 64))))
        if m_dtype == torch.int8 and v_dtype in (torch.bfloat16, torch.float16, torch.float32):
            return 2048 if n_elements >= 2048 else (256 if n_elements >= 256 else 128)
        return 1024
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0 or (block_size & (block_size - 1)) != 0:
        return None
    if m_dtype == torch.uint8 and block_size < 2:
        return None
    return block_size


def _is_valid_quant_combo(
    param: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    exp_avg_slow: Optional[torch.Tensor],
    exp_avg_scales: Optional[torch.Tensor],
    exp_avg_sq_scales: Optional[torch.Tensor],
) -> bool:
    if exp_avg_sq.dtype == torch.int8:
        return False
    if exp_avg.dtype == torch.uint8:
        is_quad_v = (param.numel() >= 4 and param.numel() % 4 == 0 and exp_avg_sq.numel() * 4 == param.numel())
        if (
            exp_avg_scales is None
            or exp_avg_slow is not None
            or exp_avg_sq.dtype != torch.uint8
            or exp_avg_sq_scales is None
            or exp_avg.numel() != (param.numel() + 1) // 2
            or (exp_avg_sq.numel() != param.numel() and not is_quad_v)
        ):
            return False
    if exp_avg.dtype == torch.int8 and (exp_avg_scales is None or exp_avg_slow is not None):
        return False
    if exp_avg_sq.dtype == torch.uint8 and exp_avg_sq_scales is None:
        return False
    return True


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
    stochastic: bool = True,
) -> bool:
    """
    Executes the Triton fused AdamW kernel on a single tensor.

    Supports pure FP32, cross-precision 16-bit, fused 8-bit (INT8 m + UINT8 v),
    1.5-byte NF4 (INT4 packed m + UINT8 v), and hybrid 3-byte (INT8 m + 16-bit v) paths.
    """
    if not is_triton_available():
        return False

    n_elements = param.numel()
    if n_elements == 0:
        return True

    # Pre-flight validation before any mutation
    if not skip_preflight:
        reason = _triton_preflight_check(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            exp_avg_slow,
            exp_avg_scales=exp_avg_scales,
            exp_avg_sq_scales=exp_avg_sq_scales,
            block_size=block_size,
        )
        if reason is not None:
            if os.environ.get("APEXW_DEBUG", "0") == "1":
                print(f"[ApexW] Triton pre-flight rejected: {reason}")
            return False  # Safe: no kernel has run yet.

    block_size = _resolve_block_size(n_elements, exp_avg.dtype, exp_avg_sq.dtype, block_size)
    if block_size is None:
        return False

    has_slow = exp_avg_slow is not None and beta3 is not None
    if not _is_valid_quant_combo(
        param, exp_avg, exp_avg_sq, exp_avg_slow if has_slow else None, exp_avg_scales, exp_avg_sq_scales
    ):
        return False

    # Stage non-contiguous tensors safely without mutating .data storage pointers
    p_work = param if param.is_contiguous() else param.contiguous()
    g_work = grad if grad.is_contiguous() else grad.contiguous()
    m_work = exp_avg if exp_avg.is_contiguous() else exp_avg.contiguous()
    v_work = exp_avg_sq if exp_avg_sq.is_contiguous() else exp_avg_sq.contiguous()
    m_slow_work = (
        (exp_avg_slow if exp_avg_slow.is_contiguous() else exp_avg_slow.contiguous())
        if exp_avg_slow is not None
        else None
    )

    def _sync_back() -> bool:
        if p_work is not param:
            param.copy_(p_work)
        if m_work is not exp_avg:
            exp_avg.copy_(m_work)
        if v_work is not exp_avg_sq:
            exp_avg_sq.copy_(v_work)
        if m_slow_work is not None and m_slow_work is not exp_avg_slow:
            exp_avg_slow.copy_(m_slow_work)
        return True

    eps, eps_scaled = _resolve_fp16_eps(
        eps,
        eps_scaled,
        p_work.dtype == torch.float16 or v_work.dtype == torch.float16,
    )
    num_blocks = triton.cdiv(n_elements, block_size) if (triton is not None and hasattr(triton, "cdiv")) else (n_elements + block_size - 1) // block_size

    # 0A. Fused 0.75-byte / 1.5-byte NF4 INT4 Momentum + Log-UINT8 Variance (precision='8bit')
    if (
        m_work.dtype == torch.uint8
        and v_work.dtype == torch.uint8
        and exp_avg_scales is not None
        and exp_avg_sq_scales is not None
        and _adamw_kernel_triton_8bit_int4 is not None
    ):
        is_unscaled = bool(inv_scale == 1.0)
        is_quad_v = bool(n_elements >= 4 and n_elements % 4 == 0 and v_work.numel() * 4 == n_elements and block_size >= 4)
        half_block = block_size // 2
        quad_block = max(1, block_size // 4)
        num_warps = 4 if (quad_block >= 128 if is_quad_v else half_block >= 128) else 2
        num_stages = _get_num_stages(param.device)
        _adamw_kernel_triton_8bit_int4[(num_blocks,)](
            p_work,
            g_work,
            m_work,
            exp_avg_scales,
            v_work,
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
            HALF_BLOCK=half_block,
            cautious=bool(cautious),
            step=int(step),
            is_unscaled=is_unscaled,
            stochastic=bool(stochastic),
            QUAD_BLOCK=quad_block,
            IS_QUAD_V=is_quad_v,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return _sync_back()

    # 0B. Fused 8-Bit (INT8 m + UINT8 v) and 3-Byte Fast Paths (precision='8bit' / 'ultra')
    if m_work.dtype == torch.int8 and exp_avg_scales is not None:
        if v_work.dtype == torch.uint8 and exp_avg_sq_scales is not None and _adamw_kernel_triton_8bit is not None:
            is_unscaled = bool(inv_scale == 1.0)
            num_warps = 4 if block_size >= 256 else 2
            num_stages = _get_num_stages(param.device)
            _adamw_kernel_triton_8bit[(num_blocks,)](
                p_work,
                g_work,
                m_work,
                exp_avg_scales,
                v_work,
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
                is_unscaled=is_unscaled,
                stochastic=bool(stochastic),
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return _sync_back()
        elif v_work.dtype in (torch.bfloat16, torch.float16, torch.float32) and _adamw_kernel_triton_8bit_m_16bit_v is not None:
            _adamw_kernel_triton_8bit_m_16bit_v[(num_blocks,)](
                p_work,
                g_work,
                m_work,
                exp_avg_scales,
                v_work,
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
            return _sync_back()
        return False

    if m_work.dtype in (torch.int8, torch.uint8) or v_work.dtype in (torch.int8, torch.uint8):
        return False

    # 1. Pure FP32 Fast Path:
    is_quad_v_float = bool(n_elements >= 4 and n_elements % 4 == 0 and v_work.numel() * 4 == n_elements and block_size >= 4)
    quad_block_float = max(1, block_size // 4)
    is_pure_fp32 = (
        p_work.dtype == torch.float32
        and g_work.dtype == torch.float32
        and m_work.dtype == torch.float32
        and v_work.dtype == torch.float32
        and not is_quad_v_float
        and scale_v == 1.0
        and sqrt_scale_v == 1.0
        and not has_slow
        and not cautious
    )

    if is_pure_fp32 and _adamw_kernel_triton_fp32 is not None:
        _adamw_kernel_triton_fp32[(num_blocks,)](
            p_work,
            g_work,
            m_work,
            v_work,
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
        return _sync_back()

    # 2. Cross-Precision Fast Path (precision='compact' with FP32 weights):
    is_cross_precision = (
        p_work.dtype == torch.float32
        and (m_work.dtype in (torch.bfloat16, torch.float16) or v_work.dtype in (torch.bfloat16, torch.float16))
    )

    if is_cross_precision and _adamw_kernel_triton_cross_precision is not None:
        has_scale_v = bool(scale_v != 1.0 or sqrt_scale_v != 1.0)
        _adamw_kernel_triton_cross_precision[(num_blocks,)](
            p_work,
            g_work,
            m_work,
            v_work,
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
            m_slow_ptr=m_slow_work if has_slow else p_work,
            beta3=float(beta3) if beta3 is not None else 0.0,
            slow_ratio=float(slow_ratio),
            has_slow_momentum=bool(has_slow),
            cautious=bool(cautious),
            QUAD_BLOCK=quad_block_float,
            IS_QUAD_V=is_quad_v_float,
        )
        return _sync_back()

    # 3. Standard Unified Triton Kernel (all other cases)
    if _adamw_kernel_triton is not None:
        _adamw_kernel_triton[(num_blocks,)](
            p_work,
            g_work,
            m_work,
            v_work,
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
            m_slow_ptr=m_slow_work if has_slow else p_work,
            beta3=float(beta3) if beta3 is not None else 0.0,
            slow_ratio=float(slow_ratio),
            has_slow_momentum=bool(has_slow),
            cautious=bool(cautious),
            QUAD_BLOCK=quad_block_float,
            IS_QUAD_V=is_quad_v_float,
        )
        return _sync_back()
    return False


def _build_or_reuse_batched_plan(
    cache: Optional[Dict[Any, Any]],
    key: tuple,
    indices: List[int],
    params: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    exp_avg_scales: List[Optional[torch.Tensor]],
    exp_avg_sq_scales: List[Optional[torch.Tensor]],
    wd_flags: List[int],
    block_size: int,
    scales_v_vals: Optional[List[float]] = None,
) -> Dict[str, Any]:
    i0 = indices[0]
    i_last = indices[-1]
    fast_sig = (
        len(indices),
        params[i0].data_ptr(),
        params[i_last].data_ptr(),
        exp_avgs[i0].data_ptr(),
        exp_avg_sqs[i0].data_ptr(),
    )
    entry = cache.get(key) if cache is not None else None
    if entry is not None and entry.get("fast_sig") == fast_sig and entry["wd_sig"] == tuple(wd_flags):
        return entry

    has_q_scales = exp_avg_scales[i0] is not None
    device = params[i0].device
    p_ptrs = torch.tensor([params[idx].data_ptr() for idx in indices], dtype=torch.int64, device=device)
    m_ptrs = torch.tensor([exp_avgs[idx].data_ptr() for idx in indices], dtype=torch.int64, device=device)
    v_ptrs = torch.tensor([exp_avg_sqs[idx].data_ptr() for idx in indices], dtype=torch.int64, device=device)
    numels = torch.tensor([params[idx].numel() for idx in indices], dtype=torch.int64, device=device)
    wd_t = torch.tensor(wd_flags, dtype=torch.int8, device=device)

    blk_tensor_ids = []
    blk_local_ids = []
    for t_i, idx in enumerate(indices):
        n_el = params[idx].numel()
        nb = (n_el + block_size - 1) // block_size
        blk_tensor_ids.extend([t_i] * nb)
        blk_local_ids.extend(range(nb))

    blk_to_tensor = torch.tensor(blk_tensor_ids, dtype=torch.int32, device=device)
    blk_local_id = torch.tensor(blk_local_ids, dtype=torch.int32, device=device)

    entry = {
        "fast_sig": fast_sig,
        "wd_sig": tuple(wd_flags),
        "total_blocks": len(blk_tensor_ids),
        "p_ptrs": p_ptrs,
        "m_ptrs": m_ptrs,
        "v_ptrs": v_ptrs,
        "numels": numels,
        "wd_flags": wd_t,
        "blk_to_tensor": blk_to_tensor,
        "blk_local_id": blk_local_id,
    }
    if has_q_scales:
        entry["ms_ptrs"] = torch.tensor(
            [exp_avg_scales[idx].data_ptr() for idx in indices], dtype=torch.int64, device=device
        )
        entry["vs_ptrs"] = torch.tensor(
            [exp_avg_sq_scales[idx].data_ptr() for idx in indices], dtype=torch.int64, device=device
        )
    if scales_v_vals is not None:
        entry["scale_v"] = torch.tensor(scales_v_vals, dtype=torch.float32, device=device)
        entry["sqrt_scale_v"] = torch.sqrt(entry["scale_v"])
    else:
        empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
        entry["scale_v"] = empty_f32
        entry["sqrt_scale_v"] = empty_f32
    if cache is not None:
        cache[key] = entry
    return entry


def _resolve_g_ptrs_tensor(
    plan: Dict[str, Any],
    idxs: List[int],
    grads: List[torch.Tensor],
    dev: torch.device,
) -> torch.Tensor:
    """Reuse cached GPU gradient pointer array when pointers repeat; stage via single C++ tensor copy otherwise."""
    sig = [grads[idx].data_ptr() for idx in idxs]
    g_ptrs = plan.get("g_ptrs")
    if g_ptrs is not None and plan.get("g_sig") == sig:
        return g_ptrs

    n = len(idxs)
    if g_ptrs is None or g_ptrs.numel() != n or g_ptrs.device != dev:
        g_ptrs = torch.empty(n, dtype=torch.int64, device=dev)
        plan["g_ptrs"] = g_ptrs

    host_t = torch.tensor(sig, dtype=torch.int64)
    if dev.type == "cuda" and torch.cuda.is_available():
        host = plan.get("g_ptrs_host")
        if host is None or host.numel() != n:
            host = torch.empty(n, dtype=torch.int64, pin_memory=True)
            plan["g_ptrs_host"] = host
        host.copy_(host_t)
        g_ptrs.copy_(host, non_blocking=True)
    else:
        g_ptrs.copy_(host_t)
    plan["g_sig"] = sig
    return g_ptrs


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
    block_size: Optional[Union[int, List[Optional[int]]]] = None,
    exp_avgs_slow: Optional[List[torch.Tensor]] = None,
    beta3: Optional[float] = None,
    alpha: float = 5.0,
    cautious: bool = False,
    skip_preflight: bool = False,
    exp_avg_scales: Optional[Union[List[torch.Tensor], Dict[int, torch.Tensor]]] = None,
    exp_avg_sq_scales: Optional[Union[List[torch.Tensor], Dict[int, torch.Tensor]]] = None,
    stochastic: bool = True,
    batch_cache: Optional[Dict[Any, Any]] = None,
) -> bool:
    """
    Executes the Triton fused AdamW kernel across multiple tensors.
    Works natively on both NVIDIA CUDA and AMD ROCm GPUs with zero C++ compilation.
    Fuses gradient unscaling, AdEMAMix dual-speed momentum, cautious directional masking,
    8-bit/NF4-4bit dynamic block quantization, and batched pointer-array multi-tensor launch.
    """
    global _BATCHED_TRITON_ENABLED
    if not is_triton_available():
        return False
    if len(params) == 0:
        return True

    has_slow_list = exp_avgs_slow is not None and len(exp_avgs_slow) == len(params)
    is_scales_v_list = isinstance(scales_v, list)
    is_m_scales_list = isinstance(exp_avg_scales, list)
    is_v_scales_list = isinstance(exp_avg_sq_scales, list)
    is_bsize_list = isinstance(block_size, list)

    def _get_scale(scales_arg, is_list: bool, idx: int, p: torch.Tensor):
        if scales_arg is None:
            return None
        if is_list:
            return scales_arg[idx]
        if isinstance(scales_arg, dict):
            return scales_arg.get(id(p), None)
        return None

    fast_bucket_entry = (
        batch_cache.get("_fast_buckets")
        if (skip_preflight and batch_cache is not None and not has_slow_list)
        else None
    )
    if (
        fast_bucket_entry is not None
        and fast_bucket_entry[0] == (len(params), id(params[0]), grads[0].dtype)
    ):
        _, s_m_resolved, s_v_resolved, b_resolved, buckets_8bit, buckets_compact = fast_bucket_entry
    else:
        # Validate the entire batch up front so we never leave parameters partially stepped
        s_m_resolved = []
        s_v_resolved = []
        b_resolved = []
        for i, p in enumerate(params):
            s_m_i = _get_scale(exp_avg_scales, is_m_scales_list, i, p)
            s_v_i = _get_scale(exp_avg_sq_scales, is_v_scales_list, i, p)
            b_i = block_size[i] if is_bsize_list else block_size
            slow_i = exp_avgs_slow[i] if (has_slow_list and beta3 is not None) else None
            if not skip_preflight:
                reason = _triton_preflight_check(
                    p,
                    grads[i],
                    exp_avgs[i],
                    exp_avg_sqs[i],
                    slow_i,
                    exp_avg_scales=s_m_i,
                    exp_avg_sq_scales=s_v_i,
                    block_size=b_i,
                )
                if reason is not None:
                    if os.environ.get("APEXW_DEBUG", "0") == "1":
                        print(f"[ApexW] Triton pre-flight rejected param[{i}]: {reason}")
                    return False
            else:
                m_i, v_i = exp_avgs[i], exp_avg_sqs[i]
                if not _is_valid_quant_combo(p, m_i, v_i, slow_i, s_m_i, s_v_i):
                    return False
                if b_i is not None:
                    if not isinstance(b_i, int) or isinstance(b_i, bool) or b_i <= 0 or (b_i & (b_i - 1)) != 0:
                        return False
                    if m_i.dtype == torch.uint8 and b_i < 2:
                        return False
            s_m_resolved.append(s_m_i)
            s_v_resolved.append(s_v_i)
            b_resolved.append(_resolve_block_size(p.numel(), exp_avgs[i].dtype, exp_avg_sqs[i].dtype, b_i))

        buckets_8bit = {}
        buckets_compact = {}
        if (
            _BATCHED_TRITON_ENABLED
            and len(params) >= 2
            and triton is not None
            and _adamw_kernel_triton_8bit_batched is not None
            and _adamw_kernel_triton_compact_batched is not None
        ):
            for i, p in enumerate(params):
                n_el = p.numel()
                if n_el == 0:
                    continue
                g = grads[i]
                m = exp_avgs[i]
                v = exp_avg_sqs[i]
                b_i = b_resolved[i]
                if b_i is None or not (
                    p.is_contiguous() and g.is_contiguous() and m.is_contiguous() and v.is_contiguous()
                ):
                    continue
                p_code = _dtype_to_code(p.dtype)
                g_code = _dtype_to_code(g.dtype)
                if p_code < 0 or g_code < 0:
                    continue
                if m.dtype in (torch.int8, torch.uint8) and v.dtype == torch.uint8 and s_m_resolved[i] is not None and s_v_resolved[i] is not None:
                    is_int4_m = (m.dtype == torch.uint8)
                    is_quad_v = bool(is_int4_m and n_el >= 4 and n_el % 4 == 0 and v.numel() * 4 == n_el and b_i >= 4)
                    key = (p.device.index or 0, p_code, g_code, is_int4_m, is_quad_v, b_i)
                    buckets_8bit.setdefault(key, []).append(i)
                elif not has_slow_list and m.dtype not in (torch.int8, torch.uint8) and v.dtype not in (torch.int8, torch.uint8):
                    m_code = _dtype_to_code(m.dtype)
                    v_code = _dtype_to_code(v.dtype)
                    if m_code >= 0 and v_code >= 0:
                        is_quad_v_c = bool(n_el >= 4 and n_el % 4 == 0 and v.numel() * 4 == n_el and b_i >= 4)
                        key = (p.device.index or 0, p_code, g_code, m_code, v_code, is_quad_v_c, b_i)
                        buckets_compact.setdefault(key, []).append(i)

            if skip_preflight and batch_cache is not None and not has_slow_list:
                batch_cache["_fast_buckets"] = (
                    (len(params), id(params[0]), grads[0].dtype),
                    s_m_resolved,
                    s_v_resolved,
                    b_resolved,
                    buckets_8bit,
                    buckets_compact,
                )

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    bc2_sqrt = math.sqrt(bc2)
    step_size = lr * bc2_sqrt / bc1
    eps_scaled = eps * bc2_sqrt

    slow_ratio = 0.0
    if beta3 is not None and has_slow_list:
        bc3 = 1.0 - (beta3 ** step)
        slow_ratio = alpha * (bc1 / bc3) if bc3 > 0.0 else 0.0

    # Fast Batched Multi-Tensor Pointer-Array Triton Execution:
    # Groups >= 2 homogeneous contiguous GPU tensors so the entire bucket executes in ONE launch.
    handled = [p.numel() == 0 for p in params]
    if (
        _BATCHED_TRITON_ENABLED
        and len(params) >= 2
        and triton is not None
        and _adamw_kernel_triton_8bit_batched is not None
        and _adamw_kernel_triton_compact_batched is not None
    ):
        is_unscaled = bool(inv_scale == 1.0)
        try:
            for key, idxs in buckets_8bit.items():
                if len(idxs) < 2:
                    continue
                _, p_code, g_code, is_int4_m, is_quad_v, b_i = key
                dev = params[idxs[0]].device
                wd_flags = [
                    1 if (weight_decay != 0.0 and not (decay_norm_only and params[idx].dim() < 2)) else 0
                    for idx in idxs
                ]
                plan = _build_or_reuse_batched_plan(
                    batch_cache, ("8bit", key), idxs, params, exp_avgs, exp_avg_sqs,
                    s_m_resolved, s_v_resolved, wd_flags, b_i,
                )
                g_ptrs = _resolve_g_ptrs_tensor(plan, idxs, grads, dev)
                half_b = max(1, b_i // 2)
                quad_b = max(1, b_i // 4)
                num_warps = 8 if (b_i >= 1024 and not is_quad_v) else (4 if ((quad_b >= 128 if is_quad_v else half_b >= 128) if is_int4_m else b_i >= 256) else 2)
                num_stages = _get_num_stages(dev)
                b_eps, b_eps_scaled = _resolve_fp16_eps(eps, eps_scaled, False)
                _adamw_kernel_triton_8bit_batched[(plan["total_blocks"],)](
                    plan["p_ptrs"],
                    g_ptrs,
                    plan["m_ptrs"],
                    plan["ms_ptrs"],
                    plan["v_ptrs"],
                    plan["vs_ptrs"],
                    plan["numels"],
                    plan["wd_flags"],
                    plan["blk_to_tensor"],
                    plan["blk_local_id"],
                    float(lr),
                    float(beta1),
                    float(beta2),
                    float(b_eps),
                    float(weight_decay),
                    float(step_size),
                    float(b_eps_scaled),
                    float(inv_scale),
                    BLOCK_SIZE=b_i,
                    HALF_BLOCK=half_b,
                    P_CODE=p_code,
                    G_CODE=g_code,
                    IS_INT4_M=bool(is_int4_m),
                    cautious=bool(cautious),
                    step=int(step),
                    is_unscaled=is_unscaled,
                    stochastic=bool(stochastic),
                    QUAD_BLOCK=quad_b,
                    IS_QUAD_V=bool(is_quad_v),
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                for idx in idxs:
                    handled[idx] = True

            for key, idxs in buckets_compact.items():
                if len(idxs) < 2:
                    continue
                _, p_code, g_code, m_code, v_code, is_quad_v_c, b_i = key
                dev = params[idxs[0]].device
                wd_flags = [
                    1 if (weight_decay != 0.0 and not (decay_norm_only and params[idx].dim() < 2)) else 0
                    for idx in idxs
                ]
                sv_vals = []
                has_sv = False
                for idx in idxs:
                    sv = 1.0
                    if scales_v is not None:
                        if is_scales_v_list:
                            sv = scales_v[idx]
                        elif id(params[idx]) in scales_v:
                            sv = scales_v[id(params[idx])]
                    if sv != 1.0:
                        has_sv = True
                    sv_vals.append(float(sv))
                plan = _build_or_reuse_batched_plan(
                    batch_cache, ("compact", key, has_sv), idxs, params, exp_avgs, exp_avg_sqs,
                    s_m_resolved, s_v_resolved, wd_flags, b_i,
                    scales_v_vals=sv_vals if has_sv else None,
                )
                g_ptrs = _resolve_g_ptrs_tensor(plan, idxs, grads, dev)
                sv_t = plan["scale_v"]
                ssv_t = plan["sqrt_scale_v"]
                _, b_eps_scaled = _resolve_fp16_eps(eps, eps_scaled, False)
                quad_b_c = max(1, b_i // 4)
                num_warps = 4 if (quad_b_c >= 128 if is_quad_v_c else b_i >= 512) else 2
                num_stages = _get_num_stages(dev)
                _adamw_kernel_triton_compact_batched[(plan["total_blocks"],)](
                    plan["p_ptrs"],
                    g_ptrs,
                    plan["m_ptrs"],
                    plan["v_ptrs"],
                    plan["numels"],
                    plan["wd_flags"],
                    sv_t,
                    ssv_t,
                    plan["blk_to_tensor"],
                    plan["blk_local_id"],
                    float(lr),
                    float(beta1),
                    float(beta2),
                    float(weight_decay),
                    float(step_size),
                    float(b_eps_scaled),
                    float(inv_scale),
                    BLOCK_SIZE=b_i,
                    P_CODE=p_code,
                    G_CODE=g_code,
                    M_CODE=m_code,
                    V_CODE=v_code,
                    has_scale_v=has_sv,
                    cautious=bool(cautious),
                    QUAD_BLOCK=quad_b_c,
                    IS_QUAD_V=bool(is_quad_v_c),
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                for idx in idxs:
                    handled[idx] = True
        except Exception as e:
            _BATCHED_TRITON_ENABLED = False
            if os.environ.get("APEXW_DEBUG", "0") == "1":
                print(f"[ApexW] Batched Triton fallback to per-tensor Triton: {e}")

    for i, p in enumerate(params):
        if handled[i]:
            continue
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

        ok = triton_adamw_step(
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
            block_size=b_resolved[i],
            skip_preflight=True,
            exp_avg_slow=m_slow,
            beta3=beta3,
            slow_ratio=slow_ratio,
            cautious=cautious,
            exp_avg_scales=s_m_resolved[i],
            exp_avg_sq_scales=s_v_resolved[i],
            step=step,
            stochastic=stochastic,
        )
        if not ok:
            return False

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
    step: int = 1,
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
        step=step,
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


def triton_factored_v_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq_row: torch.Tensor,
    exp_avg_sq_col: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    *,
    apply_wd: bool = True,
    inv_scale: float = 1.0,
    cautious: bool = False,
    stochastic: bool = True,
    is_6bit_m: bool = False,
    exp_avg_scales: Optional[torch.Tensor] = None,
    block_size: int = 256,
) -> bool:
    """
    Launch Triton factored-v AdamW kernel (Compact BF16/FP16/FP32, INT8 m, or 6-bit packed m).
    Updates Adafactor row/col statistics in bounded row chunks before launching the SRAM kernel.
    Returns False cleanly if Triton is unavailable or the tensor configuration is unsupported.
    """
    if not _HAS_TRITON or _adamw_kernel_triton_factored_v is None:
        return False
    if not (param.is_cuda and grad.is_cuda and exp_avg.is_cuda
            and exp_avg_sq_row.is_cuda and exp_avg_sq_col.is_cuda):
        return False
    if isinstance(inv_scale, torch.Tensor):
        if inv_scale.numel() == 1:
            inv_scale = float(inv_scale.item())
        else:
            return False
    if not (param.is_contiguous() and grad.is_contiguous() and exp_avg.is_contiguous()
            and exp_avg_sq_row.is_contiguous() and exp_avg_sq_col.is_contiguous()):
        return False

    numel = param.numel()
    rows = exp_avg_sq_row.numel()
    cols = exp_avg_sq_col.numel()
    if numel <= 0 or rows <= 0 or cols <= 0 or rows * cols != numel:
        return False

    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        block_size = 256

    is_int8_m = exp_avg.dtype == torch.int8
    if is_6bit_m:
        if cautious or _adamw_kernel_triton_factored_v_6bit is None:
            return False
        if exp_avg.dtype != torch.uint8 or exp_avg_scales is None or not exp_avg_scales.is_cuda:
            return False
        if block_size % 4 != 0:
            return False
    elif is_int8_m:
        if exp_avg_scales is None or not exp_avg_scales.is_cuda or exp_avg.numel() != numel:
            return False
    else:
        if exp_avg.dtype not in (torch.float32, torch.float16, torch.bfloat16) or exp_avg.numel() != numel:
            return False

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    bc2_sqrt = math.sqrt(bc2)
    step_size = float(lr) * bc2_sqrt / bc1
    effective_eps = float(eps)
    eps_scaled = effective_eps * bc2_sqrt
    inv_s = float(inv_scale)
    is_unscaled = (inv_s == 1.0)

    # Fast Adafactor row/col statistics update with zero in-place mutation of caller grad
    chunk_rows = max(1, 262144 // cols)
    g_2d = grad.view(rows, cols)
    scale_alpha = (1.0 - beta2) * (inv_s * inv_s)
    if rows <= chunk_rows:
        g_sq = g_2d.float().nan_to_num_(nan=0.0, posinf=65504.0, neginf=-65504.0).square_()
        exp_avg_sq_row.mul_(beta2).add_(g_sq.mean(dim=1), alpha=scale_alpha)
        exp_avg_sq_col.mul_(beta2).add_(g_sq.mean(dim=0), alpha=scale_alpha)
        del g_sq
    else:
        col_sum_sq = torch.zeros((cols,), device=param.device, dtype=torch.float32)
        for r0 in range(0, rows, chunk_rows):
            r1 = min(r0 + chunk_rows, rows)
            g_chunk = g_2d[r0:r1].float().nan_to_num_(nan=0.0, posinf=65504.0, neginf=-65504.0).square_()
            exp_avg_sq_row[r0:r1].mul_(beta2).add_(g_chunk.mean(dim=1), alpha=scale_alpha)
            col_sum_sq.add_(g_chunk.sum(dim=0))
            del g_chunk
        exp_avg_sq_col.mul_(beta2).add_(col_sum_sq, alpha=scale_alpha / max(rows, 1))
        del col_sum_sq

    exp_avg_sq_row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    exp_avg_sq_col.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    inv_r_mean = 1.0 / exp_avg_sq_row.mean().clamp_min_(1e-30)
    row_norm = (exp_avg_sq_row * inv_r_mean).sqrt_()
    col_sqrt = exp_avg_sq_col.clamp_min(0.0).sqrt()

    grid = ((numel + block_size - 1) // block_size,)
    if is_6bit_m:
        n_quads = (numel + 3) // 4
        _adamw_kernel_triton_factored_v_6bit[grid](
            param,
            grad,
            exp_avg,
            exp_avg_scales,
            row_norm,
            col_sqrt,
            numel,
            n_quads,
            cols,
            float(lr),
            float(beta1),
            effective_eps,
            float(weight_decay),
            step_size,
            eps_scaled,
            inv_s,
            apply_wd=bool(apply_wd),
            BLOCK_SIZE=block_size,
            QUADS=block_size // 4,
            cautious=False,
            step=int(step),
            is_unscaled=is_unscaled,
            stochastic=bool(stochastic),
        )
    else:
        m_scales_ptr = exp_avg_scales if is_int8_m else param
        _adamw_kernel_triton_factored_v[grid](
            param,
            grad,
            exp_avg,
            m_scales_ptr,
            row_norm,
            col_sqrt,
            numel,
            cols,
            float(lr),
            float(beta1),
            effective_eps,
            float(weight_decay),
            step_size,
            eps_scaled,
            inv_s,
            apply_wd=bool(apply_wd),
            IS_INT8_M=is_int8_m,
            BLOCK_SIZE=block_size,
            cautious=bool(cautious),
            step=int(step),
            is_unscaled=is_unscaled,
            stochastic=bool(stochastic),
        )
    return True

