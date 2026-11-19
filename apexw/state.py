"""State management, dtype resolution, and factored second-moment utilities for ApexW."""

import math
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor

from .quantization import (
    _dequantize_blockwise_6bit,
    _dequantize_blockwise_int8,
    _quantize_blockwise_6bit,
    _quantize_blockwise_int8,
)

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
_CUDA_DTYPE_TRIOS = {
    (torch.float32, torch.float32, torch.float32),
    (torch.float32, torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.bfloat16, torch.float32),
    (torch.float32, torch.float16, torch.float16),
    (torch.float32, torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.float32, torch.float32),
    (torch.float16, torch.float16, torch.float16),
    (torch.float16, torch.float16, torch.float32),
    (torch.float16, torch.float32, torch.float32),
}

_BETA2_WARN_THRESHOLD = 0.995
_BETA2_V_FP32_THRESHOLD = 0.995
_FACTORED_CHUNK_ELEMS = 262144


def _mps_supports_bfloat16() -> bool:
    """Return True if the Apple Silicon MPS backend supports bfloat16."""
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


def _is_peft_parameter(p: Tensor) -> bool:
    """Return True if a parameter belongs to a PEFT/LoRA adapter."""
    if getattr(p, "_is_lora", False):
        return True
    for attr in ("_param_name", "name"):
        val = getattr(p, attr, None)
        if isinstance(val, str) and any(
            k in val.lower() for k in ("lora_a", "lora_b", "lora_embedding", "adapter", "lora_", "lora")
        ):
            return True
    if p.dim() == 2 and min(p.shape) <= 64 and max(p.shape) >= 256:
        return True
    return False


def _detect_peft_context(param_groups: List[Dict[str, Any]]) -> bool:
    """Return True if parameter groups contain PEFT/QLoRA tensors."""
    for g in param_groups:
        params = g.get("params", [])
        for p in params:
            if isinstance(p, Tensor):
                if _is_peft_parameter(p):
                    return True
                if (
                    hasattr(p, "quant_state")
                    or hasattr(p, "quant_type")
                    or type(p).__name__ in ("Params4bit", "Linear4bit")
                ):
                    return True
    return False


def _normalize_precision(precision: Union[str, torch.dtype]) -> str:
    """Normalize user-provided precision mode into '8bit', 'ultra', 'compact', 'hybrid', 'auto', or 'full'."""
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
        if p_clean in ("hybrid", "hybrid_apexw", "hybrid-apexw", "apex_hybrid", "hybrid_compact"):
            return "hybrid"
        if p_clean in ("auto", "smart", "adaptive"):
            return "auto"
    raise ValueError(
        f"Invalid precision mode '{precision}'. Must be '8bit', 'compact', 'hybrid', 'ultra', 'auto', 'full', "
        f"or a valid torch dtype (e.g. torch.float32, torch.bfloat16, torch.float16, torch.int8)."
    )


def _get_compact_dtype(p: Tensor) -> torch.dtype:
    """Select the 16-bit storage dtype supported by the tensor's device."""
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


def _get_v_dtype_for_beta2(
    compact_dtype: torch.dtype,
    beta2: float,
    force_compact_v: bool = False,
    force_fp32_v: bool = False,
    stochastic_rounding: bool = True,
) -> torch.dtype:
    """Return the storage dtype for the second-moment buffer given beta2.

    When stochastic rounding is enabled (default in ApexW), 16-bit second moments
    (bfloat16/float16) track gradient variance with zero systematic truncation bias,
    preserving full 50% VRAM savings (4.0 B/param) and enabling 128-bit vectorized
    CUDA kernel execution across all beta2 values.

    If stochastic_rounding is explicitly disabled and beta2 >= _BETA2_V_FP32_THRESHOLD
    (or force_fp32_v is requested), exp_avg_sq is stored in float32.
    """
    if force_compact_v:
        return compact_dtype
    if force_fp32_v:
        return torch.float32
    if not stochastic_rounding and compact_dtype in (torch.bfloat16, torch.float16) and beta2 >= _BETA2_V_FP32_THRESHOLD:
        return torch.float32
    return compact_dtype


def _get_scalar_dtype(is_fused: bool = False) -> torch.dtype:
    """Return the step counter dtype matching PyTorch Optimizer conventions."""
    if is_fused:
        return torch.float32
    return torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32


def _get_value(x):
    """Extract Python scalar from a 0-dim tensor or return scalar as-is."""
    if isinstance(x, Tensor):
        return x.item()
    return x


class PackedStateProxy(MutableMapping):
    """Sliced view into a flat optimizer state for auto-packed parameters."""

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
    """Wraps optimizer.state to support state lookup for auto-packed parameters."""

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


def _can_use_factored_v_8bit(
    p: Tensor,
    *,
    is_peft: bool,
    amsgrad: bool,
    is_lion: bool,
    has_slow_momentum: bool,
) -> bool:
    """Return True if a 2D+ tensor qualifies for factored FP32 v + quantized m."""
    if (
        p.dim() < 2
        or amsgrad
        or is_lion
        or has_slow_momentum
        or torch.is_complex(p)
        or p.dtype == torch.float64
    ):
        return False
    rows = p.shape[0]
    cols = p.numel() // rows
    if is_peft or p.numel() < 65536:
        return False
    return rows >= 128 and cols >= 128


def _can_use_factored_v_compact(
    p: Tensor,
    *,
    is_peft: bool,
    amsgrad: bool,
    is_lion: bool,
    has_slow_momentum: bool,
) -> bool:
    """Return True if a large 2D+ tensor qualifies for factored FP32 v in compact mode."""
    if (
        p.dim() < 2
        or amsgrad
        or is_lion
        or has_slow_momentum
        or torch.is_complex(p)
        or p.dtype == torch.float64
    ):
        return False
    rows = p.shape[0]
    cols = p.numel() // rows
    if is_peft or p.numel() < 524288:
        return False
    return rows >= 128 and cols >= 128


def _partition_factored_v_indices(
    state: dict,
    params: List[Tensor],
) -> tuple[List[int], List[int]]:
    """Split parameter indices into elementwise and factored-v groups."""
    elem_idxs: List[int] = []
    factored_idxs: List[int] = []
    for i, p in enumerate(params):
        st = state.get(p)
        if st is not None and st.get("is_factored_v", False):
            factored_idxs.append(i)
        else:
            elem_idxs.append(i)
    return elem_idxs, factored_idxs


def _compute_factored_v_denom(
    g_f: Tensor,
    r_t: Tensor,
    c_t: Tensor,
    *,
    beta2: float,
    eps_scaled: float,
    effective_eps: float,
    out_shape: torch.Size,
) -> Tensor:
    """Update row/column second moments and return sqrt(v) + eps."""
    rows = r_t.numel()
    cols = c_t.numel()
    g_sq = g_f.view(rows, cols).square()
    r_t.mul_(beta2).add_(g_sq.mean(dim=1), alpha=1.0 - beta2)
    c_t.mul_(beta2).add_(g_sq.mean(dim=0), alpha=1.0 - beta2)
    del g_sq
    r_t.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    c_t.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    inv_r_mean = 1.0 / r_t.mean().clamp_min_(1e-30)
    denom = (r_t * inv_r_mean).sqrt().unsqueeze(1) * c_t.sqrt().unsqueeze(0)
    denom.add_(eps_scaled).clamp_min_(effective_eps)
    return denom.view(out_shape)


def _prep_grad_chunk_fp32(
    g_slice: Tensor,
    *,
    sanitize_g: bool,
    maximize: bool,
    inv_scale: Optional[Union[float, Tensor]],
    force_copy: bool = False,
) -> Tensor:
    """Return a float32 gradient chunk with optional sanitization, negation, and AMP unscaling."""
    need_transform = sanitize_g or maximize or (inv_scale is not None)
    if g_slice.dtype == torch.float32 and not need_transform and not force_copy:
        return g_slice
    gc = g_slice.float() if (g_slice.dtype != torch.float32 or need_transform) else g_slice.clone()
    if gc is g_slice and (need_transform or force_copy):
        gc = g_slice.clone()
    if maximize:
        gc.neg_()
    if sanitize_g:
        gc.nan_to_num_(nan=0.0, posinf=65504.0, neginf=-65504.0)
    if inv_scale is not None:
        gc.mul_(inv_scale)
    return gc


def _apply_factored_v_chunk_body(
    p_slice: Tensor,
    g_f_chunk: Tensor,
    denom_c: Tensor,
    m_buf: Tensor,
    m_scales: Optional[Tensor],
    *,
    off: int,
    end: int,
    chunk_n: int,
    is_6bit: bool,
    is_int8: bool,
    beta1: float,
    step_size: float,
    nesterov: bool,
    cautious: bool,
    stochastic: bool,
    block_size: int,
    wd_factor: Optional[float],
    is_last_chunk: bool = False,
) -> None:
    """Update one contiguous slice of m and p given its factored denominator."""
    blk0 = off // block_size
    if is_6bit:
        byte0 = (off // 4) * 3
        if is_last_chunk:
            q_slice = m_buf[byte0:]
            sc_slice = m_scales[blk0:]
        else:
            q_slice = m_buf[byte0 : byte0 + (chunk_n // 4) * 3]
            sc_slice = m_scales[blk0 : blk0 + (chunk_n // block_size)]
        m_c = _dequantize_blockwise_6bit(q_slice, sc_slice, numel=chunk_n, block_size=block_size)
    elif is_int8:
        m_flat = m_buf.reshape(-1)
        m_slice = m_flat[off:end]
        if is_last_chunk:
            sc_slice = m_scales[blk0:]
        else:
            sc_slice = m_scales[blk0 : blk0 + (chunk_n // block_size)]
        m_c = _dequantize_blockwise_int8(m_slice, sc_slice, block_size=block_size)
    else:
        m_flat = m_buf.reshape(-1)
        m_slice = m_flat[off:end]
        m_c = m_slice.float()

    m_c.lerp_(g_f_chunk, 1.0 - beta1)
    num_c = torch.lerp(g_f_chunk, m_c, beta1) if nesterov else m_c
    if cautious:
        mask = (num_c * g_f_chunk >= 0).to(torch.float32)
        mask_mean = mask.mean().clamp(min=1e-3)
        num_c = num_c * (mask / mask_mean)

    torch.div(num_c, denom_c, out=denom_c)
    denom_c.mul_(step_size)

    if p_slice.dtype == torch.float32:
        if wd_factor is not None:
            p_slice.mul_(wd_factor)
        p_slice.sub_(denom_c)
    elif p_slice.dtype == torch.float16:
        p_f = p_slice.float()
        if wd_factor is not None:
            p_f.mul_(wd_factor)
        p_f.sub_(denom_c).clamp_(-65504.0, 65504.0)
        p_slice.copy_(p_f.to(torch.float16))
    else:
        if wd_factor is not None:
            p_slice.mul_(wd_factor)
        p_slice.sub_(denom_c.to(p_slice.dtype))

    if is_6bit:
        _quantize_blockwise_6bit(
            m_c,
            block_size=block_size,
            stochastic=stochastic,
            out_q=q_slice,
            out_scales=sc_slice,
            old_scales=sc_slice,
        )
    elif is_int8:
        _quantize_blockwise_int8(
            m_c,
            block_size=block_size,
            stochastic=stochastic,
            out_q=m_slice,
            out_scales=sc_slice,
            old_scales=sc_slice,
        )
    else:
        m_slice.copy_(m_c if m_c.dtype == m_slice.dtype else m_c.to(m_slice.dtype))


def _apply_factored_v_int8_update(
    p: Tensor,
    g: Tensor,
    p_state: dict,
    *,
    beta1: float,
    beta2: float,
    step_size: float,
    eps_scaled: float,
    effective_eps: float,
    nesterov: bool,
    cautious: bool,
    stochastic: bool,
    block_size: int,
    wd_factor: Optional[float] = None,
    inv_scale: Optional[Union[float, Tensor]] = None,
    maximize: bool = False,
    sanitize_g: bool = False,
) -> None:
    """Apply chunked update for factored FP32 v with 6-bit, INT8, or compact m."""
    numel = p.numel()
    if numel == 0:
        return
    r_t = p_state["exp_avg_sq_row"]
    c_t = p_state["exp_avg_sq_col"]
    rows = r_t.numel()
    cols = c_t.numel()
    m_buf = p_state["exp_avg"]
    m_scales = p_state.get("exp_avg_scales")
    is_6bit = bool(p_state.get("is_6bit_m", False))
    is_int8 = (not is_6bit) and (m_buf.dtype == torch.int8)

    if not p.is_contiguous() or not m_buf.is_contiguous():
        g_f = _prep_grad_chunk_fp32(
            g.reshape(-1), sanitize_g=sanitize_g, maximize=maximize, inv_scale=inv_scale
        ).view(p.shape)
        m_deq = (
            _dequantize_blockwise_int8(m_buf, m_scales, block_size=block_size)
            if is_int8
            else m_buf.float()
        )
        m_deq.lerp_(g_f, 1.0 - beta1)
        denom = _compute_factored_v_denom(
            g_f, r_t, c_t, beta2=beta2, eps_scaled=eps_scaled, effective_eps=effective_eps, out_shape=p.shape
        )
        num = torch.lerp(g_f, m_deq, beta1) if nesterov else m_deq
        if cautious:
            mask = (num * g_f >= 0).to(torch.float32)
            num = num * (mask / mask.mean().clamp(min=1e-3))
        torch.div(num, denom, out=denom)
        denom.mul_(step_size)
        if p.dtype == torch.float32:
            if wd_factor is not None:
                p.mul_(wd_factor)
            p.sub_(denom)
        elif p.dtype == torch.float16:
            p_f = p.float()
            if wd_factor is not None:
                p_f.mul_(wd_factor)
            p_f.sub_(denom).clamp_(-65504.0, 65504.0)
            p.copy_(p_f.to(torch.float16))
        else:
            if wd_factor is not None:
                p.mul_(wd_factor)
            p.sub_(denom.to(p.dtype))
        if is_int8:
            _quantize_blockwise_int8(
                m_deq, block_size=block_size, stochastic=stochastic,
                out_q=m_buf, out_scales=m_scales, old_scales=m_scales,
            )
        else:
            m_buf.copy_(m_deq.to(m_buf.dtype))
        return

    p_flat = p.reshape(-1)
    g_flat = g.reshape(-1) if g.is_contiguous() else g.contiguous().reshape(-1)

    if numel <= _FACTORED_CHUNK_ELEMS:
        g_f = _prep_grad_chunk_fp32(
            g_flat, sanitize_g=sanitize_g, maximize=maximize, inv_scale=inv_scale
        )
        denom = _compute_factored_v_denom(
            g_f, r_t, c_t, beta2=beta2, eps_scaled=eps_scaled, effective_eps=effective_eps, out_shape=p.shape
        ).reshape(-1)
        _apply_factored_v_chunk_body(
            p_flat, g_f, denom, m_buf, m_scales,
            off=0, end=numel, chunk_n=numel,
            is_6bit=is_6bit, is_int8=is_int8,
            beta1=beta1, step_size=step_size,
            nesterov=nesterov, cautious=cautious,
            stochastic=stochastic, block_size=block_size,
            wd_factor=wd_factor, is_last_chunk=True,
        )
        return

    r_step_1 = max(1, _FACTORED_CHUNK_ELEMS // cols)
    c_acc = torch.zeros(cols, dtype=torch.float32, device=p.device)
    for r0 in range(0, rows, r_step_1):
        r1 = min(r0 + r_step_1, rows)
        gc = _prep_grad_chunk_fp32(
            g_flat[r0 * cols : r1 * cols],
            sanitize_g=sanitize_g,
            maximize=maximize,
            inv_scale=inv_scale,
        )
        g_sq = gc.view(r1 - r0, cols).square()
        r_t[r0:r1].mul_(beta2).add_(g_sq.mean(dim=1), alpha=1.0 - beta2)
        c_acc.add_(g_sq.sum(dim=0))
        del gc, g_sq

    c_t.mul_(beta2).add_(c_acc, alpha=(1.0 - beta2) / rows)
    del c_acc
    r_t.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    c_t.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    inv_r_mean = 1.0 / r_t.mean().clamp_min_(1e-30)
    row_norm = (r_t * inv_r_mean).sqrt_()
    col_sqrt = c_t.sqrt()

    align_unit = math.lcm(block_size, 4) if is_6bit else (block_size if is_int8 else 1)
    g_step = math.gcd(cols, align_unit)
    row_mult = align_unit // g_step
    if row_mult * cols <= _FACTORED_CHUNK_ELEMS * 2:
        r_step_2 = max(row_mult, ((_FACTORED_CHUNK_ELEMS // (row_mult * cols)) or 1) * row_mult)
        for r0 in range(0, rows, r_step_2):
            r1 = min(r0 + r_step_2, rows)
            off = r0 * cols
            end = r1 * cols
            chunk_n = end - off
            gc = _prep_grad_chunk_fp32(
                g_flat[off:end], sanitize_g=sanitize_g, maximize=maximize, inv_scale=inv_scale
            )
            denom_c = (row_norm[r0:r1].unsqueeze(1) * col_sqrt.unsqueeze(0)).add_(eps_scaled).clamp_min_(effective_eps).reshape(-1)
            _apply_factored_v_chunk_body(
                p_flat[off:end], gc, denom_c, m_buf, m_scales,
                off=off, end=end, chunk_n=chunk_n,
                is_6bit=is_6bit, is_int8=is_int8,
                beta1=beta1, step_size=step_size,
                nesterov=nesterov, cautious=cautious,
                stochastic=stochastic, block_size=block_size,
                wd_factor=wd_factor, is_last_chunk=(r1 >= rows),
            )
    else:
        chunk_elems = ((_FACTORED_CHUNK_ELEMS // align_unit) or 1) * align_unit
        for off in range(0, numel, chunk_elems):
            end = min(off + chunk_elems, numel)
            chunk_n = end - off
            gc = _prep_grad_chunk_fp32(
                g_flat[off:end], sanitize_g=sanitize_g, maximize=maximize, inv_scale=inv_scale
            )
            idx = torch.arange(off, end, device=p.device)
            r_idx = torch.div(idx, cols, rounding_mode="floor")
            c_idx = idx - r_idx * cols
            denom_c = row_norm[r_idx].mul(col_sqrt[c_idx]).add_(eps_scaled).clamp_min_(effective_eps)
            _apply_factored_v_chunk_body(
                p_flat[off:end], gc, denom_c, m_buf, m_scales,
                off=off, end=end, chunk_n=chunk_n,
                is_6bit=is_6bit, is_int8=is_int8,
                beta1=beta1, step_size=step_size,
                nesterov=nesterov, cautious=cautious,
                stochastic=stochastic, block_size=block_size,
                wd_factor=wd_factor, is_last_chunk=(end >= numel),
            )


@torch.no_grad()
def _execute_factored_v_batch(
    params: List[Tensor],
    grads: List[Tensor],
    state: dict,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    decay_norm_only: bool = False,
    inv_scale: float = 1.0,
    cautious: bool = False,
    stochastic: bool = True,
) -> None:
    """Update factored-v parameters using chunked execution."""
    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    bc2_sqrt = math.sqrt(bc2)
    step_size = lr * bc2_sqrt / bc1
    wd_factor_base = max(1.0 - lr * weight_decay, 0.0) if weight_decay != 0.0 else 1.0
    need_scale = isinstance(inv_scale, Tensor) or inv_scale != 1.0

    for p, g in zip(params, grads):
        p_state = state[p]
        effective_eps = eps
        eps_scaled = effective_eps * bc2_sqrt
        apply_wd = weight_decay != 0.0 and (not decay_norm_only or p.dim() >= 2)
        _apply_factored_v_int8_update(
            p,
            g,
            p_state,
            beta1=beta1,
            beta2=beta2,
            step_size=step_size,
            eps_scaled=eps_scaled,
            effective_eps=effective_eps,
            nesterov=False,
            cautious=cautious,
            stochastic=p_state.get("stochastic_rounding", stochastic),
            block_size=p_state.get("block_size_8bit", 256),
            wd_factor=wd_factor_base if apply_wd else None,
            inv_scale=inv_scale if need_scale else None,
            sanitize_g=True,
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
    """Partition multi-tensor lists into memory-bounded chunks."""
    total_elements = sum(p.numel() for p in params)
    if len(params) <= 1 or total_elements <= max_chunk_elements:
        return [
            (
                params,
                grads,
                exp_avgs,
                exp_avgs_slow,
                exp_avg_sqs,
                max_exp_avg_sqs,
                step_sizes,
                bc2_sqrts,
            )
        ]

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
