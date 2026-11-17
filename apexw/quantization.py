"""Blockwise state quantization and stochastic rounding utilities for ApexW."""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor

_SR_POOL_CACHE: Dict[Tuple[str, int], Tuple[Tensor, int, int, int]] = {}
_SR_POOL_LEN = 524288


def _sample_sr_noise(shape: torch.Size, device: torch.device) -> Tensor:
    """Return an unbiased uniform [-0.5, 0.5) tensor from a cached pool on CPU."""
    numel = 1
    for d in shape:
        numel *= d
    if device.type != "cpu" or numel > _SR_POOL_LEN:
        return torch.empty(shape, device=device, dtype=torch.float32).uniform_(-0.5, 0.5)

    key = (device.type, device.index if device.index is not None else -1)
    entry = _SR_POOL_CACHE.get(key)
    if entry is None:
        pool = torch.empty(_SR_POOL_LEN, device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
        offset = 0
        calls = 0
        sign = 1
    else:
        pool, offset, calls, sign = entry
        if calls >= 64:
            pool.uniform_(-0.5, 0.5)
            calls = 0

    max_start = _SR_POOL_LEN - numel
    if offset > max_start:
        offset = 0
    slice_1d = pool.narrow(0, offset, numel)
    next_offset = (offset + 65537) % (max_start + 1) if max_start > 0 else 0
    _SR_POOL_CACHE[key] = (pool, next_offset, calls + 1, -sign)
    out = slice_1d.view(shape)
    return out if sign > 0 else -out


def _quantize_blockwise_int8(
    tensor,
    block_size: int = 256,
    stochastic: bool = False,
    out_q=None,
    out_scales=None,
    old_scales=None,
):
    """Quantize a float tensor into blockwise INT8 with EMA-MAX scale smoothing."""
    shape = tensor.shape
    numel = tensor.numel()
    if numel == 0:
        return (
            torch.empty_like(tensor, dtype=torch.int8),
            torch.empty((0, 1), device=tensor.device, dtype=torch.float32),
        )

    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        blocks = tensor.reshape(-1, block_size).float()
    else:
        flat = torch.nn.functional.pad(tensor.reshape(-1), (0, pad))
        blocks = flat.view(-1, block_size).float()

    abs_max = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    target_scales = (abs_max / 127.0).to(torch.float32)
    if old_scales is not None:
        old_sc = old_scales.view(-1, 1).float()
        scales = torch.maximum(target_scales, old_sc * 0.9)
    else:
        scales = target_scales
    scaled = blocks / scales
    if stochastic:
        noise = _sample_sr_noise(blocks.shape, tensor.device)
        q_blocks = scaled.add_(noise).round_().clamp_(-127, 127).to(torch.int8)
    else:
        q_blocks = scaled.round_().clamp_(-127, 127).to(torch.int8)

    q = q_blocks.view(shape) if pad == 0 else q_blocks.view(-1)[:numel].view(shape)
    if out_q is not None:
        out_q.copy_(q)
    if out_scales is not None:
        out_scales.copy_(scales)
    return (out_q if out_q is not None else q), (out_scales if out_scales is not None else scales)


def _dequantize_blockwise_int8(q, scales, block_size: int = 256):
    """Dequantize blockwise INT8 back into a float32 tensor matching q's shape."""
    shape = q.shape
    numel = q.numel()
    if numel == 0:
        return torch.empty_like(q, dtype=torch.float32)

    sc = scales.view(-1, 1) if scales.ndim < 2 else scales
    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        return (q.reshape(-1, block_size).float() * sc.float()).view(shape)

    flat_q = torch.nn.functional.pad(q.reshape(-1), (0, pad))
    deq_blocks = flat_q.view(-1, block_size).float() * sc.float()
    return deq_blocks.view(-1)[:numel].view(shape)


def _quantize_blockwise_6bit(
    tensor,
    block_size: int = 256,
    stochastic: bool = False,
    out_q=None,
    out_scales=None,
    old_scales=None,
):
    """Quantize a float tensor into packed 6-bit integers (4 values per 3 uint8 bytes)."""
    shape_len = tensor.numel()
    if shape_len == 0:
        return torch.empty(0, dtype=torch.uint8, device=tensor.device), torch.empty(
            (0, 1), device=tensor.device, dtype=torch.float32
        )

    pad = (block_size - (shape_len % block_size)) % block_size
    padded_len = shape_len + pad
    if padded_len % 4 != 0:
        rem4 = (4 - (padded_len % 4)) % 4
        pad += rem4
        padded_len += rem4

    if pad == 0:
        blocks = tensor.reshape(-1, block_size).float()
    else:
        flat = torch.nn.functional.pad(tensor.reshape(-1), (0, pad))
        blocks = flat.view(-1, block_size).float()

    abs_max = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    target_scales = (abs_max / 31.0).to(torch.float32)
    if old_scales is not None:
        old_sc = old_scales.view(-1, 1).float()
        scales = torch.maximum(target_scales, old_sc * 0.9)
    else:
        scales = target_scales

    scaled = blocks / scales
    if stochastic:
        noise = _sample_sr_noise(blocks.shape, tensor.device)
        scaled.add_(noise)
    u = scaled.round_().clamp_(-31.0, 31.0).add_(31.0).to(torch.uint8).reshape(-1, 4)

    n_quads = u.shape[0]
    if out_q is not None:
        packed = out_q.view(n_quads, 3)
    else:
        out_q = torch.empty(n_quads * 3, dtype=torch.uint8, device=tensor.device)
        packed = out_q.view(n_quads, 3)

    u0 = u[:, 0]
    u1 = u[:, 1]
    u2 = u[:, 2]
    u3 = u[:, 3]
    packed[:, 0] = (u0 & 0x3F) | ((u1 & 0x03) << 6)
    packed[:, 1] = ((u1 >> 2) & 0x0F) | ((u2 & 0x0F) << 4)
    packed[:, 2] = ((u2 >> 4) & 0x03) | ((u3 & 0x3F) << 2)

    if out_scales is not None:
        out_scales.copy_(scales)
        return out_q, out_scales
    return out_q, scales


def _dequantize_blockwise_6bit(
    q_packed,
    scales,
    numel: Optional[int] = None,
    block_size: int = 256,
):
    """Dequantize packed 6-bit uint8 (4 values per 3 bytes) back to 1D float32."""
    n_bytes = q_packed.numel()
    if n_bytes == 0:
        return torch.empty(0, dtype=torch.float32, device=q_packed.device)

    n_quads = n_bytes // 3
    total_vals = n_quads * 4
    packed = q_packed.view(n_quads, 3)
    b0 = packed[:, 0]
    b1 = packed[:, 1]
    b2 = packed[:, 2]

    unpacked = torch.empty((n_quads, 4), dtype=torch.uint8, device=q_packed.device)
    unpacked[:, 0] = b0 & 0x3F
    unpacked[:, 1] = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
    unpacked[:, 2] = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
    unpacked[:, 3] = (b2 >> 2) & 0x3F

    sc = scales.view(-1, 1).float() if scales.ndim < 2 else scales.float()
    deq = (unpacked.reshape(-1, block_size).float() - 31.0) * sc
    flat_deq = deq.reshape(-1)
    if numel is not None and numel < total_vals:
        return flat_deq[:numel]
    return flat_deq


# 15-level symmetric NF4 codebook for momentum quantization (index 7 = 0.0).
_NF4_M_CODEBOOK_RAW = torch.tensor(
    [
        -1.0000, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911,  0.0000,
         0.0911,  0.1848,  0.2844,  0.3949,
         0.5251,  0.6962,  1.0000,  1.0000,
    ],
    dtype=torch.float32,
)

_NF4_M_MIDPOINTS_RAW = (_NF4_M_CODEBOOK_RAW[:15:1] + _NF4_M_CODEBOOK_RAW[1:16:1]) / 2.0
_NF4_M_MIDPOINTS_RAW = _NF4_M_MIDPOINTS_RAW[:14].clone()

_NF4_M_LUT_CACHE: Dict[str, Tensor] = {}
_NF4_M_MIDS_CACHE: Dict[str, Tensor] = {}


def _get_nf4_m_lut(device) -> Tensor:
    key = str(device)
    lut = _NF4_M_LUT_CACHE.get(key)
    if lut is None or lut.device != device:
        lut = _NF4_M_CODEBOOK_RAW.to(device)
        _NF4_M_LUT_CACHE[key] = lut
    return lut


def _get_nf4_m_mids(device) -> Tensor:
    key = str(device)
    mids = _NF4_M_MIDS_CACHE.get(key)
    if mids is None or mids.device != device:
        mids = _NF4_M_MIDPOINTS_RAW.to(device)
        _NF4_M_MIDS_CACHE[key] = mids
    return mids


_NF4_PAIR_LUT_RAW = torch.zeros(256, 2, dtype=torch.float32)
for _b in range(256):
    _NF4_PAIR_LUT_RAW[_b, 0] = _NF4_M_CODEBOOK_RAW[_b >> 4]
    _NF4_PAIR_LUT_RAW[_b, 1] = _NF4_M_CODEBOOK_RAW[_b & 0x0F]

_NF4_PAIR_LUT_CACHE: Dict[str, Tensor] = {}


def _get_nf4_pair_lut(device) -> Tensor:
    key = str(device)
    lut = _NF4_PAIR_LUT_CACHE.get(key)
    if lut is None or lut.device != device:
        lut = _NF4_PAIR_LUT_RAW.to(device)
        _NF4_PAIR_LUT_CACHE[key] = lut
    return lut


def _quantize_blockwise_int4(
    tensor,
    block_size: int = 256,
    stochastic: bool = False,
    out_q=None,
    out_scales=None,
    old_scales=None,
):
    """Quantize a float tensor into blockwise symmetric NF4 INT4 (packed 2 values per byte)."""
    lut = _get_nf4_m_lut(tensor.device)
    numel = tensor.numel()
    if numel == 0:
        return (
            torch.empty(0, dtype=torch.uint8, device=tensor.device),
            torch.empty((0, 1), device=tensor.device, dtype=torch.float32),
        )

    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        blocks = tensor.reshape(-1, block_size).float()
    else:
        flat = torch.nn.functional.pad(tensor.reshape(-1), (0, pad))
        blocks = flat.view(-1, block_size).float()

    abs_max = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    if old_scales is not None:
        old_sc = old_scales.view(-1, 1).float()
        abs_max = torch.maximum(abs_max, old_sc * 0.9)
    scales = abs_max.to(torch.float32)

    normed = (blocks / scales).clamp_(-1.0, 1.0)
    flat_normed = normed.reshape(-1)
    if stochastic:
        lut_15 = lut[:15]
        hi = torch.searchsorted(lut_15, flat_normed, right=False).clamp_(0, 14)
        lo = (hi - 1).clamp_min(0)
        c_lo = lut_15[lo]
        c_hi = lut_15[hi]
        gap = c_hi - c_lo
        prob_hi = torch.where(
            gap > 0, (flat_normed - c_lo) / gap.clamp_min(1e-12), torch.ones_like(flat_normed)
        )
        rand = _sample_sr_noise(flat_normed.shape, tensor.device) + 0.5
        q_indices = torch.where(rand < prob_hi, hi, lo).to(torch.uint8)
    else:
        mids = _get_nf4_m_mids(tensor.device)
        q_indices = torch.searchsorted(mids, flat_normed).clamp_(0, 14).to(torch.uint8)
    q_indices = q_indices.view(blocks.shape)

    flat_q = q_indices.view(-1)[:numel]
    if numel % 2 != 0:
        flat_q = torch.nn.functional.pad(flat_q, (0, 1), value=7)
    pairs = flat_q.view(-1, 2)
    packed = ((pairs[:, 0] << 4) | pairs[:, 1]).to(torch.uint8)

    if out_q is not None:
        out_q.copy_(packed)
    if out_scales is not None:
        out_scales.copy_(scales)
    return (out_q if out_q is not None else packed), (out_scales if out_scales is not None else scales)


def _dequantize_blockwise_int4(q_packed, scales, numel: int, block_size: int = 256):
    """Dequantize packed NF4 INT4 momentum back to a flat float32 tensor of length `numel`."""
    if numel == 0:
        return torch.empty(0, dtype=torch.float32, device=q_packed.device)

    pair_lut = _get_nf4_pair_lut(q_packed.device)
    sc = scales.view(-1, 1) if scales.ndim < 2 else scales

    flat = pair_lut[q_packed.long()].reshape(-1)
    if numel < flat.numel():
        flat = flat[:numel]

    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        return (flat.view(-1, block_size) * sc.float()).view(-1)

    flat = torch.nn.functional.pad(flat, (0, pad))
    deq_blocks = flat.view(-1, block_size) * sc.float()
    return deq_blocks.view(-1)[:numel]


_V_DYNAMIC_RANGE_BITS = 24.0
_V_SCALE_FACTOR = 254.0 / _V_DYNAMIC_RANGE_BITS
_V_INV_SCALE_FACTOR = _V_DYNAMIC_RANGE_BITS / 254.0
_V_MIN_RATIO = 2.0 ** (-_V_DYNAMIC_RANGE_BITS)

_V_LOG_LUT = torch.zeros(256, dtype=torch.float32)
for _q in range(1, 256):
    _V_LOG_LUT[_q] = 2.0 ** ((_q - 1) * _V_INV_SCALE_FACTOR - _V_DYNAMIC_RANGE_BITS)

_V_LOG_LUT_CACHE: Dict[str, Tensor] = {}


def _get_v_log_lut(device) -> Tensor:
    key = str(device)
    lut = _V_LOG_LUT_CACHE.get(key)
    if lut is None or lut.device != device:
        lut = _V_LOG_LUT.to(device)
        _V_LOG_LUT_CACHE[key] = lut
    return lut


def _quantize_blockwise_uint8(
    tensor,
    block_size: int = 256,
    stochastic: bool = True,
    out_q=None,
    out_scales=None,
    old_scales=None,
):
    """Quantize a non-negative variance tensor into blockwise 24-octave logarithmic UINT8."""
    shape = tensor.shape
    numel = tensor.numel()
    if numel == 0:
        return (
            torch.empty_like(tensor, dtype=torch.uint8),
            torch.empty((0, 1), device=tensor.device, dtype=torch.float32),
        )

    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        blocks = tensor.reshape(-1, block_size).float()
    else:
        flat = torch.nn.functional.pad(tensor.reshape(-1), (0, pad))
        blocks = flat.view(-1, block_size).float()

    blocks = torch.nan_to_num(blocks, nan=0.0, posinf=0.0, neginf=0.0)
    v_max = blocks.amax(dim=1, keepdim=True).clamp_min(1e-12)
    v_max = v_max.nan_to_num_(nan=1.0, posinf=1.0)
    if old_scales is not None:
        old_sc = old_scales.view(-1, 1).float()
        v_max = torch.maximum(v_max, old_sc * 0.95)

    r = (blocks / v_max).clamp_min(_V_MIN_RATIO)
    q_float = torch.log2(r).mul_(_V_SCALE_FACTOR).add_(255.0)
    if stochastic:
        noise = _sample_sr_noise(blocks.shape, tensor.device)
        q_calc = q_float.add_(noise).round_()
    else:
        q_calc = q_float.round_()

    q_calc.clamp_(1.0, 255.0)
    q_blocks = q_calc.to(torch.uint8)
    q_blocks.masked_fill_(blocks <= 0, 0)

    q = q_blocks.view(shape) if pad == 0 else q_blocks.view(-1)[:numel].view(shape)
    if out_q is not None:
        out_q.copy_(q)
    if out_scales is not None:
        out_scales.copy_(v_max)
    return (out_q if out_q is not None else q), (out_scales if out_scales is not None else v_max)


def _dequantize_blockwise_uint8(q, scales, block_size: int = 256):
    """Dequantize blockwise logarithmic UINT8 back into a float32 tensor matching q's shape."""
    shape = q.shape
    numel = q.numel()
    if numel == 0:
        return torch.empty_like(q, dtype=torch.float32)

    lut = _get_v_log_lut(q.device)
    sc = scales.view(-1, 1) if scales.ndim < 2 else scales
    pad = (block_size - (numel % block_size)) % block_size
    if pad == 0:
        return (lut[q.reshape(-1, block_size).long()] * sc.float()).view(shape)

    flat_q = torch.nn.functional.pad(q.reshape(-1), (0, pad))
    blocks = flat_q.view(-1, block_size).long()
    deq_blocks = lut[blocks] * sc.float()
    return deq_blocks.view(-1)[:numel].view(shape)


def _copy_fp32_to_16bit_sr_(p: Tensor, p_f: Tensor, stochastic: bool = True) -> None:
    """In-place store from FP32 tensor p_f into 16-bit parameter p with IEEE-754 stochastic rounding."""
    if p.dtype == torch.float32:
        if p_f is not p:
            p.copy_(p_f)
        return
    if not stochastic or p.dtype not in (torch.float16, torch.bfloat16) or p_f.device.type != p.device.type:
        if p.dtype == torch.float16:
            p_f = p_f.clamp(-65504.0, 65504.0)
        p.copy_(p_f.to(p.dtype))
        return
    try:
        if p.dtype == torch.float16:
            p_c = p_f.clamp(-65504.0, 65504.0).contiguous()
            u = p_c.view(torch.int32)
            noise = torch.randint(0, 8192, p_c.shape, dtype=torch.int32, device=p_c.device)
            mag = (u & 0x7FFFFFFF) + noise
            u_sr = ((u & ~0x7FFFFFFF) | (mag & 0x7FFFFFFF)) & ~0x1FFF
            res_f = torch.where(p_c.abs() >= 6.103515625e-5, u_sr.view(torch.float32), p_c)
            p.copy_(res_f.to(torch.float16))
        else:
            p_c = p_f.contiguous()
            u = p_c.view(torch.int32)
            noise = torch.randint(0, 65536, p_c.shape, dtype=torch.int32, device=p_c.device)
            mag = (u & 0x7FFFFFFF) + noise
            u_sr = ((u & ~0x7FFFFFFF) | (mag & 0x7FFFFFFF)) & ~0xFFFF
            res_f = torch.where((u & 0x7FFFFFFF) < 0x7F7F0000, u_sr.view(torch.float32), p_c)
            p.copy_(res_f.to(torch.bfloat16))
    except Exception:
        if p.dtype == torch.float16:
            p_f = p_f.clamp(-65504.0, 65504.0)
        p.copy_(p_f.to(p.dtype))
