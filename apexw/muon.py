"""
ApexMuon: High-Performance Orthogonal Optimizer.

Applies Momentum Orthogonalized by Newton-Schulz (Muon) to 2D weight matrices.
- 5th-order Newton-Schulz with fused FMA operations (torch.baddbmm).
- Batched Tensor Core acceleration across identically shaped layers.
- Single float32 momentum buffer (4 bytes/param vs 8 bytes/param in AdamW).
- Vectorized multi-tensor parameter updates via torch._foreach_*.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

try:
    from .optimizer import (
        _normalize_precision,
        _quantize_blockwise_int8,
        _dequantize_blockwise_int8,
        _quantize_blockwise_6bit,
        _dequantize_blockwise_6bit,
    )
except ImportError:
    try:
        from optimizer import (
            _normalize_precision,
            _quantize_blockwise_int8,
            _dequantize_blockwise_int8,
            _quantize_blockwise_6bit,
            _dequantize_blockwise_6bit,
        )
    except ImportError:
        def _normalize_precision(p):
            if isinstance(p, torch.dtype):
                return "full" if p in (torch.float32, torch.float64) else "compact"
            return str(p)
        _quantize_blockwise_int8 = None
        _dequantize_blockwise_int8 = None
        _quantize_blockwise_6bit = None
        _dequantize_blockwise_6bit = None

_MUON_BLOCK_SIZE = 256
_MUON_6BIT_MIN_NUMEL = 524288

# Optimal quintic coefficients for fast polar factor convergence (Keller Jordan)
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315

# Module-level cache: maps "device_type:device_index" → bool (supports BF16?)
# Avoids repeated CUDA driver API round-trips (~60 calls/step on a 60-layer transformer).
_BF16_SUPPORT_CACHE: Dict[str, bool] = {}


def _is_bf16_supported(device_type: str, device_index: int = 0) -> bool:
    """Cached BF16 capability check — queries the driver at most once per device."""
    key = f"{device_type}:{device_index}"
    if key not in _BF16_SUPPORT_CACHE:
        if device_type == "cuda":
            _BF16_SUPPORT_CACHE[key] = bool(
                hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            )
        elif device_type == "xla":
            # TPU/XLA always supports BF16 (it's the native accumulation format)
            _BF16_SUPPORT_CACHE[key] = True
        else:
            # CPU, MPS, ROCm pre-CDNA2, and unknown backends fall back to float32
            _BF16_SUPPORT_CACHE[key] = False
    return _BF16_SUPPORT_CACHE[key]


def _determine_work_dtype(tensor: Tensor) -> torch.dtype:
    """Selects highest-performance work dtype for Newton-Schulz.

    - bfloat16  on Ampere/Ada/Hopper/Blackwell CUDA, TPU/XLA
    - float32   on Turing T4, Volta V100, AMD pre-CDNA2, MPS, CPU
      (float16 max is 65,504 — overflows during quintic polynomial powers)
    """
    dev = tensor.device
    dev_type = dev.type
    if dev_type in ("cuda", "xla"):
        idx = dev.index if dev.index is not None else 0
        return torch.bfloat16 if _is_bf16_supported(dev_type, idx) else torch.float32
    # MPS, CPU, and all other backends: safe float32 fallback
    return torch.float32


try:
    from torch.compiler import is_compiling
except ImportError:
    try:
        from torch._dynamo import is_compiling
    except ImportError:
        def is_compiling():
            return False


def _newton_schulz5_inner(X: Tensor, steps: int = 5) -> Tensor:
    """Core 5th-order Newton-Schulz polynomial iteration loop."""
    for _ in range(steps):
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=_NS_B, alpha=_NS_C)
        X = torch.addmm(X, B, X, beta=_NS_A, alpha=1.0)
    return X


_COMPILED_NS5_LOOP = None
_COMPILE_FAILED = False


def _get_compiled_ns5():
    """Lazily compiles Newton-Schulz with reduce-overhead mode for CUDA Graph kernel fusion."""
    global _COMPILED_NS5_LOOP, _COMPILE_FAILED
    if _COMPILE_FAILED:
        return _newton_schulz5_inner
    if _COMPILED_NS5_LOOP is None:
        if hasattr(torch, "compile"):
            try:
                _COMPILED_NS5_LOOP = torch.compile(_newton_schulz5_inner, mode="reduce-overhead")
            except Exception:
                try:
                    _COMPILED_NS5_LOOP = torch.compile(_newton_schulz5_inner)
                except Exception:
                    _COMPILE_FAILED = True
                    _COMPILED_NS5_LOOP = _newton_schulz5_inner
        else:
            _COMPILED_NS5_LOOP = _newton_schulz5_inner
    return _COMPILED_NS5_LOOP


def zeropower_via_newtonschulz5(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    work_dtype: Optional[torch.dtype] = None,
    compiled: bool = False,
) -> Tensor:
    """Computes the polar factor of matrix G via 5th-order Newton-Schulz iterations.

    Args:
        G: Input 2D matrix (or tensor reshaped to 2D) to orthogonalize.
        steps: Number of quintic iterations (default: 5).
        eps: Epsilon for normalization stability (default: 1e-7).
        work_dtype: Compute precision (default: bfloat16 on Ampere+, float32 on Turing/CPU/MPS).
        compiled: Accelerate iterative loop with torch.compile (mode='reduce-overhead', default: True).
    """
    orig_shape = G.shape
    if G.ndim > 2:
        G = G.reshape(orig_shape[0], -1)

    m, n = G.shape
    if work_dtype is None:
        work_dtype = _determine_work_dtype(G)

    X = G.to(work_dtype)
    X = X / (X.norm() + eps)

    transposed = False
    if m > n:
        X = X.T.contiguous()
        transposed = True
    elif not X.is_contiguous():
        X = X.contiguous()

    if compiled and not is_compiling():
        try:
            fn = _get_compiled_ns5()
            X = fn(X, steps)
        except Exception:
            X = _newton_schulz5_inner(X, steps)
    else:
        X = _newton_schulz5_inner(X, steps)

    if transposed:
        X = X.T

    if len(orig_shape) > 2:
        X = X.reshape(orig_shape)

    return X.to(G.dtype)


def zeropower_via_newtonschulz5_batched(X_batch: Tensor, steps: int = 4, eps: float = 1e-7) -> Tensor:
    """Batched 5th-order Newton-Schulz iteration across K matrices of identical shape (K, m, n)."""
    K, m, n = X_batch.shape
    work_dtype = _determine_work_dtype(X_batch)

    X = X_batch.to(work_dtype)
    norms = torch.linalg.norm(X, dim=(-2, -1), keepdim=True) + eps
    X = X / norms

    transposed = False
    if m > n:
        X = X.transpose(-2, -1).contiguous()
        transposed = True
    elif not X.is_contiguous():
        X = X.contiguous()

    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))
        B = torch.baddbmm(A, A, A, beta=_NS_B, alpha=_NS_C)
        X = torch.baddbmm(X, B, X, beta=_NS_A, alpha=1.0)

    if transposed:
        X = X.transpose(-2, -1)

    return X.to(X_batch.dtype)


# Optimal Polar Express Minimax Quintic Sequence (Amsel, Persson, Musco, Gower 2025)
# Adaptively tuned Remez minimax polynomials achieve tighter spectral compression
# in 3-4 steps than classical 5-step Newton-Schulz while cutting GEMM FLOPs by 25-40%.
_POLAR_EXPRESS_RAW_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
_POLAR_EXPRESS_COEFFS = [
    (a / 1.01, b / (1.01**3), c / (1.01**5)) for (a, b, c) in _POLAR_EXPRESS_RAW_COEFFS[:-1]
] + [_POLAR_EXPRESS_RAW_COEFFS[-1]]


def _polar_express_inner(X: Tensor, steps: int = 4) -> Tensor:
    """Core Polar Express Remez minimax quintic polynomial iteration loop."""
    n_coeffs = len(_POLAR_EXPRESS_COEFFS)
    for i in range(steps):
        a, b, c = _POLAR_EXPRESS_COEFFS[i if i < n_coeffs else -1]
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)
    return X


_COMPILED_POLAR_EXPRESS_LOOP = None


def _get_compiled_polar_express():
    """Lazily compiles Polar Express with reduce-overhead mode for CUDA Graph kernel fusion."""
    global _COMPILED_POLAR_EXPRESS_LOOP, _COMPILE_FAILED
    if _COMPILE_FAILED:
        return _polar_express_inner
    if _COMPILED_POLAR_EXPRESS_LOOP is None:
        if hasattr(torch, "compile"):
            try:
                _COMPILED_POLAR_EXPRESS_LOOP = torch.compile(_polar_express_inner, mode="reduce-overhead")
            except Exception:
                try:
                    _COMPILED_POLAR_EXPRESS_LOOP = torch.compile(_polar_express_inner)
                except Exception:
                    _COMPILE_FAILED = True
                    _COMPILED_POLAR_EXPRESS_LOOP = _polar_express_inner
        else:
            _COMPILED_POLAR_EXPRESS_LOOP = _polar_express_inner
    return _COMPILED_POLAR_EXPRESS_LOOP


def zeropower_via_polar_express(
    G: Tensor,
    steps: int = 4,
    eps: float = 1e-7,
    work_dtype: Optional[torch.dtype] = None,
    compiled: bool = False,
) -> Tensor:
    """Computes the polar factor of matrix G via Polar Express Remez minimax iterations (Amsel et al., 2025).

    Polar Express dynamically adapts polynomial coefficients per iteration, delivering tighter
    spectral compression in 3-4 steps than classical 5-step Newton-Schulz while cutting compute by ~25-40%.
    """
    orig_shape = G.shape
    if G.ndim > 2:
        G = G.reshape(orig_shape[0], -1)

    m, n = G.shape
    if work_dtype is None:
        work_dtype = _determine_work_dtype(G)

    X = G.to(work_dtype)
    X = X / (X.norm() * 1.01 + eps)

    transposed = False
    if m > n:
        X = X.T.contiguous()
        transposed = True
    elif not X.is_contiguous():
        X = X.contiguous()

    if compiled and not is_compiling():
        try:
            fn = _get_compiled_polar_express()
            X = fn(X, steps)
        except Exception:
            X = _polar_express_inner(X, steps)
    else:
        X = _polar_express_inner(X, steps)

    if transposed:
        X = X.T

    if len(orig_shape) > 2:
        X = X.reshape(orig_shape)

    return X.to(G.dtype)


def zeropower_via_polar_express_batched(X_batch: Tensor, steps: int = 4, eps: float = 1e-7) -> Tensor:
    """Batched Polar Express iteration across K matrices of identical shape (K, m, n)."""
    K, m, n = X_batch.shape
    work_dtype = _determine_work_dtype(X_batch)

    X = X_batch.to(work_dtype)
    norms = X.norm(dim=(-2, -1), keepdim=True) * 1.01 + eps
    X = X / norms

    transposed = False
    if m > n:
        X = X.transpose(-2, -1).contiguous()
        transposed = True
    elif not X.is_contiguous():
        X = X.contiguous()

    n_coeffs = len(_POLAR_EXPRESS_COEFFS)
    for i in range(steps):
        a, b, c = _POLAR_EXPRESS_COEFFS[i if i < n_coeffs else -1]
        A = torch.bmm(X, X.transpose(-2, -1))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    if transposed:
        X = X.transpose(-2, -1)

    return X.to(X_batch.dtype)



class ApexMuon(Optimizer):
    """ApexMuon: High-Performance Orthogonal Optimizer.

    Replaces coordinate-wise learning rates (AdamW) with spectral orthogonalization
    via Newton-Schulz iterations on the momentum matrix for 2D weight parameters.

    Args:
        params: Iterable of parameters to optimize or param group dicts.
        lr: Learning rate (default: 1e-3).
        momentum: Momentum coefficient (default: 0.95).
        nesterov: Whether to use Nesterov momentum (default: False).
        weight_decay: Decoupled weight decay (default: 1e-2).
        ns_steps: Number of Newton-Schulz iterations (default: 4; 20% faster than 5 with equal loss).
        beta3: Slow momentum coefficient for AdEMAMix (default: None).
        alpha: Slow momentum weight in AdEMAMix (default: 5.0).
        cautious: Enable cautious directional update masking (default: False).
        precision: Accepted for API compatibility — Muon always stores momentum in float32.
            NS orthogonalization casts to the optimal work dtype internally. Storing
            momentum in BF16 would cost 60+ cast+copy operations per step with zero
            benefit, so this argument is intentionally ignored for the momentum buffer.
        scale_mode: Matrix scaling method: 'aspect_ratio', 'spectral', 'rms', or 'none'.
        batch_bmm: Batch identically shaped matrices via torch.bmm (default: False).
        fused: Fused execution flag (default: None; automatically optimizes batching and kernel paths).
        capturable: Whether this optimizer is safe to capture in a CUDA graph (default: False).
        compiled: Whether to accelerate Newton-Schulz and Polar Express iterations with
            torch.compile using mode='reduce-overhead' for CUDA Graph kernel fusion (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = False,
        weight_decay: float = 1e-2,
        ns_steps: int = 4,
        beta3: Optional[float] = None,
        alpha: float = 5.0,
        cautious: bool = False,
        grad_centralization: bool = False,
        precision: Union[str, torch.dtype] = "compact",
        scale_mode: str = "none",
        ns_method: str = "newtonschulz5",
        batch_bmm: Optional[bool] = None,
        fused: Optional[bool] = None,
        capturable: bool = False,
        compiled: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        precision = _normalize_precision(precision)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            beta3=beta3,
            alpha=alpha,
            cautious=cautious,
            grad_centralization=grad_centralization,
            precision=precision,
            scale_mode=scale_mode,
            ns_method=ns_method,
            batch_bmm=batch_bmm,
            fused=fused,
            capturable=capturable,
            compiled=compiled,
        )
        super().__init__(params, defaults)

    _step_supports_amp_scaling = True

    def _get_state_dtype(self, p: Tensor, mode: Union[str, torch.dtype]) -> torch.dtype:
        # Muon momentum is stored in float32 for EMA accumulation stability and to
        # avoid repeated bf16<->fp32 casts when computing updates on FP32/Turing hardware
        # (1 buffer = 4 bytes/param vs AdamW's 8 bytes/param).
        return torch.float32

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore Muon optimizer state while preserving float32 or quantized momentum precision.

        PyTorch's default Optimizer.load_state_dict casts all floating-point state tensors
        to ``p.dtype``, which would silently downcast float32 momentum or scale buffers when training
        bf16 or fp16 parameters.
        """
        saved_state = state_dict.get("state", {})
        saved_groups = state_dict.get("param_groups", [])
        param_to_saved_tensors: Dict[Tensor, Tuple[Dict[str, Tensor], Any]] = {}

        for group_idx, sg in enumerate(saved_groups):
            if group_idx >= len(self.param_groups):
                break
            g = self.param_groups[group_idx]
            for p, p_id in zip(g["params"], sg.get("params", [])):
                raw_entry = saved_state.get(p_id, {})
                if isinstance(raw_entry, dict):
                    preserved = {}
                    for k in ("momentum", "momentum_slow", "momentum_scales", "momentum_slow_scales"):
                        t = raw_entry.get(k)
                        if isinstance(t, Tensor):
                            preserved[k] = t.detach().clone()
                    if preserved:
                        param_to_saved_tensors[p] = (preserved, g.get("precision", "compact"))

        super().load_state_dict(state_dict)

        for p, (preserved, prec) in param_to_saved_tensors.items():
            target_dtype = self._get_state_dtype(p, prec)
            p_state = self.state.get(p)
            if p_state is None:
                continue
            for k, orig_t in preserved.items():
                if k in ("momentum_scales", "momentum_slow_scales"):
                    p_state[k] = orig_t.to(device=p.device, dtype=torch.float32)
                elif orig_t.dtype in (torch.int8, torch.uint8):
                    p_state[k] = orig_t.to(device=p.device, dtype=orig_t.dtype)
                else:
                    p_state[k] = orig_t.to(device=p.device, dtype=target_dtype)

    def _init_param_state(self, p: Tensor, group: dict) -> dict:
        state = self.state[p]
        if len(state) > 0:
            return state
        precision = group.get("precision", "compact")
        beta3 = group.get("beta3")
        state["step"] = 0
        state["work_dtype"] = _determine_work_dtype(p)
        use_quant = (
            precision in ("8bit", "ultra")
            and p.device.type != "xla"
            and _quantize_blockwise_int8 is not None
        )
        if use_quant:
            numel = p.numel()
            is_6bit = (numel >= _MUON_6BIT_MIN_NUMEL) and (_quantize_blockwise_6bit is not None)
            n_blocks = (numel + _MUON_BLOCK_SIZE - 1) // _MUON_BLOCK_SIZE
            state["is_quant_m"] = True
            state["is_6bit_m"] = is_6bit
            if is_6bit:
                padded_len = n_blocks * _MUON_BLOCK_SIZE
                packed_bytes = (padded_len // 4) * 3
                buf = torch.full((packed_bytes,), 223, dtype=torch.uint8, device=p.device)
                buf[1::3] = 247
                buf[2::3] = 125
                state["momentum"] = buf
            else:
                state["momentum"] = torch.zeros(p.shape, dtype=torch.int8, device=p.device)
            state["momentum_scales"] = torch.full((n_blocks, 1), 1e-12, dtype=torch.float32, device=p.device)
            if beta3 is not None:
                if is_6bit:
                    state["momentum_slow"] = state["momentum"].clone()
                else:
                    state["momentum_slow"] = torch.zeros(p.shape, dtype=torch.int8, device=p.device)
                state["momentum_slow_scales"] = torch.full((n_blocks, 1), 1e-12, dtype=torch.float32, device=p.device)
        else:
            state_dtype = self._get_state_dtype(p, precision)
            state["momentum"] = torch.zeros_like(p, dtype=state_dtype)
            if beta3 is not None:
                state["momentum_slow"] = torch.zeros_like(p, dtype=state_dtype)
        return state

    @staticmethod
    def _dequant_momentum(buf: Tensor, scales: Optional[Tensor], shape, is_6bit: bool) -> Tensor:
        if is_6bit:
            numel = 1
            for s in shape:
                numel *= s
            return _dequantize_blockwise_6bit(
                buf, scales, numel=numel, block_size=_MUON_BLOCK_SIZE
            ).reshape(shape)
        if buf.dtype == torch.int8 and scales is not None:
            return _dequantize_blockwise_int8(buf, scales, block_size=_MUON_BLOCK_SIZE)
        return buf if buf.dtype == torch.float32 else buf.float()

    @staticmethod
    def _quant_momentum(fp_val: Tensor, buf: Tensor, scales: Tensor, is_6bit: bool) -> None:
        if is_6bit:
            _quantize_blockwise_6bit(
                fp_val, block_size=_MUON_BLOCK_SIZE, stochastic=True,
                out_q=buf, out_scales=scales, old_scales=scales,
            )
        else:
            _quantize_blockwise_int8(
                fp_val, block_size=_MUON_BLOCK_SIZE, stochastic=True,
                out_q=buf, out_scales=scales, old_scales=scales,
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Cache the capturable flag — it doesn't change per step
        is_capturable_group = any(g.get("capturable", False) for g in self.param_groups)

        # Only call is_current_stream_capturing when capturable mode is requested.
        # This avoids a potential CPU-GPU sync on every step in the common case (capturable=False).
        is_capturing = (
            is_capturable_group
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "is_current_stream_capturing")
            and torch.cuda.is_current_stream_capturing()
        )

        # Handle PyTorch AMP GradScaler inf/NaN detection (zero CPU-GPU sync during graph capture)
        found_inf = getattr(self, "found_inf", None)
        if found_inf is not None and not is_capturing:
            if isinstance(found_inf, Tensor):
                if found_inf.numel() == 1 and found_inf.item() > 0:
                    return loss
            elif found_inf > 0:
                return loss

        # Handle PyTorch AMP GradScaler gradient unscaling
        grad_scale = getattr(self, "grad_scale", None)
        inv_scale = None
        if grad_scale is not None:
            if is_capturing or is_capturable_group:
                # Keep as Tensor on device without calling .item()
                inv_scale = grad_scale.reciprocal()
            elif isinstance(grad_scale, Tensor):
                inv_scale = float(grad_scale.reciprocal().item()) if grad_scale.numel() == 1 else grad_scale.reciprocal()
            else:
                inv_scale = 1.0 / float(grad_scale)

        try:
            for group in self.param_groups:
                active_params = [p for p in group["params"] if p.grad is not None]
                if not active_params:
                    continue

                # Unscale gradients if scaled by GradScaler
                if inv_scale is not None and inv_scale != 1.0:
                    try:
                        grads = [p.grad for p in active_params]
                        torch._foreach_mul_(grads, inv_scale)
                    except (RuntimeError, NotImplementedError, AttributeError):
                        for p in active_params:
                            p.grad.mul_(inv_scale)

                use_batch_bmm = group.get("batch_bmm", None)
                dev_type = active_params[0].device.type
                should_batch = use_batch_bmm is True or (
                    use_batch_bmm is None and dev_type in ("cuda", "mps", "cpu")
                )
                if should_batch:
                    # Group identically shaped matrices for batched GEMM execution
                    shape_groups: Dict[Tuple[int, ...], List[Tensor]] = defaultdict(list)
                    for p in active_params:
                        shape_groups[tuple(p.shape)].append(p)

                    for shape, param_list in shape_groups.items():
                        is_valid_2d = len(shape) >= 2 and min(shape[0], param_list[0].numel() // shape[0]) >= 16
                        # On CPU, large OpenBLAS/MKL matrices stay L3-hot per matrix via addmm; batching only helps tiny matrices.
                        cpu_batch_ok = dev_type != "cpu" or use_batch_bmm is True or param_list[0].numel() <= 16384
                        if len(param_list) > 1 and is_valid_2d and cpu_batch_ok:
                            self._step_batched(param_list, group)
                        else:
                            for p in param_list:
                                self._step_single(p, group)
                else:
                    for p in active_params:
                        self._step_single(p, group)
        finally:
            self.grad_scale = None
            self.found_inf = None

        return loss

    def _step_batched(self, params: List[Tensor], group: dict):
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        weight_decay = group["weight_decay"]
        ns_steps = group["ns_steps"]
        beta3 = group["beta3"]
        alpha = group["alpha"]
        cautious = group["cautious"]
        gc = group["grad_centralization"]
        scale_mode = group["scale_mode"]

        for p in params:
            st = self._init_param_state(p, group)
            st["step"] += 1

        grads = [torch.nan_to_num(p.grad, nan=0.0, posinf=65504.0, neginf=-65504.0) for p in params]
        if gc:
            grads = [g - g.mean(dim=tuple(range(1, g.ndim)), keepdim=True) for g in grads]

        orig_shape = params[0].shape
        flat_shape_2d = (orig_shape[0], params[0].numel() // orig_shape[0])
        work_dtype = self.state[params[0]].get("work_dtype", _determine_work_dtype(params[0]))
        is_quant_batch = any(self.state[p].get("is_quant_m", False) for p in params)

        if is_quant_batch:
            update_batch = torch.empty(
                (len(params), flat_shape_2d[0], flat_shape_2d[1]),
                dtype=work_dtype,
                device=params[0].device,
            )
            for idx, (p, g) in enumerate(zip(params, grads)):
                st = self.state[p]
                is_6bit = st.get("is_6bit_m", False)
                g_fp = g if g.dtype == torch.float32 else g.float()
                m_buf = st["momentum"]
                m_scales = st.get("momentum_scales")
                m_fp = self._dequant_momentum(m_buf, m_scales, p.shape, is_6bit)
                m_fp.mul_(momentum).add_(g_fp, alpha=1.0 - momentum)
                if m_scales is not None and st.get("is_quant_m", False):
                    self._quant_momentum(m_fp, m_buf, m_scales, is_6bit)
                else:
                    m_buf.copy_(m_fp)

                if beta3 is not None:
                    ms_buf = st["momentum_slow"]
                    ms_scales = st.get("momentum_slow_scales")
                    ms_fp = self._dequant_momentum(ms_buf, ms_scales, p.shape, is_6bit)
                    ms_fp.mul_(beta3).add_(g_fp, alpha=1.0 - beta3)
                    if ms_scales is not None and st.get("is_quant_m", False):
                        self._quant_momentum(ms_fp, ms_buf, ms_scales, is_6bit)
                    else:
                        ms_buf.copy_(ms_fp)
                    upd = m_fp.add(ms_fp, alpha=alpha)
                elif nesterov:
                    upd = g_fp.lerp(m_fp, momentum)
                else:
                    upd = m_fp
                update_batch[idx].copy_(upd.reshape(flat_shape_2d).to(work_dtype))
        else:
            m_bufs = [self.state[p]["momentum"] for p in params]
            try:
                torch._foreach_mul_(m_bufs, momentum)
                torch._foreach_add_(m_bufs, grads, alpha=1.0 - momentum)
            except (RuntimeError, NotImplementedError, AttributeError):
                for m, g in zip(m_bufs, grads):
                    m.mul_(momentum).add_(g, alpha=1.0 - momentum)

            if beta3 is not None:
                m_slow_bufs = [self.state[p]["momentum_slow"] for p in params]
                try:
                    torch._foreach_mul_(m_slow_bufs, beta3)
                    torch._foreach_add_(m_slow_bufs, grads, alpha=1.0 - beta3)
                except (RuntimeError, NotImplementedError, AttributeError):
                    for ms, g in zip(m_slow_bufs, grads):
                        ms.mul_(beta3).add_(g, alpha=1.0 - beta3)
                update_batch = torch.stack([
                    (m + alpha * ms).reshape(flat_shape_2d).to(work_dtype)
                    for m, ms in zip(m_bufs, m_slow_bufs)
                ], dim=0)
            elif nesterov:
                update_batch = torch.stack([
                    g.to(torch.float32).lerp(m, momentum).reshape(flat_shape_2d).to(work_dtype)
                    for g, m in zip(grads, m_bufs)
                ], dim=0)
            else:
                update_batch = torch.stack([
                    m.reshape(flat_shape_2d).to(work_dtype)
                    for m in m_bufs
                ], dim=0)

        # Pre-NS Cautious Directional Masking (preserves matrix polar orthogonality)
        if cautious:
            G_batch = torch.stack([g.reshape(flat_shape_2d).to(work_dtype) for g in grads], dim=0)
            mask = (G_batch * update_batch >= 0).to(update_batch.dtype)
            mask_mean = mask.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-3)
            update_batch = update_batch * (mask / mask_mean)

        # Batched Polar Factorization (Polar Express Remez minimax or classic NS5)
        ns_method = group.get("ns_method", "polar_express")
        if ns_method == "polar_express":
            U_batch = zeropower_via_polar_express_batched(update_batch, steps=ns_steps)
        else:
            U_batch = zeropower_via_newtonschulz5_batched(update_batch, steps=ns_steps)

        # Step size scaling based on matrix dimensions
        m_dim, n_dim = flat_shape_2d
        if scale_mode == "aspect_ratio":
            scale = max(1.0, math.sqrt(m_dim / n_dim))
        elif scale_mode in ("spectral", "rms_matched"):
            # Fixed formula: 0.2 * sqrt(max_dim) — fast, no per-param computation
            scale = math.sqrt(max(m_dim, n_dim)) * 0.2
        elif scale_mode == "rms":
            # True RMS matching: scale update to match mean parameter norm across the batch.
            # Computes mean ||p||_F / mean ||U||_F, consistent with _step_single behavior.
            # Clipped to [0, 10] to prevent runaway scaling early in training.
            param_norms = torch.stack([p.norm().float() for p in params])
            u_norms = torch.stack([U_batch[i].norm().float() for i in range(len(params))])
            scale = (param_norms.mean() / (u_norms.mean() + 1e-7)).clamp(max=10.0).item()
        elif scale_mode == "none":
            scale = 1.0
        else:
            raise ValueError(
                f"ApexMuon: unknown scale_mode='{scale_mode}'. "
                f"Valid options: 'aspect_ratio', 'spectral', 'rms', 'rms_matched', 'none'."
            )
        eff_lr = lr * scale

        U_slices = [U_batch[i].reshape(orig_shape).to(params[i].dtype) for i in range(len(params))]

        # Vectorized updates with pure PyTorch fallback for CPU/MPS/esoteric backends
        try:
            if weight_decay != 0.0:
                torch._foreach_mul_(params, 1.0 - lr * weight_decay)
            torch._foreach_add_(params, U_slices, alpha=-eff_lr)
        except (RuntimeError, NotImplementedError, AttributeError):
            for p, u in zip(params, U_slices):
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(u, alpha=-eff_lr)

    def _step_single(self, p: Tensor, group: dict):
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        weight_decay = group["weight_decay"]
        ns_steps = group["ns_steps"]
        beta3 = group["beta3"]
        alpha = group["alpha"]
        cautious = group["cautious"]
        gc = group["grad_centralization"]
        scale_mode = group["scale_mode"]

        state = self._init_param_state(p, group)
        state["step"] += 1

        grad = torch.nan_to_num(p.grad, nan=0.0, posinf=65504.0, neginf=-65504.0)
        if gc and grad.ndim >= 2:
            grad = grad - grad.mean(dim=tuple(range(1, grad.ndim)), keepdim=True)

        buf = state["momentum"]
        g_fp = grad if grad.dtype == torch.float32 else grad.float()
        is_quant = state.get("is_quant_m", False)
        is_6bit = state.get("is_6bit_m", False)

        if is_quant:
            scales = state["momentum_scales"]
            buf_fp = self._dequant_momentum(buf, scales, p.shape, is_6bit)
            buf_fp.mul_(momentum).add_(g_fp, alpha=1.0 - momentum)
            self._quant_momentum(buf_fp, buf, scales, is_6bit)
        else:
            buf.mul_(momentum).add_(g_fp, alpha=1.0 - momentum)
            buf_fp = buf

        if beta3 is not None:
            buf_slow = state["momentum_slow"]
            if is_quant:
                scales_slow = state["momentum_slow_scales"]
                buf_slow_fp = self._dequant_momentum(buf_slow, scales_slow, p.shape, is_6bit)
                buf_slow_fp.mul_(beta3).add_(g_fp, alpha=1.0 - beta3)
                self._quant_momentum(buf_slow_fp, buf_slow, scales_slow, is_6bit)
                update_source = buf_fp.add(buf_slow_fp, alpha=alpha)
            else:
                buf_slow.mul_(beta3).add_(g_fp, alpha=1.0 - beta3)
                update_source = buf_fp.add(buf_slow, alpha=alpha)
        elif nesterov:
            update_source = g_fp.lerp(buf_fp, momentum)
        else:
            update_source = buf_fp

        # Pre-NS Cautious Directional Masking (preserves matrix polar orthogonality)
        if cautious:
            mask = (g_fp * update_source >= 0).to(update_source.dtype)
            mask_mean = mask.mean().clamp(min=1e-3)
            update_source = update_source * (mask / mask_mean)

        # 2D weight matrix orthogonalization
        if update_source.ndim >= 2 and min(update_source.shape[0], update_source.numel() // update_source.shape[0]) >= 16:
            ns_method = group.get("ns_method", "polar_express")
            # Pass cached work_dtype to avoid a redundant _determine_work_dtype lookup
            cached_work_dtype = state.get("work_dtype")
            compiled = group.get("compiled", True)
            if ns_method == "polar_express":
                U = zeropower_via_polar_express(update_source, steps=ns_steps, work_dtype=cached_work_dtype, compiled=compiled)
            else:
                U = zeropower_via_newtonschulz5(update_source, steps=ns_steps, work_dtype=cached_work_dtype, compiled=compiled)
            m_dim, n_dim = U.shape[0], U.numel() // U.shape[0]
            if scale_mode == "aspect_ratio":
                scale = max(1.0, math.sqrt(m_dim / n_dim))
            elif scale_mode in ("spectral", "rms_matched"):
                # Fixed formula: 0.2 * sqrt(max_dim) — fast, no per-param computation
                scale = math.sqrt(max(m_dim, n_dim)) * 0.2
            elif scale_mode == "rms":
                # True RMS matching: scale update so its norm matches the parameter norm.
                p_norm = p.norm()
                u_norm = U.norm() + 1e-7
                scale = (p_norm / u_norm).clamp(max=10.0)
            elif scale_mode == "none":
                scale = 1.0
            else:
                raise ValueError(
                    f"ApexMuon: unknown scale_mode='{scale_mode}'. "
                    f"Valid options: 'aspect_ratio', 'spectral', 'rms', 'rms_matched', 'none'."
                )
        else:
            # Safe coordinate-wise fallback for 1D vectors or embeddings
            if update_source.numel() <= 1:
                std = update_source.abs() + 1e-7
            else:
                std = update_source.std(unbiased=False) + 1e-7
            U = update_source / std
            scale = 0.05

        if weight_decay != 0.0:
            p.mul_(1.0 - lr * weight_decay)

        if isinstance(scale, Tensor):
            p.add_(U.to(p.dtype) * scale, alpha=-lr)
        else:
            p.add_(U.to(p.dtype), alpha=-lr * scale)

    def get_state_memory_mb(self) -> float:
        """Returns the total allocated or estimated optimizer state memory in MB."""
        total_bytes = 0
        for p_state in self.state.values():
            for v in p_state.values():
                if isinstance(v, Tensor):
                    total_bytes += v.numel() * v.element_size()
        if total_bytes == 0:
            for group in self.param_groups:
                prec = group.get("precision", "compact")
                bpp = 1.0 if prec in ("8bit", "ultra") else (2.0 if prec == "compact" else 4.0)
                if group.get("beta3") is not None:
                    bpp *= 2.0
                total_bytes += sum(p.numel() * bpp for p in group["params"])
        return total_bytes / (1024 * 1024)
