import math
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


def _get_compact_dtype(p: Tensor) -> torch.dtype:
    """Determine optimal 16-bit storage dtype based on hardware and tensor device.

    Complex tensors must retain their complex dtype because PyTorch does not
    support 16-bit complex types or mixed-precision view_as_real operations.
    On Apple Silicon (MPS), checks if Metal supports bfloat16, falling back
    safely to float16 with dynamic scaling to prevent hardware errors.
    On CUDA/ROCm/CPU, uses bfloat16 for full dynamic range protection.
    """
    if torch.is_complex(p):
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
            square (default: (0.9, 0.999)). If a 3-tuple is provided, the 3rd element
            is used as ``beta3`` for AdEMAMix.
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
        dynamic_scaling (bool, optional): Whether to apply dynamic range scaling to 16-bit
            variance states to prevent underflow during extended training (default: True).
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

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Union[Tuple[float, float], Tuple[float, float, float]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        nesterov: bool = False,
        cautious: bool = False,
        grad_centralization: bool = False,
        decay_norm_only: bool = True,   # Automatically protect 1D norms (RMSNorm, LayerNorm) & biases
        clip_grad_norm: Optional[float] = None,
        beta3: Optional[float] = None,
        alpha: float = 5.0,
        lion: bool = False,
        fused: Optional[bool] = None,
        precision: str = "auto",     # "auto" (smart partitioning), "compact" (bfloat16), or "full" (float32)
        threshold: int = 65536,      # Size threshold: 2D+ or >= 64K tensors get compact BF16, 1D norms get FP32
        maximize: bool = False,
        foreach: Optional[bool] = None,  # None = auto-detect: True on CUDA/XPU/ROCm, fast single on CPU
        capturable: bool = False,
        dynamic_scaling: bool = True, # Layer-wise dynamic range scaling to protect precision during long runs
        differentiable: bool = False,
        backend: Optional[str] = None, # None / "auto", "triton", "cuda", "mps", "foreach", "xla"
        auto_pack: bool = False,       # Automatically pack small tensors into contiguous flat buffers
        pack_threshold: int = 50000,   # Maximum numel for tensors to be auto-packed
    ):
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
        if precision not in ("auto", "compact", "full"):
            raise ValueError(f"Invalid precision mode '{precision}'. Must be 'auto', 'compact', or 'full'.")
        if backend is not None and backend not in ("auto", "triton", "cuda", "mps", "foreach", "_multi_tensor", "xla"):
            raise ValueError(f"Invalid backend '{backend}'. Must be 'auto', 'triton', 'cuda', 'mps', 'foreach', or 'xla'.")

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
            layer_scale_v=1.0,
            sqrt_layer_scale_v=1.0,
            differentiable=differentiable,
            backend=backend,
            auto_pack=auto_pack,
            pack_threshold=pack_threshold,
        )
        super().__init__(params, defaults)
        self._workspace_pool = {}
        self.auto_pack = auto_pack
        self.pack_threshold = pack_threshold
        self.packed_param_groups: List[Dict[str, Any]] = []
        self._param_to_packed: Dict[int, Tuple] = {}
        if self.auto_pack:
            self._setup_auto_packing()

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
            group.setdefault("layer_scale_v", 1.0)
            group.setdefault("sqrt_layer_scale_v", 1.0)
            group.setdefault("clip_grad_norm", None)
            group.setdefault("beta3", None)
            group.setdefault("alpha", 5.0)
            group.setdefault("lion", False)
            group.setdefault("fused", None)
            group.setdefault("precision", "auto")
            group.setdefault("threshold", 65536)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("backend", None)

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
        for p, s in self.state.items():
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

        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                if p.grad.is_sparse:
                    raise RuntimeError("ApexW does not support sparse gradients")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(),
                            device=p.device,
                        )
                        if (group["capturable"] or p.device.type == "xla" or group.get("backend") == "xla")
                        else torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
                    )

                    threshold = group.get("threshold", 65536)
                    if group["precision"] == "auto":
                        # Smart Auto Partitioning:
                        # Massive 2D+ tensors or tensors >= threshold (>99.8% of model parameters)
                        # receive compact bfloat16 state storage, cutting state VRAM in half (50% cut).
                        # Small 1D tensors (LayerNorm, RMSNorm, BatchNorm, biases < threshold)
                        # receive full precision (float32) for zero precision drift and fast native foreach.
                        is_massive = (p.dim() >= 2) or (p.numel() >= threshold)
                        if is_massive:
                            state_dtype = _get_compact_dtype(p)
                        else:
                            state_dtype = p.dtype if torch.is_complex(p) else torch.float32
                    elif group["precision"] == "compact":
                        state_dtype = _get_compact_dtype(p)
                    else:  # "full"
                        state_dtype = p.dtype if torch.is_complex(p) else torch.float32
                    state["state_dtype"] = state_dtype

                    # Primary moment buffer
                    state["exp_avg"] = torch.zeros_like(
                        p, dtype=state_dtype, memory_format=torch.preserve_format
                    )

                    # AdEMAMix slow moment buffer
                    if has_slow_momentum:
                        state["exp_avg_slow"] = torch.zeros_like(
                            p, dtype=state_dtype, memory_format=torch.preserve_format
                        )

                    # Second moment buffer (variance EMA, not needed in Lion mode)
                    if not is_lion:
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=state_dtype, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, dtype=state_dtype, memory_format=torch.preserve_format
                            )

                # Defensive initialization if amsgrad or AdEMAMix was enabled after initial step
                state_dtype = state.get("state_dtype", p.dtype)
                if has_slow_momentum and "exp_avg_slow" not in state:
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, dtype=state_dtype, memory_format=torch.preserve_format
                    )
                if not is_lion and group["amsgrad"] and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, dtype=state_dtype, memory_format=torch.preserve_format
                    )

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

        # Synchronize individual parameter gradients into flat gradient buffers for auto-packed groups
        if getattr(self, "auto_pack", False):
            for packed in self.packed_param_groups:
                flat_param = packed["flat_param"]
                individual_params = packed["individual_params"]
                is_1d = packed["is_1d"]

                if any(p.grad is not None for p in individual_params):
                    grads = []
                    for p in individual_params:
                        if p.grad is not None:
                            grads.append(p.grad.detach().contiguous().view(-1))
                        else:
                            grads.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
                    cat_grad = torch.cat(grads)
                    flat_param.grad = cat_grad if is_1d else cat_grad.unsqueeze(-1)
                else:
                    flat_param.grad = None

        for group in self.param_groups:
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
            use_foreach = (foreach is True) or (foreach is None and is_accelerator)

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

            inv_scale = 1.0
            if backend_grad_scale is not None and not is_capturing and not compiling:
                inv_scale = float(backend_grad_scale.reciprocal().item())

            # -------------------------------------------------------------
            # PATH 2: OpenAI Triton Fused GPU Kernel (NVIDIA CUDA & AMD ROCm)
            # -------------------------------------------------------------
            # Automatically compiles down to highly efficient machine code
            # for both NVIDIA GPUs (PTX/SASS) and AMD GPUs (ROCm / HIP).
            # Fuses AMP gradient unscaling directly in registers (zero auxiliary VRAM allocation).
            can_use_triton = (
                (backend == "triton" or (backend in (None, "auto") and is_triton_available()))
                and is_cuda
                and not cautious
                and not nesterov
                and beta3 is None
                and not lion
                and not amsgrad
                and not capturable
                and not compiling
                and not maximize
            )
            # Only evaluate steps_are_uniform when Triton or custom CUDA is viable.
            # Avoid calling _get_value (Tensor.item()) under torch.compile to prevent Dynamo graph breaks.
            steps_are_uniform = False
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
                and found_inf is None
                and is_cuda_kernel_available()
            )
            if can_use_triton or can_use_cuda_fused:
                steps_are_uniform = len(state_steps) <= 1 or all(
                    _get_value(s) == _get_value(state_steps[0]) for s in state_steps[1:]
                )

            if can_use_triton and steps_are_uniform:
                step_val = int(_get_value(state_steps[0])) + 1
                torch._foreach_add_(state_steps, 1)
                scales_v = {}
                if group.get("dynamic_scaling", True):
                    for p in params_with_grad:
                        if "scale_v" in self.state[p]:
                            scales_v[id(p)] = self.state[p]["scale_v"]

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
                    scales_v=scales_v if scales_v else None,
                    inv_scale=inv_scale,
                )
                if success:
                    continue
                # If Triton execution did not succeed, roll back step counter for fallback
                torch._foreach_sub_(state_steps, 1)

            # -------------------------------------------------------------
            # PATH 3: Custom C++/CUDA 128-bit Vectorized Fused Kernel
            # -------------------------------------------------------------
            # High-performance multi-tensor fused CUDA kernel fast path
            # (SM 7.0+ Volta/Turing/Ampere/Ada/Hopper)
            if (
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
                and found_inf is None
                and steps_are_uniform
                and is_cuda_kernel_available()
            ):
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
                    continue
                # If CUDA execution did not succeed, roll back step counter for fallback
                torch._foreach_sub_(state_steps, 1)

            # -------------------------------------------------------------
            # PATH 4: Apple Silicon (MPS) & PyTorch Native Multi-Tensor
            # -------------------------------------------------------------
            # PyTorch translates native _foreach_ multi-tensor operations directly
            # to Apple Metal Performance Shaders on Apple Silicon (M1/M2/M3/M4 Macs).
            if (
                fused
                and precision == "full"
                and beta3 is None
                and not lion
                and not cautious
                and not nesterov
                and not decay_norm_only
                and hasattr(torch, "_fused_adamw_")
                and is_cuda
                and not capturable
            ):
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
                    grad_scale=backend_grad_scale,
                    found_inf=found_inf,
                )
            elif use_foreach or is_mps or backend in ("mps", "foreach", "_multi_tensor"):
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
                except (NotImplementedError, RuntimeError):
                    # Graceful fallback to single tensor step if an op is missing on MPS
                    if is_mps or backend in ("mps", "foreach", "_multi_tensor"):
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
                        raise
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
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
    ):
        """Vectorized execution using PyTorch native C++/CUDA _fused_adamw_ kernel."""
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

            # Dynamic range scaling for 16-bit variance states (prevents underflow)
            scale_v = 1.0
            sqrt_scale_v = 1.0
            if dynamic_scaling and v is not None and v.dtype in (torch.bfloat16, torch.float16):
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
            key = (dev, param_dt, state_dt)

            # Dynamic range scaling check & recalibration across subgroup (protects 16-bit states from underflow)
            scale_v = 1.0
            sqrt_scale_v = 1.0
            if dynamic_scaling and not lion and target_dtype in (torch.bfloat16, torch.float16):
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
                # Cast grads to compact dtype if needed
                if c_grads[0].dtype != target_dtype:
                    c_grads_comp = [g.to(target_dtype) for g in c_grads]
                else:
                    c_grads_comp = c_grads

                # 2. Lion Mode multi-tensor update
                if lion:
                    updates = torch._foreach_lerp(c_grads_comp, c_exp_avgs, beta1)
                    updates = [u.sign() for u in updates]
                    torch._foreach_add_(c_params, updates, alpha=-lr)
                    torch._foreach_lerp_(c_exp_avgs, c_grads_comp, 1.0 - beta2)
                    del updates, c_grads_comp
                    continue

                # Numerical Contract: FP32 arithmetic with compact (16-bit) state storage.
                # Aligns multi-tensor CPU/fallback with Triton and CUDA kernels by accumulating
                # moving averages in float32 before storing back into compact 16-bit buffers.
                if target_dtype in (torch.bfloat16, torch.float16):
                    c_grads_f = [g.float() for g in c_grads]
                    c_exp_avgs_f = [m.float() for m in c_exp_avgs]
                    torch._foreach_lerp_(c_exp_avgs_f, c_grads_f, 1.0 - beta1)
                    for dst, src in zip(c_exp_avgs, c_exp_avgs_f):
                        dst.copy_(src)

                    if beta3 is not None:
                        c_exp_avgs_slow_f = [m.float() for m in c_exp_avgs_slow]
                        torch._foreach_lerp_(c_exp_avgs_slow_f, c_grads_f, 1.0 - beta3)
                        for dst, src in zip(c_exp_avgs_slow, c_exp_avgs_slow_f):
                            dst.copy_(src)

                    c_exp_avg_sqs_f = [v.float() for v in c_exp_avg_sqs]
                    torch._foreach_mul_(c_exp_avg_sqs_f, beta2)
                    torch._foreach_addcmul_(c_exp_avg_sqs_f, c_grads_f, c_grads_f, value=(1.0 - beta2) * scale_v)
                    for dst, src in zip(c_exp_avg_sqs, c_exp_avg_sqs_f):
                        dst.copy_(src)
                else:
                    # Update first-order momentum
                    torch._foreach_lerp_(c_exp_avgs, c_grads, 1.0 - beta1)
                    if beta3 is not None:
                        torch._foreach_lerp_(c_exp_avgs_slow, c_grads, 1.0 - beta3)

                    # Update second-order momentum with dynamic range scale
                    torch._foreach_mul_(c_exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(c_exp_avg_sqs, c_grads, c_grads, value=(1.0 - beta2) * scale_v)

                if amsgrad:
                    torch._foreach_maximum_(c_max_exp_avg_sqs, c_exp_avg_sqs)
                    v_sources = c_max_exp_avg_sqs
                else:
                    v_sources = c_exp_avg_sqs

                # Free c_grads_comp early if not needed for nesterov/cautious
                if not cautious and not nesterov:
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
                if cautious or nesterov:
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
