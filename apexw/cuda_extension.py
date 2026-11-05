"""
ApexW Custom CUDA Fused Kernel JIT Loader & Interface.

Compiles and loads the custom 128-bit vectorized mixed-precision AdamW kernel
when running on NVIDIA GPUs (SM 7.0+, including Google Colab T4, V100, A100, H100)
with CUDA available.
Provides silent, zero-crash fallback to PyTorch native vectorized engine on CPU
or non-CUDA architectures.
"""

import os
import sys
import math
import warnings
from typing import Optional, List
import torch

_CUDA_OP = None
_JIT_ATTEMPTED = False


def _get_cuda_home() -> Optional[str]:
    """Find CUDA_HOME from environment or default locations."""
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is not None and os.path.exists(CUDA_HOME):
        return CUDA_HOME
    for path in ["/usr/local/cuda", "/usr/cuda", os.environ.get("CUDA_PATH", "")]:
        if path and os.path.exists(path):
            return path
    return None


def get_cuda_kernel():
    """JIT-compiles or loads the custom ApexW CUDA kernel if available."""
    global _CUDA_OP, _JIT_ATTEMPTED
    if _JIT_ATTEMPTED:
        return _CUDA_OP
    _JIT_ATTEMPTED = True

    if not torch.cuda.is_available():
        return None

    # First check if pre-compiled extension is already installed
    try:
        import apexw_cuda_ext
        _CUDA_OP = apexw_cuda_ext
        return _CUDA_OP
    except ImportError:
        pass

    try:
        from apexw import apexw_cuda_ext
        _CUDA_OP = apexw_cuda_ext
        return _CUDA_OP
    except ImportError:
        pass

    cuda_home = _get_cuda_home()
    if cuda_home is None:
        return None

    try:
        cap = torch.cuda.get_device_capability()
        # Compute Capability >= 7.0 covers Volta (V100), Turing (T4), Ampere (A100, RTX 30x), Ada (L4, RTX 40x), Hopper (H100)
        if cap[0] < 7:
            return None

        from torch.utils.cpp_extension import load

        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_dirs = [
            os.path.join(current_dir, "csrc"),
            os.path.join(os.path.dirname(current_dir), "csrc"),
            os.path.join(os.path.dirname(os.path.dirname(current_dir)), "csrc"),
        ]
        cpp_source = None
        cu_source = None
        for cdir in candidate_dirs:
            p_cpp = os.path.join(cdir, "apexw_cuda.cpp")
            p_cu = os.path.join(cdir, "apexw_cuda_kernel.cu")
            if os.path.exists(p_cpp) and os.path.exists(p_cu):
                cpp_source = p_cpp
                cu_source = p_cu
                break

        if cpp_source is None or cu_source is None:
            return None

        cuda_cflags = [
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{cap[0]}{cap[1]},code=sm_{cap[0]}{cap[1]}",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
        ]

        _CUDA_OP = load(
            name="apexw_cuda_ext",
            sources=[cpp_source, cu_source],
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_cflags,
            verbose=False,
        )
        return _CUDA_OP
    except Exception as e:
        if os.environ.get("APEXW_DEBUG", "0") == "1":
            print(f"[ApexW] JIT compilation notice: {e}")
        _CUDA_OP = None
        return None


def is_cuda_kernel_available() -> bool:
    """Returns True if the custom CUDA fused kernel is active."""
    return get_cuda_kernel() is not None


def step_fused(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int = 1,
    decay_norm_only: bool = False,
) -> bool:
    """Executes the custom batched multi-tensor fused CUDA kernel in A SINGLE KERNEL LAUNCH
    across all tensors. Returns True if successfully executed, False otherwise.
    """
    kernel = get_cuda_kernel()
    if kernel is not None and len(params) > 0 and params[0].is_cuda:
        try:
            if hasattr(kernel, "step_fused"):
                kernel.step_fused(
                    params, grads, exp_avgs, exp_avg_sqs,
                    lr, beta1, beta2, eps, weight_decay, step, decay_norm_only
                )
                return True
            elif hasattr(kernel, "multi_tensor_adamw_step"):
                kernel.multi_tensor_adamw_step(
                    params, grads, exp_avgs, exp_avg_sqs,
                    lr, beta1, beta2, eps, weight_decay, step, decay_norm_only
                )
                return True
        except Exception as e:
            if os.environ.get("APEXW_DEBUG", "0") == "1":
                print(f"[ApexW] CUDA step_fused fallback: {e}")
            return False
    return False


def fused_multi_tensor_adamw_cuda_step(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: int = 1,
    decay_norm_only: bool = False,
) -> bool:
    """Convenience alias for step_fused."""
    return step_fused(
        params, grads, exp_avgs, exp_avg_sqs,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
        step=step, decay_norm_only=decay_norm_only,
    )


def fused_adamw_cuda_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: int,
) -> bool:
    """
    Executes the custom fused CUDA kernel on a single tensor.
    Returns True if successfully executed, False otherwise.
    """
    kernel = get_cuda_kernel()
    if kernel is not None and param.is_cuda:
        try:
            kernel.adamw_step(
                param, grad, exp_avg, exp_avg_sq,
                lr, beta1, beta2, weight_decay, eps, step
            )
            return True
        except Exception as e:
            if os.environ.get("APEXW_DEBUG", "0") == "1":
                print(f"[ApexW] CUDA single-tensor step fallback: {e}")
            return False
    return False
