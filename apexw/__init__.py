"""
ApexW: Memory-Efficient Fused Optimizer for PyTorch.

Drop-in replacement for torch.optim.AdamW with ~50% lower optimizer state memory,
multi-hardware acceleration across:
1. OpenAI Triton: Automatic GPU machine-code compilation for NVIDIA CUDA & AMD ROCm
2. Apple Silicon (MPS): PyTorch native vectorized Multi-Tensor (_multi_tensor) engine
3. Google TPUs: PyTorch XLA global graph compilation and ASIC kernel fusion
4. NVIDIA CUDA: Custom C++/CUDA 128-bit vectorized fused kernels (SM 7.0+)
"""

from .optimizer import ApexW, build_llm_param_groups
from .zero import ApexWZeRO
from .cuda_extension import is_cuda_kernel_available
from .triton_kernel import is_triton_available, is_rocm
from .xla_support import is_xla_available, is_xla_device
from .checkpointing import apply_activation_checkpointing, CheckpointWrapper, get_memory_breakdown
from .flatten import flatten_parameters, is_flattened

__all__ = [
    "ApexW",
    "ApexWZeRO",
    "build_llm_param_groups",
    "is_cuda_kernel_available",
    "is_triton_available",
    "is_rocm",
    "is_xla_available",
    "is_xla_device",
    "apply_activation_checkpointing",
    "CheckpointWrapper",
    "get_memory_breakdown",
    "flatten_parameters",
    "is_flattened",
]
__version__ = "1.2.0"
