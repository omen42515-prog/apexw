"""
ApexW: Memory-Efficient Fused and Orthogonal Next-Gen Optimizer for PyTorch.

Drop-in replacement for torch.optim.AdamW and Muon with:
1. ApexMuon: High-Performance Spectral Orthogonal Optimizer for 2D Hidden Layers (2 bytes/param)
2. HybridApexW: Unified Automatic Routing (Muon for 2D matrices, ApexW for 1D/embeddings)
3. ApexW: Memory-Efficient Fused AdamW with ~50% lower optimizer state memory (4 bytes/param)
4. Multi-hardware acceleration across OpenAI Triton, PyTorch Foreach/MPS, and CUDA C++.
"""

from .optimizer import ApexW, build_llm_param_groups
from .muon import (
    ApexMuon,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
    zeropower_via_polar_express,
    zeropower_via_polar_express_batched,
)
from .hybrid import (
    HybridApexW,
    partition_parameters_for_hybrid,
)
from .zero import ApexWZeRO, reshard_zero_checkpoint, ZeROStateAdapter
from .fsdp import is_fsdp_flat_param, extract_fsdp_param_infos, DistMuon
from .hf import get_apexw_optimizer, register_hf_trainer_optimizer, ApexWTrainer
from .cuda_extension import is_cuda_kernel_available
from .triton_kernel import (
    is_triton_available,
    is_rocm,
    triton_adamw_step,
    triton_multi_tensor_adamw_step,
    triton_adamw_fp32_step,
    triton_adamw_cross_precision_step,
)
from .xla_support import is_xla_available, is_xla_device
from .checkpointing import apply_activation_checkpointing, CheckpointWrapper, get_memory_breakdown
from .flatten import flatten_parameters, is_flattened

__all__ = [
    "ApexW",
    "ApexMuon",
    "HybridApexW",
    "zeropower_via_newtonschulz5",
    "zeropower_via_newtonschulz5_batched",
    "zeropower_via_polar_express",
    "zeropower_via_polar_express_batched",
    "partition_parameters_for_hybrid",
    "get_apexw_optimizer",
    "register_hf_trainer_optimizer",
    "ApexWTrainer",
    "is_fsdp_flat_param",
    "extract_fsdp_param_infos",
    "DistMuon",
    "ApexWZeRO",
    "reshard_zero_checkpoint",
    "ZeROStateAdapter",
    "build_llm_param_groups",
    "is_cuda_kernel_available",
    "is_triton_available",
    "is_rocm",
    "triton_adamw_step",
    "triton_multi_tensor_adamw_step",
    "triton_adamw_fp32_step",
    "triton_adamw_cross_precision_step",
    "is_xla_available",
    "is_xla_device",
    "apply_activation_checkpointing",
    "CheckpointWrapper",
    "get_memory_breakdown",
    "flatten_parameters",
    "is_flattened",
]
__version__ = "1.2.0"
