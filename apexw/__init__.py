"""ApexW: Memory-efficient fused AdamW and Muon optimizers for PyTorch."""

from .optimizer import (
    ApexW,
    build_llm_param_groups,
    _quantize_blockwise_int8,
    _dequantize_blockwise_int8,
    _quantize_blockwise_uint8,
    _dequantize_blockwise_uint8,
)
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
    triton_adamw_8bit_step,
    triton_multi_tensor_adamw_8bit_step,
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
    "_quantize_blockwise_int8",
    "_dequantize_blockwise_int8",
    "_quantize_blockwise_uint8",
    "_dequantize_blockwise_uint8",
    "is_cuda_kernel_available",
    "is_triton_available",
    "is_rocm",
    "triton_adamw_step",
    "triton_multi_tensor_adamw_step",
    "triton_adamw_fp32_step",
    "triton_adamw_cross_precision_step",
    "triton_adamw_8bit_step",
    "triton_multi_tensor_adamw_8bit_step",
    "is_xla_available",
    "is_xla_device",
    "apply_activation_checkpointing",
    "CheckpointWrapper",
    "get_memory_breakdown",
    "flatten_parameters",
    "is_flattened",
]
__version__ = "1.3.0"
