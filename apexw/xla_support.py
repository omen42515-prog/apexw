"""
ApexW Google TPU (PyTorch XLA) Support Module.

Provides utilities and execution support for Google Cloud TPUs (v2/v3/v4/v5e/v5p/v6e)
running via PyTorch/XLA (torch_xla).

Ensures zero CPU-TPU synchronization barriers and enables full XLA graph fusion
into custom Google TPU ASICs.
"""

from typing import Any, Optional
import torch


def is_xla_available() -> bool:
    """Returns True if PyTorch/XLA (torch_xla) is installed in the environment."""
    try:
        import torch_xla
        return True
    except ImportError:
        return False


def is_xla_tensor(tensor: torch.Tensor) -> bool:
    """Returns True if tensor is residing on a Google TPU (device type 'xla')."""
    return tensor.device.type == "xla"


def is_xla_device(device: Any) -> bool:
    """Returns True if device represents an XLA TPU device."""
    if isinstance(device, str):
        return device.startswith("xla")
    if isinstance(device, torch.device):
        return device.type == "xla"
    return False


def mark_step() -> None:
    """
    Triggers an XLA step barrier to compile and execute accumulated graphs.
    Safe no-op if PyTorch/XLA is not active.
    """
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except Exception:
        pass


def get_xla_device(index: Optional[int] = None) -> torch.device:
    """Returns the PyTorch XLA device for Google TPU."""
    import torch_xla.core.xla_model as xm
    return xm.xla_device(index)
