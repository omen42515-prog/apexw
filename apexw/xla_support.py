"""PyTorch/XLA device detection utilities."""

from typing import Any
import torch


def is_xla_available() -> bool:
    """Return True if torch_xla is installed."""
    try:
        import torch_xla  # noqa: F401
        return True
    except ImportError:
        return False


def is_xla_tensor(tensor: torch.Tensor) -> bool:
    """Return True if tensor resides on an XLA device."""
    return tensor.device.type == "xla"


def is_xla_device(device: Any) -> bool:
    """Return True if device represents an XLA device."""
    if isinstance(device, str):
        return device.startswith("xla")
    if isinstance(device, torch.device):
        return device.type == "xla"
    return False
