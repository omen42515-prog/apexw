"""
ApexW Parameter Flattening & Grouping Module.

Provides transparent 1D tensor buffer packing for models with many small parameters
(e.g., multi-head assemblies, deep networks with thousands of tiny weights/biases).

By packing thousands of small, separate parameters into a single contiguous 1D parameter
tensor, the optimizer processes exactly 1 unified parameter instead of thousands.
This completely eliminates kernel launch overhead, Python loop overhead, and Dynamo
graph breaks under torch.compile().

All sub-layer parameter references (e.g., `layer.weight`, `layer.bias`) are dynamically
redirected to sliced, zero-copy views into the underlying flat tensor.
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import torch
import torch.nn as nn


def flatten_parameters(
    model: nn.Module,
    max_numel: Optional[int] = None,
    param_name: str = "flat_param",
) -> nn.Parameter:
    """
    Packs model parameters into a single contiguous 1D torch.nn.Parameter.

    Sub-module attributes (e.g., `layer.weight`, `layer.bias`) are transparently
    replaced with dynamic zero-copy slice views into the unified buffer. Forward
    passes, backward passes, and autograd gradient accumulation operate directly
    on the contiguous memory block.

    Args:
        model: Target PyTorch nn.Module to flatten.
        max_numel: Optional upper bound on parameter element count. If specified,
            only parameters with `p.numel() <= max_numel` are packed.
        param_name: Attribute name under which the flattened parameter is registered
            on `model`. Defaults to 'flat_param'.

    Returns:
        flat_param: The newly created unified torch.nn.Parameter.
    """
    if getattr(model, "_is_flattened", False):
        return getattr(model, param_name)

    named_params = list(model.named_parameters())
    if not named_params:
        raise ValueError("Model has no parameters to flatten.")

    # Filter target parameters
    targets = []
    untargeted = []
    for name, p in named_params:
        if max_numel is None or p.numel() <= max_numel:
            targets.append((name, p))
        else:
            untargeted.append((name, p))

    if not targets:
        raise ValueError(f"No parameters matched the criteria (max_numel={max_numel}).")

    # Verify device and dtype homogeneity across packed parameters
    first_device = targets[0][1].device
    first_dtype = targets[0][1].dtype
    for name, p in targets[1:]:
        if p.device != first_device:
            raise ValueError(
                f"Cannot flatten heterogeneous devices: '{name}' is on {p.device}, "
                f"expected {first_device}"
            )
        if p.dtype != first_dtype:
            raise ValueError(
                f"Cannot flatten heterogeneous dtypes: '{name}' has {p.dtype}, "
                f"expected {first_dtype}"
            )

    # Build 1D contiguous flat tensor
    param_info: List[Tuple[str, int, int, torch.Size, torch.dtype]] = []
    curr = 0
    tensors = []
    for name, p in targets:
        numel = p.numel()
        tensors.append(p.detach().flatten())
        param_info.append((name, curr, curr + numel, p.shape, p.dtype))
        curr += numel

    flat_tensor = torch.cat(tensors)
    flat_param = nn.Parameter(flat_tensor, requires_grad=True)

    # Group target parameters by parent submodule
    mod_to_params = defaultdict(list)
    for name, start, end, shape, dt in param_info:
        parts = name.split(".")
        mod_path = ".".join(parts[:-1])
        attr_name = parts[-1]
        mod_to_params[mod_path].append((attr_name, start, end, shape))

    # Redirect module attributes to dynamic slice views
    for mod_path, p_list in mod_to_params.items():
        submod = model
        if mod_path:
            for part in mod_path.split("."):
                submod = getattr(submod, part)

        props: Dict[str, Any] = {}
        for attr_name, s, e, sh in p_list:
            if attr_name in submod._parameters:
                del submod._parameters[attr_name]
            # Zero-copy view property directly into the flat buffer
            props[attr_name] = (
                lambda s=s, e=e, sh=sh: property(lambda self: flat_param[s:e].view(sh))
            )()

        # Subclass the specific instance to avoid polluting other instances of the same class
        cls_name = f"Flattened_{submod.__class__.__name__}"
        submod_cls = type(cls_name, (submod.__class__,), props)
        submod.__class__ = submod_cls

    model.register_parameter(param_name, flat_param)
    model._is_flattened = True
    model._flatten_metadata = param_info
    model._flatten_param_name = param_name

    # Checkpoint interoperability helpers
    def get_unflattened_state_dict() -> Dict[str, torch.Tensor]:
        """Returns standard PyTorch state_dict with individual parameter keys."""
        sd: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, s, e, sh, _ in model._flatten_metadata:
                sd[name] = flat_param.data[s:e].view(sh).clone()
            for name, p in untargeted:
                sd[name] = p.data.clone()
        return sd

    def load_unflattened_state_dict(
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ) -> None:
        """Loads a standard unflattened state_dict directly into the contiguous buffer."""
        with torch.no_grad():
            for name, s, e, sh, _ in model._flatten_metadata:
                if name in state_dict:
                    flat_param.data[s:e].copy_(state_dict[name].flatten())
                elif strict:
                    raise KeyError(f"Missing key in state_dict: '{name}'")
            for name, p in untargeted:
                if name in state_dict:
                    p.data.copy_(state_dict[name])
                elif strict:
                    raise KeyError(f"Missing key in state_dict: '{name}'")

    model.get_unflattened_state_dict = get_unflattened_state_dict
    model.load_unflattened_state_dict = load_unflattened_state_dict

    return flat_param


def is_flattened(model: nn.Module) -> bool:
    """Returns True if the model has been flattened with flatten_parameters."""
    return getattr(model, "_is_flattened", False)
