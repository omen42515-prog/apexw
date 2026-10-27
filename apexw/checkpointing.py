"""
ApexW Activation & Gradient Checkpointing Utilities.

Provides plug-and-play activation checkpointing and memory breakdown diagnostics
for activation-heavy workloads (such as deep 2D/3D CNNs, U-Nets, and high-resolution Vision Transformers),
where activation memory during the forward pass dominates peak VRAM.

Combining ApexW (50% optimizer state reduction) with Activation Checkpointing (60-80% activation reduction)
delivers maximum end-to-end VRAM savings across all model architectures.
"""

from typing import Optional, Tuple, List, Callable, Type, Union, Dict, Any
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointWrapper(nn.Module):
    """
    Lightweight wrapper that executes a child module's forward pass using
    activation checkpointing (recomputing forward activations during backward pass).

    Args:
        module (nn.Module): The module to wrap.
        use_reentrant (bool): Whether to use reentrant autograd checkpointing.
            Default is False (recommended for PyTorch 2.x, DDP, and AMP).
        preserve_rng_state (bool): Whether to preserve RNG state for dropout / stochastic ops.
    """
    def __init__(
        self,
        module: nn.Module,
        use_reentrant: bool = False,
        preserve_rng_state: bool = True,
    ):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state

    def forward(self, *args, **kwargs):
        if not self.training or not any(
            isinstance(a, torch.Tensor) and a.requires_grad for a in args
        ) and not any(p.requires_grad for p in self.module.parameters()):
            return self.module(*args, **kwargs)

        def custom_forward(*inputs):
            return self.module(*inputs, **kwargs)

        return checkpoint(
            custom_forward,
            *args,
            use_reentrant=self.use_reentrant,
            preserve_rng_state=self.preserve_rng_state,
        )


def apply_activation_checkpointing(
    model: nn.Module,
    layer_types: Optional[Union[Type[nn.Module], Tuple[Type[nn.Module], ...]]] = None,
    check_fn: Optional[Callable[[nn.Module], bool]] = None,
    every_n_layers: int = 1,
    use_reentrant: bool = False,
    preserve_rng_state: bool = True,
) -> int:
    """
    Automatically applies activation checkpointing to matching submodules in a model.

    In activation-heavy architectures (e.g. CNNs with large feature maps or deep Vision Transformers),
    forward activations dominate peak VRAM. Checkpointing discards intermediate activations
    during forward pass and recomputes them on backward pass, slashing activation memory by up to 80%.

    Args:
        model (nn.Module): The model to instrument with activation checkpointing.
        layer_types (Type or Tuple[Type], optional): Module types to wrap with checkpointing
            (e.g., BasicBlock, Bottleneck, TransformerBlock, nn.Sequential).
        check_fn (Callable[[nn.Module], bool], optional): Custom predicate returning True
            for modules that should be checkpointed.
        every_n_layers (int): Checkpoint every N-th candidate layer (default: 1, checkpoints all).
        use_reentrant (bool): Set to False (default) for PyTorch 2.x compatibility with AMP & DDP.
        preserve_rng_state (bool): Preserves random state for dropout/stochastic layers (default: True).

    Returns:
        int: Total number of submodules wrapped with activation checkpointing.

    Example:
        >>> from apexw.checkpointing import apply_activation_checkpointing
        >>> num_wrapped = apply_activation_checkpointing(resnet_model)
        >>> print(f"Wrapped {num_wrapped} layers with activation checkpointing.")
    """
    wrapped_count = 0
    candidate_modules: List[Tuple[nn.Module, str, nn.Module]] = []

    # Strategy 1: User specified layer types or custom check function
    if layer_types is not None or check_fn is not None:
        for parent_name, parent_mod in model.named_modules():
            for child_name, child_mod in parent_mod.named_children():
                if isinstance(child_mod, CheckpointWrapper):
                    continue
                match = False
                if check_fn is not None and check_fn(child_mod):
                    match = True
                elif layer_types is not None and isinstance(child_mod, layer_types):
                    match = True
                if match:
                    candidate_modules.append((parent_mod, child_name, child_mod))
    else:
        # Strategy 2: Automatic discovery of block containers (ModuleList / Sequential / repeated blocks)
        for parent_name, parent_mod in model.named_modules():
            if isinstance(parent_mod, (nn.Sequential, nn.ModuleList)) and len(parent_mod) > 1:
                for child_name, child_mod in parent_mod.named_children():
                    if not isinstance(child_mod, CheckpointWrapper) and len(list(child_mod.parameters())) > 0:
                        candidate_modules.append((parent_mod, child_name, child_mod))

        # If no containers found, look for modules with children that have significant parameters
        if not candidate_modules:
            for parent_name, parent_mod in model.named_modules():
                for child_name, child_mod in parent_mod.named_children():
                    param_count = sum(p.numel() for p in child_mod.parameters())
                    if param_count > 10_000 and not isinstance(child_mod, CheckpointWrapper):
                        candidate_modules.append((parent_mod, child_name, child_mod))

    # Apply checkpointing to candidate modules with stride every_n_layers
    for idx, (parent, name, target_mod) in enumerate(candidate_modules):
        if idx % every_n_layers == 0:
            wrapper = CheckpointWrapper(
                target_mod,
                use_reentrant=use_reentrant,
                preserve_rng_state=preserve_rng_state,
            )
            setattr(parent, name, wrapper)
            wrapped_count += 1

    return wrapped_count


def get_memory_breakdown(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    input_tensor: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Profiles and diagnoses memory distribution between:
    - Model Parameters
    - Optimizer States (reporting ApexW compact state memory savings)
    - Activations (reporting potential gains from activation checkpointing)

    Returns:
        Dict[str, Any] with detailed metrics and a human-readable diagnosis string.
    """
    dev = device if device is not None else next(model.parameters()).device
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 * 1024)

    # Calculate optimizer state memory
    adamw_equiv_mb = (sum(p.numel() for p in model.parameters()) * 8) / (1024 * 1024)
    if optimizer is not None:
        has_stepped = len(optimizer.state) > 0 or (
            hasattr(optimizer, "_local_opt") and len(optimizer._local_opt.state) > 0
        )
        if hasattr(optimizer, "get_state_memory_mb") and has_stepped:
            opt_state_mb = optimizer.get_state_memory_mb()
        else:
            opt_bytes = sum(
                sum(v.nelement() * v.element_size() for v in state.values() if torch.is_tensor(v) and v.ndim >= 1)
                for state in optimizer.state.values()
            )
            if opt_bytes > 0:
                opt_state_mb = opt_bytes / (1024 * 1024)
            else:
                # Estimate compact optimizer memory based on model parameters if not stepped yet
                opt_state_mb = (sum(p.numel() for p in model.parameters()) * 4) / (1024 * 1024)
    else:
        opt_state_mb = (sum(p.numel() for p in model.parameters()) * 4) / (1024 * 1024)

    opt_saved_mb = max(0.0, adamw_equiv_mb - opt_state_mb)

    # Estimate activation memory if input_tensor is supplied
    activation_mb = None
    peak_vram_mb = None
    is_cuda = (dev.type == "cuda" if isinstance(dev, torch.device) else "cuda" in str(dev)) and torch.cuda.is_available()

    if input_tensor is not None and is_cuda:
        # Save model state before profiling to avoid polluting a live training loop
        was_training = model.training
        saved_grads = {p: p.grad.clone() if p.grad is not None else None for p in model.parameters()}
        try:
            model.eval()
            torch.cuda.reset_peak_memory_stats(dev)
            mem_before = torch.cuda.memory_allocated(dev)
            with torch.no_grad():
                out = model(input_tensor)
            mem_after_fwd = torch.cuda.memory_allocated(dev)
            activation_mb = max(0.0, (mem_after_fwd - mem_before) / (1024 * 1024))

            if target is not None and criterion is not None:
                # Re-enable grad only for the backward pass measurement
                model.train()
                out = model(input_tensor)
                loss = criterion(out, target)
                loss.backward()
                peak_vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
        finally:
            # Restore training mode and original gradients
            if was_training:
                model.train()
            else:
                model.eval()
            for p in model.parameters():
                p.grad = saved_grads[p]

    # Formulate architectural advice
    if activation_mb is not None and activation_mb > (param_mb + opt_state_mb) * 2:
        bottleneck = "ACTIVATION-HEAVY"
        recommendation = (
            "Activations dominate peak VRAM (>66% of memory). "
            "Pair ApexW with `apply_activation_checkpointing(model)` to slash activation footprint by 60-80%."
        )
    elif opt_state_mb > param_mb:
        bottleneck = "OPTIMIZER-BOUND"
        recommendation = (
            f"Optimizer states dominate memory footprint. "
            f"ApexW provides peak return on this model, cutting state memory by ~50% ({opt_saved_mb:.2f} MB saved)."
        )
    else:
        bottleneck = "BALANCED"
        recommendation = "Balanced parameter and activation footprint. ApexW provides ~50% state memory reduction."

    return {
        "param_memory_mb": round(param_mb, 2),
        "optimizer_state_mb": round(opt_state_mb, 2),
        "adamw_state_mb": round(adamw_equiv_mb, 2),
        "optimizer_saved_mb": round(opt_saved_mb, 2),
        "activation_memory_mb": round(activation_mb, 2) if activation_mb is not None else None,
        "peak_vram_mb": round(peak_vram_mb, 2) if peak_vram_mb is not None else None,
        "bottleneck": bottleneck,
        "recommendation": recommendation,
    }
