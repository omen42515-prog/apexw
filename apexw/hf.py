"""
ApexW Hugging Face Transformers & Trainer Integration.

Provides first-class, drop-in integration with Hugging Face Trainer:
1. `get_apexw_optimizer(model, ...)`: One-line optimizer factory for `Trainer(optimizers=(opt, None))`.
2. `register_hf_trainer_optimizer()`: Registers 'apexw', 'hybrid_apexw', and 'apex_muon' into
   Hugging Face TrainingArguments (e.g. `TrainingArguments(..., optim="hybrid_apexw")`).
3. `ApexWTrainer`: Drop-in subclass of Hugging Face `Trainer` with built-in parameter routing.
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .optimizer import ApexW
from .muon import ApexMuon
from .hybrid import HybridApexW


def get_apexw_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    muon_lr: Optional[float] = None,
    weight_decay: float = 1e-2,
    use_muon: bool = True,
    adamw_betas: Tuple[float, float] = (0.9, 0.99),
    muon_momentum: float = 0.95,
    precision: str = "compact",
    cautious: bool = False,
    min_muon_dim: int = 32,
    exclude_embeddings: bool = True,
    **kwargs,
) -> Optimizer:
    """
    Constructs an optimal ApexW or HybridApexW optimizer for a PyTorch or Hugging Face model.

    Automatically routes 2D hidden projection matrices to ApexMuon (2 bytes/param) and
    1D normalization layers, biases, and token embeddings to ApexW (4 bytes/param).

    Args:
        model: PyTorch nn.Module or Hugging Face PreTrainedModel.
        lr: Learning rate for 1D/embedding parameters (default: 1e-3).
        muon_lr: Learning rate for 2D hidden weight matrices (default: 2e-2).
        weight_decay: Decoupled weight decay (default: 1e-2).
        use_muon: Whether to enable Hybrid routing with ApexMuon (default: True).
        adamw_betas: Beta coefficients for AdamW moments (default: (0.9, 0.95)).
        muon_momentum: Momentum coefficient for Muon (default: 0.95).
        precision: State storage precision ('auto', 'compact', or 'full').
        cautious: Enable cautious directional update masking (default: False).
        min_muon_dim: Minimum matrix dimension for Muon routing (default: 32).
        exclude_embeddings: Protect token embeddings from Muon (default: True).

    Returns:
        torch.optim.Optimizer: Configured HybridApexW or ApexW optimizer instance.

    Example with Hugging Face Trainer:
        >>> from transformers import Trainer, TrainingArguments
        >>> from apexw.hf import get_apexw_optimizer
        >>> optimizer = get_apexw_optimizer(model, lr=5e-5, use_muon=True)
        >>> trainer = Trainer(model=model, args=training_args, optimizers=(optimizer, None))
    """
    if use_muon:
        return HybridApexW(
            model,
            lr=lr,
            muon_lr=muon_lr,
            muon_momentum=muon_momentum,
            adamw_betas=adamw_betas,
            weight_decay=weight_decay,
            precision=precision,
            cautious=cautious,
            min_muon_dim=min_muon_dim,
            exclude_embeddings=exclude_embeddings,
            **kwargs,
        )
    return ApexW(
        model,
        lr=lr,
        betas=adamw_betas,
        weight_decay=weight_decay,
        precision=precision,
        cautious=cautious,
        **kwargs,
    )


def register_hf_trainer_optimizer() -> bool:
    """
    Dynamically registers 'apexw', 'hybrid_apexw', and 'apex_muon' into Hugging Face
    Transformers TrainingArguments and Trainer.

    Enables setting `--optim apexw` or `--optim hybrid_apexw` in CLI / TrainingArguments:
        >>> from apexw.hf import register_hf_trainer_optimizer
        >>> register_hf_trainer_optimizer()
        >>> args = TrainingArguments(..., optim="hybrid_apexw")
        >>> trainer = Trainer(model=model, args=args)

    Returns:
        bool: True if registration was successful, False if transformers is not installed.
    """
try:
    from transformers.training_args import OptimizerNames
except ImportError:
    OptimizerNames = None


def _get_trainer_grouped_params(trainer):
    """Filter parameters honoring requires_grad and Hugging Face Trainer weight decay rules."""
    opt_model = trainer.model
    decay_params = trainer.get_decay_parameter_names(opt_model) if hasattr(trainer, "get_decay_parameter_names") else []
    trainable = [p for p in opt_model.parameters() if p.requires_grad]
    if decay_params and trainable:
        return [
            {
                "params": [p for n, p in opt_model.named_parameters() if n in decay_params and p.requires_grad],
                "weight_decay": trainer.args.weight_decay,
            },
            {
                "params": [p for n, p in opt_model.named_parameters() if n not in decay_params and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
    return trainable if trainable else list(opt_model.parameters())


def patched_get_optimizer_cls_and_kwargs(args, model=None):
    """Resolves optimizer class and keyword arguments for Hugging Face TrainingArguments."""
    optim_name = getattr(args, "optim", None)
    if OptimizerNames is not None and isinstance(optim_name, OptimizerNames):
        optim_name = optim_name.value
    elif isinstance(optim_name, str):
        optim_name = optim_name.lower()

    if optim_name in ("hybrid_apexw", "apexw_hybrid", "apex_hybrid"):
        kwargs = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adamw_betas": (args.adam_beta1, args.adam_beta2),
        }
        return HybridApexW, kwargs

    if optim_name in ("apexw_8bit", "apex_8bit", "apexw_bnb_8bit", "apex_bnb_8bit"):
        kwargs = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            "precision": "8bit",
            "adamw_compat": True,
        }
        return ApexW, kwargs

    if optim_name in ("apexw_ultra", "apex_ultra"):
        kwargs = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            "precision": "ultra",
            "adamw_compat": True,
        }
        return ApexW, kwargs

    if optim_name in ("apexw", "apex_adamw", "apexw_compact", "apex_compact"):
        kwargs = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            "precision": "compact",
            "adamw_compat": True,
        }
        return ApexW, kwargs

    if optim_name in ("apex_muon", "muon"):
        kwargs = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
        return ApexMuon, kwargs

    raise ValueError(f"Unknown optimizer {optim_name}")


def patched_create_optimizer(self):
    """Constructs optimizer inside Hugging Face Trainer with correct parameter grouping."""
    optim_name = getattr(self.args, "optim", None)
    if OptimizerNames is not None and isinstance(optim_name, OptimizerNames):
        optim_name = optim_name.value
    elif isinstance(optim_name, str):
        optim_name = optim_name.lower()

    if optim_name in ("hybrid_apexw", "apexw_hybrid", "apex_hybrid"):
        if self.optimizer is None:
            self.optimizer = get_apexw_optimizer(
                self.model,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                adamw_betas=(self.args.adam_beta1, self.args.adam_beta2),
                use_muon=True,
            )
        return self.optimizer

    if optim_name in ("apexw_8bit", "apex_8bit", "apexw_bnb_8bit", "apex_bnb_8bit"):
        if self.optimizer is None:
            params_to_use = _get_trainer_grouped_params(self)
            self.optimizer = ApexW(
                params_to_use,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                precision="8bit",
                adamw_compat=True,
            )
        return self.optimizer

    if optim_name in ("apexw_ultra", "apex_ultra"):
        if self.optimizer is None:
            params_to_use = _get_trainer_grouped_params(self)
            self.optimizer = ApexW(
                params_to_use,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                precision="ultra",
                adamw_compat=True,
            )
        return self.optimizer

    if optim_name in ("apexw", "apex_adamw", "apexw_compact", "apex_compact"):
        if self.optimizer is None:
            params_to_use = _get_trainer_grouped_params(self)
            self.optimizer = ApexW(
                params_to_use,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                precision="compact",
                adamw_compat=True,
            )
        return self.optimizer

    orig_create = getattr(self, "_orig_create_optimizer", None)
    if orig_create is not None:
        return orig_create(self)
    raise ValueError(f"Unknown optimizer {optim_name}")


def register_hf_trainer_optimizer() -> bool:
    """
    Dynamically registers 'apexw', 'apexw_8bit', 'hybrid_apexw', and 'apex_muon' into
    Hugging Face Transformers TrainingArguments and Trainer.

    Enables setting `--optim apexw_8bit` or `--optim apexw` in CLI / TrainingArguments.
    """
    try:
        import transformers
        from transformers.training_args import OptimizerNames as HF_OptimizerNames
        from transformers import Trainer
    except ImportError:
        return False

    # 1. Add enum members to OptimizerNames if not already present
    names_to_add = {
        "APEXW": "apexw",
        "APEXW_8BIT": "apexw_8bit",
        "APEX_8BIT": "apex_8bit",
        "APEXW_BNB_8BIT": "apexw_bnb_8bit",
        "APEX_BNB_8BIT": "apex_bnb_8bit",
        "APEXW_ULTRA": "apexw_ultra",
        "APEX_ULTRA": "apex_ultra",
        "APEXW_COMPACT": "apexw_compact",
        "APEX_COMPACT": "apex_compact",
        "HYBRID_APEXW": "hybrid_apexw",
        "APEX_MUON": "apex_muon",
    }
    for attr, val in names_to_add.items():
        if not hasattr(HF_OptimizerNames, attr):
            try:
                object.__setattr__(HF_OptimizerNames, attr, val)
            except Exception:
                pass

    # 2. Hook Trainer.get_optimizer_cls_and_kwargs and create_optimizer
    Trainer.get_optimizer_cls_and_kwargs = staticmethod(patched_get_optimizer_cls_and_kwargs)
    if not hasattr(Trainer, "_orig_create_optimizer"):
        Trainer._orig_create_optimizer = Trainer.create_optimizer
    Trainer.create_optimizer = patched_create_optimizer
    return True


patch_huggingface_trainer = register_hf_trainer_optimizer


try:
    from transformers import Trainer

    class ApexWTrainer(Trainer):
        """
        Drop-in subclass of Hugging Face `Trainer` with native ApexW & Hybrid routing.

        Example:
            >>> from apexw.hf import ApexWTrainer
            >>> from transformers import TrainingArguments
            >>> trainer = ApexWTrainer(
            ...     model=model,
            ...     args=TrainingArguments(output_dir="./out", learning_rate=5e-5),
            ...     train_dataset=train_ds,
            ...     use_muon=True,
            ... )
            >>> trainer.train()
        """
        def __init__(self, *args, use_muon: bool = True, muon_lr: float = 2e-2, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_muon = use_muon
            self.muon_lr = muon_lr

        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = get_apexw_optimizer(
                    self.model,
                    lr=self.args.learning_rate,
                    muon_lr=self.muon_lr,
                    weight_decay=self.args.weight_decay,
                    use_muon=self.use_muon,
                    adamw_betas=(self.args.adam_beta1, self.args.adam_beta2),
                )
            return self.optimizer

except ImportError:
    ApexWTrainer = None  # type: ignore


# Automatically register if transformers is present in the environment
register_hf_trainer_optimizer()
