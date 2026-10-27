#!/usr/bin/env python3
"""
ApexW Quick Verification and Diagnostic Runner.

Run this script to verify your PyTorch environment, inspect optimizer state memory,
and test training convergence comparing ApexW against standard PyTorch AdamW.

Usage:
    python run.py
    python run.py --device cuda --steps 20
    python run.py --help
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn

# Ensure local apexw package is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from apexw import (
    ApexW,
    __version__,
    is_cuda_kernel_available,
    is_triton_available,
    is_rocm,
    is_xla_available,
)


def get_default_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except Exception:
            pass
    return torch.device("cpu")


class SyntheticNet(nn.Module):
    """Simple 3-layer MLP for verification."""
    def __init__(self, in_features: int = 256, hidden_dim: int = 512, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_optimizer_state_memory(optimizer: torch.optim.Optimizer) -> float:
    """Calculates total state buffer memory in megabytes (MB)."""
    total_bytes = 0
    for state in optimizer.state.values():
        for val in state.values():
            if torch.is_tensor(val) and val.ndim >= 1:
                total_bytes += val.nelement() * val.element_size()
    return total_bytes / (1024 * 1024)


def _resolve_active_backend(device: torch.device, requested: str) -> str:
    """
    Report the backend ApexW will actually dispatch to on this device,
    not just the --backend flag string.  The optimizer's step() follows
    the same priority order: XLA → Triton → CUDA kernel → MPS/Foreach.
    """
    if requested not in (None, "auto"):
        return requested  # user explicitly pinned a backend
    dev_type = device.type
    if dev_type == "xla" or is_xla_available():
        return "xla"
    if dev_type == "cuda":
        if is_triton_available():
            return "triton (auto)"
        if is_cuda_kernel_available():
            return "cuda-fused (auto)"
        return "foreach (CUDA, no extension)"
    if dev_type == "mps":
        return "mps / foreach (Metal)"
    return "foreach (CPU)"


def run_verification(args):
    device = torch.device(args.device) if args.device != "auto" else get_default_device()

    triton_status = "Available (CUDA & ROCm JIT)" if is_triton_available() else "Not Available"
    mps_status = "Available (Metal Multi-Tensor)" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "Not Available"
    xla_status = "Available (Global Graph Fusion)" if is_xla_available() else "Not Available"
    # Note: is_cuda_kernel_available() may trigger JIT compilation here if build tools are present.
    # That's acceptable in the verification runner (explicit diagnostics context) but not in example.py.
    cuda_status = "Available (C++/CUDA Fused SM 7.0+)" if is_cuda_kernel_available() else "Not Active"

    active_backend = _resolve_active_backend(device, args.backend)

    print("=" * 70)
    print(f" ApexW Cross-Platform Verification Runner (v{__version__})")
    print("=" * 70)
    print(f" Python        : {sys.version.split()[0]}")
    print(f" PyTorch       : {torch.__version__}")
    print(f" Target Device : {device}")
    print(f" Active Backend: {active_backend}")
    if args.compile:
        print(f" torch.compile : Enabled")
    if device.type == "cuda":
        arch_type = "AMD ROCm" if is_rocm() else "NVIDIA CUDA"
        print(f" GPU Device    : {torch.cuda.get_device_name(0)} ({arch_type})")
    print("-" * 70)
    print(f" Hardware Acceleration Engines:")
    print(f"  - OpenAI Triton (CUDA/ROCm)  : {triton_status}")
    print(f"  - Apple Silicon (MPS)        : {mps_status}")
    print(f"  - Google TPU (PyTorch XLA)   : {xla_status}")
    print(f"  - Custom C++/CUDA Extension  : {cuda_status}")
    print("=" * 70)

    # 1. Model Initialization
    torch.manual_seed(42)
    in_dim, hidden_dim, out_dim = args.in_dim, args.hidden_dim, args.out_dim
    batch_size = args.batch_size

    model_adamw = SyntheticNet(in_dim, hidden_dim, out_dim).to(device)
    model_apexw = SyntheticNet(in_dim, hidden_dim, out_dim).to(device)
    model_apexw.load_state_dict(model_adamw.state_dict())

    if args.compile:
        model_adamw = torch.compile(model_adamw)
        model_apexw = torch.compile(model_apexw)

    param_count = sum(p.numel() for p in model_adamw.parameters())
    print(f"\nModel Configuration:")
    print(f" - Layers       : 3-Layer MLP (Linear + LayerNorm + GELU)")
    print(f" - Dimensions   : {in_dim} -> {hidden_dim} -> {hidden_dim} -> {out_dim}")
    print(f" - Parameters   : {param_count:,} ({param_count * 4 / (1024*1024):.2f} MB in FP32)")

    # 2. Synthetic Data
    X = torch.randn(batch_size, in_dim, device=device)
    y = torch.randint(0, out_dim, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()

    # 3. Optimizers
    backend_val = args.backend if args.backend != "auto" else None
    opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=args.lr, weight_decay=1e-2)
    opt_apexw = ApexW(model_apexw.parameters(), lr=args.lr, weight_decay=1e-2, backend=backend_val)

    # 4. Training Loop Comparison
    print(f"\nRunning {args.steps} optimization steps...")
    print(f"{'Step':>6} | {'AdamW Loss':>12} | {'ApexW Loss':>12} | {'Loss Diff':>12}")
    print("-" * 52)

    initial_loss_adamw = 0.0
    initial_loss_apexw = 0.0
    final_loss_adamw = 0.0
    final_loss_apexw = 0.0

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        # AdamW step
        opt_adamw.zero_grad(set_to_none=True)
        out_a = model_adamw(X)
        loss_a = criterion(out_a, y)
        loss_a.backward()
        opt_adamw.step()

        # ApexW step
        opt_apexw.zero_grad(set_to_none=True)
        out_x = model_apexw(X)
        loss_x = criterion(out_x, y)
        loss_x.backward()
        opt_apexw.step()

        val_a = loss_a.item()
        val_x = loss_x.item()

        if step == 1:
            initial_loss_adamw = val_a
            initial_loss_apexw = val_x
        if step == args.steps:
            final_loss_adamw = val_a
            final_loss_apexw = val_x

        diff = abs(val_a - val_x)
        print(f"{step:>6} | {val_a:>12.6f} | {val_x:>12.6f} | {diff:>12.6e}")

    elapsed = time.perf_counter() - t0

    # 5. Health & Numerical Verification
    has_nan_or_inf = False
    for p in model_apexw.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            has_nan_or_inf = True
            break

    # 6. Memory Comparison
    mem_adamw = compute_optimizer_state_memory(opt_adamw)
    mem_apexw = compute_optimizer_state_memory(opt_apexw)
    saved_mb = mem_adamw - mem_apexw
    reduction_pct = (saved_mb / mem_adamw * 100.0) if mem_adamw > 0 else 0.0

    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY")
    print("=" * 70)
    print(f" Loss Reduction (AdamW)  : {initial_loss_adamw:.6f} -> {final_loss_adamw:.6f}")
    print(f" Loss Reduction (ApexW)  : {initial_loss_apexw:.6f} -> {final_loss_apexw:.6f}")
    print(f" Numerical Health        : {'PASS (No NaN/Inf detected)' if not has_nan_or_inf else 'FAIL (NaN/Inf found)'}")
    print(f" Execution Wall-Clock    : {elapsed:.3f} s ({args.steps} steps)")
    print("-" * 70)
    print(f" Optimizer State Memory  :")
    print(f"  - PyTorch AdamW States : {mem_adamw:.2f} MB  (8 bytes/parameter)")
    print(f"  - ApexW Compact States : {mem_apexw:.2f} MB  (4 bytes/parameter)")
    print(f"  - Memory Saved         : {saved_mb:.2f} MB  ({reduction_pct:.1f}% reduction)")
    print("=" * 70)

    if not has_nan_or_inf and final_loss_apexw < initial_loss_apexw:
        print("\nAll checks passed. ApexW is ready for training.\n")
        return 0
    else:
        print("\nVerification warning: convergence or numerical check failed.\n")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="ApexW verification runner: checks environment and compares state memory against AdamW."
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps", "xla"],
                        help="Target device (default: auto)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "triton", "cuda", "mps", "foreach", "xla"],
                        help="Target hardware execution backend (default: auto)")
    parser.add_argument("--compile", action="store_true",
                        help="Compile model using torch.compile() for fused kernels")
    parser.add_argument("--steps", type=int, default=10, help="Number of optimization steps (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--in-dim", type=int, default=256, help="Input feature dimension (default: 256)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension (default: 512)")
    parser.add_argument("--out-dim", type=int, default=10, help="Output classes (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")

    args = parser.parse_args()
    return run_verification(args)


if __name__ == "__main__":
    sys.exit(main())
