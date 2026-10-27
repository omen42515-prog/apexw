#!/usr/bin/env python3
"""
ApexW Minimal Quickstart Example

A memory-efficient, PyTorch-compatible drop-in alternative to AdamW that cuts
optimizer state memory by ~50% (4 bytes per parameter instead of 8 bytes).

Run directly:
    python example.py              # Instant eager run (no compilation delay)
    python example.py --compile    # Optional: test with PyTorch 2.x torch.compile()
"""

import argparse
import torch
import torch.nn as nn
from apexw import ApexW
from apexw import is_triton_available, is_xla_available


def _describe_backend(device: str) -> str:
    """
    Return a human-readable description of the backend that ApexW will
    select on this device — WITHOUT triggering CUDA extension JIT compilation.

    Note: is_cuda_kernel_available() can invoke nvcc/JIT on first call if the
    build tools are present, which breaks the documented sub-second first run.
    We do a device-type check here and leave the real kernel probe to the
    optimizer's own lazy dispatch inside step().
    """
    if device == "cuda":
        if is_triton_available():
            return "OpenAI Triton (NVIDIA/ROCm JIT — auto-selected)"
        # Don't call is_cuda_kernel_available() here; it may trigger JIT.
        # The optimizer will probe the extension on the first step() call.
        return "PyTorch Vectorized Foreach (CUDA — custom kernel probed on first step)"
    if device == "mps":
        return "PyTorch Native Multi-Tensor (Apple Metal)"
    if is_xla_available():
        return "PyTorch XLA (Google TPU)"
    return "PyTorch Vectorized Foreach"


def main():
    parser = argparse.ArgumentParser(description="ApexW 1-Minute Quickstart")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile() (optional, introduces first-run tracing overhead)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 65)
    print(" ApexW Quickstart Example".center(65))
    print("=" * 65)
    print(f" Device          : {device}")
    print(f" PyTorch Version : {torch.__version__}")
    print(f" Active Backend  : {_describe_backend(device)}")
    print(f" Mode            : {'torch.compile() (Fused JIT)' if args.compile else 'Eager Execution (Instant)'}")
    print("=" * 65)

    # 1. Define model and ApexW optimizer
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Linear(512, 10),
    ).to(device)

    # ApexW is a PyTorch-compatible optimizer alternative
    optimizer = ApexW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 2. Optional: PyTorch 2.x torch.compile()
    run_model = model
    if args.compile:
        print("\n[*] Compiling model with torch.compile() (compiles forward/backward pass)...")
        run_model = torch.compile(model)

    # 3. Standard training loop (exactly identical to standard PyTorch training)
    print("\n[*] Running 5 training steps...")
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(32, 256, device=device)
    targets = torch.randint(0, 10, (32,), device=device)

    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)
        outputs = run_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"    Step {step}: Loss = {loss.item():.4f}")

    # 4. Print optimizer state memory savings
    print("\n[*] Optimizer Memory Summary:")
    optimizer.print_summary()
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
