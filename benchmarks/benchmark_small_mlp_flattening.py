"""
Benchmark: Small MLP Parameter Flattening vs Multi-Tensor & Fused AdamW.

Tests how manual parameter flattening (packing small MLP parameters into a single
contiguous 1D tensor) changes the performance equation on many-small tensor workloads.

Demonstrates:
  1. Pure optimizer step latency (ms/step) across 100 small layers.
  2. Sequential launch bottleneck elimination (100 sequential launches -> 1 unified launch).
  3. End-to-end training loop speedup (Forward + Backward + Optimizer).
  4. Heterogeneous selective flattening (large layers left direct, small layers flattened).
"""

import time
import argparse
import torch
import torch.nn as nn
from apexw import ApexW, flatten_parameters


class SmallMLP(nn.Module):
    """Deep network with many small layers (e.g. 100 layers)."""
    def __init__(self, n_layers: int = 100, dim: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class HeterogeneousModel(nn.Module):
    """Model with both large dense matrices and many small normalization/bias weights."""
    def __init__(self, n_blocks: int = 50, d_model: int = 512):
        super().__init__()
        self.large_proj = nn.Linear(d_model, d_model, bias=False)  # 262,144 elements (large)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])  # 50 x 2 x 512 (small)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(n_blocks)])  # 50 x 512 (small)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.large_proj(x)
        for norm, bias in zip(self.norms, self.biases):
            x = norm(x) + bias
        return x


def time_fn(fn, steps: int = 30, warmup: int = 10, is_cuda: bool = False) -> float:
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def run_benchmark(n_layers: int = 100, dim: int = 16, steps: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    print("=" * 76)
    print(" APEXW BENCHMARK: SMALL TENSOR PARAMETER FLATTENING")
    print(f" Target Device        : {device.type.upper()}")
    print(f" Model Architecture   : {n_layers}-Layer MLP (Linear {dim}x{dim})")
    print(f" Total Parameters     : {n_layers * 2:,} individual tensors (weights + biases)")
    print(f" Benchmark Iterations : {steps} steps (after 10 warmup steps)")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. Setup Models
    # ------------------------------------------------------------------
    m_adam_unflat = SmallMLP(n_layers=n_layers, dim=dim).to(device)
    m_adam_foreach = SmallMLP(n_layers=n_layers, dim=dim).to(device)
    m_apex_unflat = SmallMLP(n_layers=n_layers, dim=dim).to(device)
    m_apex_flat = SmallMLP(n_layers=n_layers, dim=dim).to(device)

    # Copy identical initial weights
    sd = m_adam_unflat.state_dict()
    m_adam_foreach.load_state_dict(sd)
    m_apex_unflat.load_state_dict(sd)
    m_apex_flat.load_state_dict(sd)

    # Flatten the target model
    flat_param = flatten_parameters(m_apex_flat)

    # Initialize optimizers
    # If CUDA is available, test PyTorch fused AdamW; otherwise default AdamW
    try:
        opt_adam_unflat = torch.optim.AdamW(m_adam_unflat.parameters(), lr=1e-3, fused=is_cuda)
        adam_label = "PyTorch AdamW (Fused)" if is_cuda else "PyTorch AdamW (Sequential)"
    except Exception:
        opt_adam_unflat = torch.optim.AdamW(m_adam_unflat.parameters(), lr=1e-3)
        adam_label = "PyTorch AdamW (Sequential)"

    opt_adam_foreach = torch.optim.AdamW(m_adam_foreach.parameters(), lr=1e-3, foreach=True)
    opt_apex_unflat = ApexW(m_apex_unflat.parameters(), lr=1e-3, foreach=False)
    opt_apex_flat = ApexW([flat_param], lr=1e-3)

    x = torch.randn(8, dim, device=device)

    # Initialize gradients
    for m in (m_adam_unflat, m_adam_foreach, m_apex_unflat, m_apex_flat):
        m(x).sum().backward()

    # ------------------------------------------------------------------
    # 2. Pure Optimizer Step Latency (ms/step)
    # ------------------------------------------------------------------
    t_adam_unflat = time_fn(lambda: opt_adam_unflat.step(), steps=steps, is_cuda=is_cuda)
    t_adam_foreach = time_fn(lambda: opt_adam_foreach.step(), steps=steps, is_cuda=is_cuda)
    t_apex_unflat = time_fn(lambda: opt_apex_unflat.step(), steps=steps, is_cuda=is_cuda)
    t_apex_flat = time_fn(lambda: opt_apex_flat.step(), steps=steps, is_cuda=is_cuda)

    # ------------------------------------------------------------------
    # 3. End-to-End Training Step (Forward + Backward + Optimizer)
    # ------------------------------------------------------------------
    def e2e_step(model, opt):
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        opt.step()

    e2e_adam_unflat = time_fn(lambda: e2e_step(m_adam_unflat, opt_adam_unflat), steps=steps, is_cuda=is_cuda)
    e2e_adam_foreach = time_fn(lambda: e2e_step(m_adam_foreach, opt_adam_foreach), steps=steps, is_cuda=is_cuda)
    e2e_apex_unflat = time_fn(lambda: e2e_step(m_apex_unflat, opt_apex_unflat), steps=steps, is_cuda=is_cuda)
    e2e_apex_flat = time_fn(lambda: e2e_step(m_apex_flat, opt_apex_flat), steps=steps, is_cuda=is_cuda)

    # ------------------------------------------------------------------
    # 4. Print Results Table
    # ------------------------------------------------------------------
    print("\n--- 1. PURE OPTIMIZER STEP LATENCY (100 Small Layers = 200 Tensors) ---")
    print(f" 1. {adam_label:<28} : {t_adam_unflat * 1000:7.4f} ms  (Baseline 1.00x)")
    print(f" 2. PyTorch AdamW (Foreach)      : {t_adam_foreach * 1000:7.4f} ms  ({t_adam_unflat / t_adam_foreach:5.2f}x)")
    print(f" 3. ApexW (Unflattened, 200 tns) : {t_apex_unflat * 1000:7.4f} ms  ({t_adam_unflat / t_apex_unflat:5.2f}x)")
    print(f" 4. ApexW Flattened (1 Unified)  : {t_apex_flat * 1000:7.4f} ms  ({t_adam_unflat / t_apex_flat:5.2f}x vs AdamW, {t_apex_unflat / t_apex_flat:5.2f}x vs ApexW unflat)")

    step_savings = (t_apex_unflat - t_apex_flat) / t_apex_unflat * 100.0
    print(f" >> Speedup of Flattening on ApexW: {t_apex_unflat / t_apex_flat:.1f}x ({step_savings:.2f}% reduction in step time)")

    print("\n--- 2. END-TO-END TRAINING LATENCY (Fwd + Bwd + Step) ---")
    print(f" 1. {adam_label:<28} : {e2e_adam_unflat * 1000:7.4f} ms  (Baseline 1.00x)")
    print(f" 2. PyTorch AdamW (Foreach)      : {e2e_adam_foreach * 1000:7.4f} ms  ({e2e_adam_unflat / e2e_adam_foreach:5.2f}x)")
    print(f" 3. ApexW (Unflattened)          : {e2e_apex_unflat * 1000:7.4f} ms  ({e2e_adam_unflat / e2e_apex_unflat:5.2f}x)")
    print(f" 4. ApexW Flattened (1 Unified)  : {e2e_apex_flat * 1000:7.4f} ms  ({e2e_adam_unflat / e2e_apex_flat:5.2f}x)")

    # ------------------------------------------------------------------
    # 5. Heterogeneous Selective Flattening Demo
    # ------------------------------------------------------------------
    print("\n--- 3. HETEROGENEOUS ARCHITECTURES: SELECTIVE FLATTENING ---")
    het_model = HeterogeneousModel(n_blocks=50, d_model=256).to(device)
    raw_param_count = len(list(het_model.parameters()))
    print(f" Raw Heterogeneous Model Parameters : {raw_param_count} tensors")
    print("   - 1 Large Projection (256x256 = 65,536 elements)")
    print("   - 50 LayerNorm Weights + 50 LayerNorm Biases + 50 Residual Biases (150 small tensors)")

    # Flatten only small tensors (<= 1,024 elements)
    flatten_parameters(het_model, max_numel=1024)
    packed_param_count = len(list(het_model.parameters()))
    print(f" Post-Flattening Parameters          : {packed_param_count} tensors")
    print("   - 1 Direct Large Projection (optimal for fused 2D kernels)")
    print("   - 1 Unified Flat Buffer (packing all 150 small norms & biases)")

    het_x = torch.randn(4, 256, device=device)
    het_opt = ApexW(het_model.parameters(), lr=1e-3)
    het_model(het_x).sum().backward()
    t_het = time_fn(lambda: het_opt.step(), steps=steps, is_cuda=is_cuda)
    print(f" Selective Heterogeneous Step Time   : {t_het * 1000:.4f} ms")
    print("=" * 76)

    # Save summary report
    with open("benchmarks/results_flattening.txt", "w") as f:
        f.write("APEXW PARAMETER FLATTENING BENCHMARK RESULTS\n")
        f.write("============================================\n")
        f.write(f"Architecture: {n_layers}-layer MLP ({dim}x{dim})\n")
        f.write(f"AdamW Step Latency            : {t_adam_unflat*1000:.4f} ms\n")
        f.write(f"ApexW Unflattened Step Latency: {t_apex_unflat*1000:.4f} ms\n")
        f.write(f"ApexW Flattened Step Latency  : {t_apex_flat*1000:.4f} ms\n")
        f.write(f"Speedup vs Unflattened ApexW  : {t_apex_unflat / t_apex_flat:.2f}x\n")
        f.write(f"Speedup vs AdamW              : {t_adam_unflat / t_apex_flat:.2f}x\n")
        f.write(f"End-to-End Speedup vs AdamW   : {e2e_adam_unflat / e2e_apex_flat:.2f}x\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=100)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    run_benchmark(n_layers=args.layers, dim=args.dim, steps=args.steps)
