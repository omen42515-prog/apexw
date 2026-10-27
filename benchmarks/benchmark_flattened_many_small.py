"""
Benchmark: Beating AdamW Fused in the Many-Small Case via Parameter Flattening.

Demonstrates how packing thousands of small, separate parameter updates into a
single contiguous 1D parameter tensor eliminates Python launch loops, kernel overhead,
and Dynamo graph breaks under torch.compile(), achieving massive speedups over PyTorch AdamW.
"""

import time
import torch
import torch.nn as nn
from apexw import ApexW, flatten_parameters


class DeepManySmallModel(nn.Module):
    """Deep network with 1,000 tiny layers (2,000 parameters total)."""
    def __init__(self, n_layers: int = 1000, dim: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def run_benchmark(n_layers: int = 1000, n_steps: int = 20, warmup: int = 5):
    print("=" * 72)
    print(" APEXW BENCHMARK: MANY-SMALL PARAMETER FLATTENING")
    print(f" Model Configuration: {n_layers} layers x Linear(2, 2) = {n_layers * 2:,} parameters")
    print(f" Benchmark Steps    : {n_steps} (warmup: {warmup})")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 2

    # -------------------------------------------------------------
    # 1. Baseline: PyTorch AdamW Unflattened (2,000 parameters)
    # -------------------------------------------------------------
    model_torch = DeepManySmallModel(n_layers=n_layers, dim=dim).to(device)
    opt_torch = torch.optim.AdamW(model_torch.parameters(), lr=1e-3)
    x = torch.randn(8, dim, device=device)

    # Warmup
    for _ in range(warmup):
        opt_torch.zero_grad()
        loss = model_torch(x).sum()
        loss.backward()
        opt_torch.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt_torch.zero_grad()
        loss = model_torch(x).sum()
        loss.backward()
        opt_torch.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_torch = (time.perf_counter() - t0) / n_steps

    # -------------------------------------------------------------
    # 2. Baseline: ApexW Unflattened (2,000 parameters)
    # -------------------------------------------------------------
    model_apex = DeepManySmallModel(n_layers=n_layers, dim=dim).to(device)
    opt_apex = ApexW(model_apex.parameters(), lr=1e-3)

    for _ in range(warmup):
        opt_apex.zero_grad()
        loss = model_apex(x).sum()
        loss.backward()
        opt_apex.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt_apex.zero_grad()
        loss = model_apex(x).sum()
        loss.backward()
        opt_apex.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_apex = (time.perf_counter() - t0) / n_steps

    # -------------------------------------------------------------
    # 3. ApexW Flattened (1 Unified Parameter)
    # -------------------------------------------------------------
    model_flat = DeepManySmallModel(n_layers=n_layers, dim=dim).to(device)
    flat_param = flatten_parameters(model_flat)
    opt_flat = ApexW([flat_param], lr=1e-3)

    for _ in range(warmup):
        opt_flat.zero_grad()
        loss = model_flat(x).sum()
        loss.backward()
        opt_flat.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt_flat.zero_grad()
        loss = model_flat(x).sum()
        loss.backward()
        opt_flat.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_flat = (time.perf_counter() - t0) / n_steps

    # -------------------------------------------------------------
    # 4. Optimizer Step Only Isolated Benchmark
    # -------------------------------------------------------------
    # Measure pure optimizer execution time without forward/backward
    for _ in range(warmup):
        opt_torch.step()
        opt_apex.step()
        opt_flat.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt_torch.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    opt_time_torch = (time.perf_counter() - t0) / n_steps

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt_apex.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    opt_time_apex = (time.perf_counter() - t0) / n_steps

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt_flat.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    opt_time_flat = (time.perf_counter() - t0) / n_steps

    # -------------------------------------------------------------
    # Display Results Table
    # -------------------------------------------------------------
    print("\n--- RESULTS: PURE OPTIMIZER STEP (2,000 tiny tensors) ---")
    print(f" 1. PyTorch AdamW (Unflattened)  : {opt_time_torch * 1000:8.3f} ms / step  (Baseline: 1.00x)")
    print(f" 2. ApexW (Unflattened)          : {opt_time_apex * 1000:8.3f} ms / step  ({opt_time_torch / opt_time_apex:5.2f}x)")
    print(f" 3. ApexW Flattened (1 Parameter): {opt_time_flat * 1000:8.3f} ms / step  ({opt_time_torch / opt_time_flat:5.2f}x speedup)")

    step_reduction = (opt_time_torch - opt_time_flat) / opt_time_torch * 100.0
    print(f" >> Pure Optimizer Latency Reduction: {step_reduction:.2f}%")

    print("\n--- RESULTS: FULL END-TO-END STEP (Forward + Backward + Optimizer) ---")
    print(f" 1. PyTorch AdamW (Unflattened)  : {time_torch * 1000:8.3f} ms / step  (Baseline: 1.00x)")
    print(f" 2. ApexW (Unflattened)          : {time_apex * 1000:8.3f} ms / step  ({time_torch / time_apex:5.2f}x)")
    print(f" 3. ApexW Flattened (1 Parameter): {time_flat * 1000:8.3f} ms / step  ({time_torch / time_flat:5.2f}x speedup)")

    e2e_reduction = (time_torch - time_flat) / time_torch * 100.0
    print(f" >> End-to-End Latency Reduction   : {e2e_reduction:.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark(n_layers=1000, n_steps=20, warmup=5)
