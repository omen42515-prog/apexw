import os
import sys
import time
import math
import argparse
import statistics
from typing import List, Dict
import torch
import torch.nn as nn
import apexw

# Let's locate the full benchmark structure
def choose_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def benchmark_one(name, args, device, seed):
    torch.manual_seed(seed)
    # Synthesize standard model tensors to approximate total parameter target
    # Target elements = millions * 10^6
    target_numel = int(args.millions * 1e6)

    # Create a dummy model layer with approximate weights
    dim = int(math.sqrt(target_numel))
    # Ensure we have about target_numel parameters
    layer = nn.Linear(dim, dim, bias=False).to(device)
    params = list(layer.parameters())
    actual_numel = sum(p.numel() for p in params)

    # Generate fake gradients to run step() under matching constraints
    for p in params:
        p.grad = torch.randn_like(p)

    # Warmup allocation track
    if device.type == "cuda":
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
    else:
        baseline_allocated = 0

    # Instantiate corresponding optimizer
    if name == "AdamW-native":
        optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    elif name == "AdamW-fused":
        optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, fused=True)
    elif name == "ApexW":
        optimizer = apexw.ApexW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
                                 precision=args.precision, backend=args.backend)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    # Warmup runs
    for _ in range(args.warmup):
        optimizer.step()

    if device.type == "cuda":
        sync(device)
        state_mb = (torch.cuda.memory_allocated(device) - baseline_allocated) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
    else:
        state_mb = 0.0

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        sync(device)

        wall_start = time.perf_counter()
        start_event.record()
        for _ in range(args.steps):
            optimizer.step()
        end_event.record()
        sync(device)

        wall_elapsed = time.perf_counter() - wall_start
        device_elapsed_ms = start_event.elapsed_time(end_event)
        wall_ms_step = wall_elapsed * 1000.0 / args.steps
        device_ms_step = device_elapsed_ms / args.steps
        peak_extra_mb = max(0, torch.cuda.max_memory_allocated(device) - baseline_allocated) / (1024 ** 2)
    else:
        sync(device)
        wall_start = time.perf_counter()
        for _ in range(args.steps):
            optimizer.step()
        sync(device)
        wall_elapsed = time.perf_counter() - wall_start
        wall_ms_step = wall_elapsed * 1000.0 / args.steps
        device_ms_step = float("nan")
        peak_extra_mb = float("nan")

    result = {
        "wall_ms_step": wall_ms_step,
        "device_ms_step": device_ms_step,
        "state_mb": state_mb,
        "peak_extra_mb": peak_extra_mb,
        "parameters": float(actual_numel),
    }

    del optimizer, params
    if device.type == "cuda":
        sync(device)
        torch.cuda.empty_cache()
    return result

def median_result(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    output = {}
    for key in keys:
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        output[key] = statistics.median(vals) if vals else float("nan")
    return output

# Inline argument mock object to simulate argparse
class DummyArgs:
    def __init__(self):
        self.device = "auto"
        self.millions = 8.0
        self.warmup = 30
        self.steps = 200
        self.repeats = 3
        self.lr = 1e-3
        self.backend = "auto"
        self.precision = "auto"

args = DummyArgs()
device = choose_device(args.device)
apexw_version = getattr(apexw, "__version__", "unknown")

print("=" * 78)
print("ApexW controlled optimizer-step benchmark")
print(f"Python            : {sys.version.split()[0]}")
print(f"PyTorch           : {torch.__version__}")
print(f"ApexW             : {apexw_version}")
print(f"Device            : {device}")
if device.type == "cuda":
    print(f"GPU               : {torch.cuda.get_device_name(device)}")
    print(f"CUDA runtime       : {torch.version.cuda}")
print(f"Parameters target : {args.millions:g} million")
print(f"Warmup / timed    : {args.warmup} / {args.steps} steps; {args.repeats} repeats")
print("Matched hyperparams: lr=%.4g, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01" % args.lr)
print(f"ApexW config      : precision={args.precision}, backend={args.backend}")
print("Metric note       : optimizer.step() only; pre-generated fixed gradients, no forward/backward")
print("=" * 78)

names = ["AdamW-native"]
if device.type == "cuda":
    names.append("AdamW-fused")
names.append("ApexW")

samples = {name: [] for name in names}

for repeat in range(args.repeats):
    order = names[repeat % len(names):] + names[:repeat % len(names)]
    print(f"Repeat {repeat + 1}/{args.repeats}: order = {', '.join(order)}")
    for index, name in enumerate(order):
        result = benchmark_one(name, args, device, seed=20261028 + repeat * 100)
        samples[name].append(result)
        print(
            "  %-14s wall=%8.4f ms/step | state=%8.2f MiB | extra_peak=%s"
            % (
                name,
                result["wall_ms_step"],
                result["state_mb"],
                ("%.2f MiB" % result["peak_extra_mb"]) if not math.isnan(result["peak_extra_mb"]) else "n/a",
            )
        )

medians = {name: median_result(rows) for name, rows in samples.items()}
print("\nMEDIAN RESULTS")
print("Optimizer          wall ms/step    steps/s      state MiB    peak extra MiB")
print("-" * 78)
for name in names:
    r = medians[name]
    sps = 1000.0 / r["wall_ms_step"] if r["wall_ms_step"] > 0 else float("nan")
    peak = "n/a" if math.isnan(r["peak_extra_mb"]) else f"{r['peak_extra_mb']:.2f}"
    print(f"{name:<18} {r['wall_ms_step']:>11.4f} {sps:>11.2f} {r['state_mb']:>13.2f} {peak:>17}")

apex = medians["ApexW"]
for baseline_name in names:
    if baseline_name == "ApexW":
        continue
    base = medians[baseline_name]
    speedup = base["wall_ms_step"] / apex["wall_ms_step"]
    reduction = 100.0 * (base["state_mb"] - apex["state_mb"]) / base["state_mb"] if base["state_mb"] else float("nan")
    print(f"ApexW vs {baseline_name}: speedup={speedup:.3f}x; optimizer-state reduction={reduction:.1f}%")
