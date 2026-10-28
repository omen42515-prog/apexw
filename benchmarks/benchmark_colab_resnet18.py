import sys
import os
# Ensure the local apexw installation is visible to the interpreter
sys.path.insert(0, '/content/apexw')
if os.path.exists('/content/apexw/apexw'):
    sys.path.insert(0, '/content/apexw/apexw')

import copy
import gc
import random
import statistics
import time
import warnings

import torch
import torch.nn as nn

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
MODEL_DIM = 2048
HIDDEN_DIM = MODEL_DIM * 4
BATCH_SIZE = 64

TRAIN_EXAMPLES = 2048
VAL_EXAMPLES = 512

WARMUP_STEPS = 30
STEPS_PER_TRIAL = 200
# Reduced trials from 5 to 3 for faster validation feedback loops
NUM_TRIALS = 3

LR = 1e-3
WEIGHT_DECAY = 1e-2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    raise RuntimeError(
        "This benchmark requires a CUDA GPU. "
        "No CUDA device is available."
    )

torch.manual_seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Keep these fixed across all optimizers.
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print("=" * 88)
print("APEXW OPTIMIZER BENCHMARK — CONTROLLED GPU RUN")
print("=" * 88)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"Model dimension: {MODEL_DIM}")
print(f"Hidden dimension: {HIDDEN_DIM}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Warm-up: {WARMUP_STEPS}")
print(f"Timed steps/trial: {STEPS_PER_TRIAL}")
print(f"Trials: {NUM_TRIALS}")
print(f"Learning rate: {LR}")
print(f"Weight decay: {WEIGHT_DECAY}")
print("AMP: disabled")
print()

# ============================================================
# 2. FIXED TRAINING AND VALIDATION DATA
# ============================================================

data_gen = torch.Generator(device="cpu")
data_gen.manual_seed(SEED + 100)

teacher_gen = torch.Generator(device="cpu")
teacher_gen.manual_seed(SEED + 200)

teacher = torch.randn(
    MODEL_DIM, MODEL_DIM, generator=teacher_gen
) / (MODEL_DIM ** 0.5)


def make_dataset(n):
    x = torch.randn(n, MODEL_DIM, generator=data_gen)
    y = x @ teacher

    # Per-example normalization, similar to LayerNorm.
    y = (y - y.mean(dim=-1, keepdim=True)) / (
        y.var(dim=-1, unbiased=False, keepdim=True) + 1e-5
    ).sqrt()

    return x.to(DEVICE), y.to(DEVICE)


train_x, train_y = make_dataset(TRAIN_EXAMPLES)
val_x, val_y = make_dataset(VAL_EXAMPLES)

del teacher

# Identical, reproducible batch order for each trial.
batch_gen = torch.Generator(device="cpu")
batch_gen.manual_seed(SEED + 300)

batches_per_trial = (
    WARMUP_STEPS + STEPS_PER_TRIAL
)  # one batch index per step

batch_indices = torch.randint(
    0,
    TRAIN_EXAMPLES,
    (
        NUM_TRIALS,
        batches_per_trial,
        BATCH_SIZE,
    ),
    generator=batch_gen,
)


def get_batch(trial, step):
    idx = batch_indices[trial, step].to(DEVICE)
    return train_x[idx], train_y[idx]


# ============================================================
# 3. MODEL AND IDENTICAL INITIAL WEIGHTS
# ============================================================

def build_model():
    return nn.Sequential(
        nn.Linear(MODEL_DIM, HIDDEN_DIM),
        nn.GELU(),
        nn.Linear(HIDDEN_DIM, MODEL_DIM),
        nn.LayerNorm(MODEL_DIM),
    ).to(DEVICE)


# Create a single reference initialization.
torch.manual_seed(SEED + 400)
torch.cuda.manual_seed_all(SEED + 400)

reference_model = build_model()
initial_state = copy.deepcopy(reference_model.state_dict())
del reference_model


# ============================================================
# 4. OPTIMIZER FACTORIES
# ============================================================

def build_optimizer_configs():
    configs = {
        # PyTorch Adam uses coupled L2 weight decay here.
        "Adam": lambda model: torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            fused=False,
        ),

        "AdamW": lambda model: torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            fused=False,
        ),

        "AdamW Fused": lambda model: torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            fused=True,
        ),
    }

    try:
        from optimizer import ApexW

        # Fair comparison: match AdamW's default weight-decay behavior and full FP32 precision.
        # decay_norm_only=False means all params are decayed (same as AdamW default).
        # precision="full" keeps moments in FP32 (same as AdamW).
        configs["ApexW"] = lambda model: ApexW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            decay_norm_only=False,
            precision="full",
            dynamic_scaling=False,
        )

        # Production-optimized variant: compact 16-bit moments + norm-only decay.
        # This shows the memory/speed tradeoff when ApexW's own defaults are used.
        configs["ApexW (Optimized)"] = lambda model: ApexW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    except ImportError:
        print("WARNING: apexw.ApexW unavailable; skipping ApexW.")

    try:
        from lion_pytorch import Lion

        configs["Lion"] = lambda model: Lion(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    except ImportError:
        print("WARNING: lion_pytorch.Lion unavailable; skipping Lion.")

    return configs


# ============================================================
# 5. METRICS
# ============================================================

criterion = nn.MSELoss()


def optimizer_state_mb(optimizer):
    total = 0

    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()

    return total / (1024 ** 2)


def model_and_gradient_finite(model):
    for p in model.parameters():
        if not torch.isfinite(p).all().item():
            return False

        if p.grad is not None:
            if not torch.isfinite(p.grad).all().item():
                return False

    return True


@torch.no_grad()
def evaluate(model):
    model.eval()
    total_squared_error = 0.0
    total_elements = 0

    for start in range(0, VAL_EXAMPLES, BATCH_SIZE):
        x = val_x[start:start + BATCH_SIZE]
        y = val_y[start:start + BATCH_SIZE]

        pred = model(x)

        total_squared_error += (
            (pred - y).square().sum().item()
        )
        total_elements += y.numel()

    model.train()
    return total_squared_error / total_elements


# ============================================================
# 6. RUN ONE TRIAL
# ============================================================

def run_trial(name, factory, trial_index):
    model = build_model()
    model.load_state_dict(initial_state)
    model.train()

    optimizer = factory(model)

    # Fixed batch sequence, identical across optimizers.
    def train_step(step):
        x, y = get_batch(trial_index, step)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = criterion(pred, y)

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training loss")

        loss.backward()

        if not model_and_gradient_finite(model):
            raise FloatingPointError(
                "Non-finite parameter or gradient"
            )

        optimizer.step()

        if not model_and_gradient_finite(model):
            raise FloatingPointError(
                "Non-finite parameter after optimizer step"
            )

        return loss.detach()

    # Warm-up includes lazy optimizer-state initialization.
    for step in range(WARMUP_STEPS):
        train_step(step)

    torch.cuda.synchronize()

    state_mb = optimizer_state_mb(optimizer)

    # Measure steady-state peak memory after warm-up.
    # Peak total includes live model, gradients, and optimizer state.
    # Peak extra is the increase over the live baseline.
    torch.cuda.reset_peak_memory_stats(DEVICE)
    allocated_before = torch.cuda.memory_allocated(DEVICE)
    reserved_before = torch.cuda.memory_reserved(DEVICE)

    trial_ms = []
    final_train_loss = float("nan")

    try:
        for _ in range(NUM_TRIALS):
            # The caller selects the trial. This loop is intentionally
            # not used; each invocation performs one timed trial.
            break

        torch.cuda.synchronize()
        start = time.perf_counter()

        for step in range(
            WARMUP_STEPS,
            WARMUP_STEPS + STEPS_PER_TRIAL,
        ):
            loss = train_step(step)
            final_train_loss = loss.item()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        trial_ms.append(
            1000.0 * elapsed / STEPS_PER_TRIAL
        )

        # Record the actual validation loss after timed training.
        val_loss = evaluate(model)

        torch.cuda.synchronize()

        peak_allocated = torch.cuda.max_memory_allocated(DEVICE)
        peak_reserved = torch.cuda.max_memory_reserved(DEVICE)

        result = {
            "ms_per_step": statistics.median(trial_ms),
            "steps_per_second": 1000.0 / statistics.median(trial_ms),
            "state_mb": state_mb,
            "train_loss": final_train_loss,
            "val_loss": val_loss,
            "parameters_finite": model_and_gradient_finite(model),
            "peak_total_allocated_mb": peak_allocated / (1024 ** 2),
            "peak_extra_allocated_mb": (
                peak_allocated - allocated_before
            ) / (1024 ** 2),
            "peak_total_reserved_mb": peak_reserved / (1024 ** 2),
            "extra_reserved_mb": (
                peak_reserved - reserved_before
            ) / (1024 ** 2),
        }

        return result

    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# 7. REPEATED TRIALS
# ============================================================

configs = build_optimizer_configs()
all_results = {}

for name, factory in configs.items():
    print(f"\nBenchmarking {name} ...")
    all_results[name] = []

    for trial in range(NUM_TRIALS):
        print(f"  Trial {trial + 1}/{NUM_TRIALS}", flush=True)

        try:
            result = run_trial(name, factory, trial)
            all_results[name].append(result)

            print(
                f"    {result['steps_per_second']:.2f} steps/s | "
                f"validation loss {result['val_loss']:.6f} | "
                f"peak allocated {result['peak_total_allocated_mb']:.1f} MB"
            )

        except Exception as exc:
            warnings.warn(
                f"{name}, trial {trial + 1} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            all_results[name].append({
                "error": f"{type(exc).__name__}: {exc}"
            })

        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# 8. SUMMARY — MEDIAN ACROSS SUCCESSFUL TRIALS
# ============================================================

print("\n" + "=" * 124)
print("FINAL RESULTS — MEDIAN ACROSS SUCCESSFUL TRIALS")
print("=" * 124)

print(
    f"{'Optimizer':<18}"
    f"{'Steps/s':>11}"
    f"{'ms/step':>11}"
    f"{'State MB':>12}"
    f"{'Val Loss':>14}"
    f"{'Peak Alloc MB':>16}"
    f"{'Peak Extra MB':>16}"
    f"{'Finite':>9}"
)
print("-" * 124)

for name, trials in sorted(all_results.items()):
    good = [r for r in trials if "error" not in r]

    if not good:
        print(f"{name:<18} FAILED")
        continue

    median = lambda key: statistics.median(
        r[key] for r in good
    )

    finite = all(r["parameters_finite"] for r in good)

    print(
        f"{name:<18}"
        f"{median('steps_per_second'):>11.2f}"
        f"{median('ms_per_step'):>11.3f}"
        f"{median('state_mb'):>12.2f}"
        f"{median('val_loss'):>14.6f}"
        f"{median('peak_total_allocated_mb'):>16.2f}"
        f"{median('peak_extra_allocated_mb'):>16.2f}"
        f"{str(finite):>9}"
    )

print("=" * 124)
print("Benchmark complete.")
print("Equal hyperparameters do not guarantee equal convergence.")
print("Adam uses coupled L2 decay; AdamW and Lion generally use decoupled decay.")
