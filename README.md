# ApexW

ApexW is a memory-efficient alternative to PyTorch's `torch.optim.AdamW`. By default, it cuts optimizer-state memory by roughly 50% while matching standard AdamW convergence and training throughput across modern hardware (NVIDIA GPUs, Apple Silicon Metal, AMD ROCm, and CPUs).

> [!NOTE]
> **Development Disclosure:** ApexW was designed and developed with the assistance of AI coding tools, alongside human architectural direction and coding , systematic code review, benchmark validation, and an extensive test suite.

---

## What is an optimizer, and why does it use so much memory?

If you are new to deep learning, you might wonder why an optimizer needs any memory at all.

When you train a neural network, your data passes through the model (the *forward pass*), you measure how wrong the predictions are (the *loss*), and PyTorch calculates which direction to adjust each parameter (the *gradients* in the *backward pass*).

The **optimizer's** job is to apply those adjustments to the model's weights.

Simple optimizers like basic SGD (Stochastic Gradient Descent) update weights directly from the current gradient. But modern optimizers like **AdamW** keep a running history for every single weight in your model to make training faster and more stable:

1. **First Moment (`exp_avg`, or momentum):** A rolling average of recent gradient directions. This helps the optimizer keep moving steadily in consistent directions, smoothing out noisy batch-to-batch variations.
2. **Second Moment (`exp_avg_sq`, or variance):** A rolling average of recent gradient *magnitudes* (squared). This scales the step size for each parameter individually—taking smaller, cautious steps for weights with wildly changing gradients and larger steps for calm ones.

These two tracking buffers are called the **optimizer state**.

### The Memory Problem
In standard PyTorch `AdamW`, both the first moment and second moment are stored as full 32-bit floating-point numbers (`float32`), which take 4 bytes each:

$$\text{Memory per parameter} = 4\text{ bytes (momentum)} + 4\text{ bytes (variance)} = \mathbf{8\text{ bytes per parameter}}$$

This adds up fast:
* For a **1-billion parameter model**, AdamW states consume **8 GB of GPU memory (VRAM)**.
* For a **7-billion parameter model**, AdamW states consume **56 GB of VRAM**—before accounting for the model weights, gradients, or input data.

**ApexW reduces this memory** by storing those tracking buffers in smaller, compressed formats (such as 16-bit or 8-bit representations) while computing the actual parameter updates in high precision.

---

## Quickstart

ApexW shares the standard PyTorch optimizer interface. In existing training scripts, you can swap `torch.optim.AdamW` with `ApexW`:

```python
import torch
import torch.nn as nn
from apexw import ApexW

# Define your model as usual
model = nn.Linear(512, 10).to("cuda")

# Use ApexW instead of torch.optim.AdamW
optimizer = ApexW(model.parameters(), lr=1e-3)

# Standard PyTorch training loop
for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### The Only Code Changes You Need (Before vs. After)

You only need to change **two lines of code** in your entire script:

```python
# ==========================================
# BEFORE (Standard PyTorch AdamW):
# ==========================================
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# ==========================================
# AFTER (ApexW — choose one of the 3 modes):
# ==========================================
from apexw import ApexW

# Mode 1: Default (Compact — 50% state memory savings for any model)
optimizer = ApexW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Mode 2: Low-VRAM Fine-Tuning (8-Bit — 75% state memory savings for QLoRA/Unsloth)
optimizer = ApexW(model.parameters(), lr=2e-4, precision="8bit")

# Mode 3: Pretraining from scratch (Hybrid Muon + ApexW for Transformers)
optimizer = ApexW(model, lr=1e-3, use_muon=True)
```

Everything else in your training loop (`zero_grad()`, `backward()`, `step()`, learning-rate schedulers, and `torch.autocast`) stays **100% identical**.

> [!IMPORTANT]
> **API vs. Hyperparameter Defaults:** While ApexW uses the same API as PyTorch's `AdamW`, its **default hyperparameters differ slightly** for better modern training efficiency (for example, `beta2=0.99` instead of `0.999`, and excluding 1D bias/norm parameters from weight decay). If you want an exact match to PyTorch defaults, see [How ApexW Defaults Differ from AdamW](#how-apexw-defaults-differ-from-adamw).

---

## Installation

```bash
pip install apexw
```

Or install locally from source:

```bash
git clone https://github.com/your-username/apexw.git
cd apexw
pip install -e .
```

#### Do I need to install Triton or use `torch.compile`?
* **Triton is NOT a manual prerequisite:** If you have PyTorch installed on a Linux system with an NVIDIA GPU (`pip install torch`), Triton is **already bundled inside PyTorch**. On other platforms (Apple Silicon Metal/MPS, CPU, Windows, Google TPU), ApexW automatically uses PyTorch's native C++ multi-tensor engine (`torch._foreach_*`). You never need to install Triton manually.
* **`torch.compile` is NOT required:** ApexW runs out of the box in standard PyTorch eager mode with instant startup (zero compilation overhead). While fully compatible with `torch.compile()`, you do not need to compile your model to use ApexW.
* **No external third-party dependencies:** Unlike `adamw_bnb_8bit` (which requires `bitsandbytes` and a CUDA compiler), ApexW requires zero external packages.

---

## The 3 Core Modes

Rather than forcing you to tune complex quantization flags, ApexW is organized around **three clear modes**:

| Mode | Memory per Param | Primary Use Case | How to Select |
|---|:---:|---|---|
| **1. ApexW Compact** <br>*(Default)* | **~4 bytes** <br>*(50% memory cut)* | **General training & fine-tuning.** Recommended for most models (CNNs, Transformers, Diffusion, LLMs). Stores moments in 16-bit while executing updates in 32-bit registers. | `optimizer = ApexW(model.parameters(), lr=1e-3)` |
| **2. ApexW 8-Bit** | **~2 bytes** <br>*(75% memory cut)* | **Consumer GPUs & strict VRAM limits.** Designed for Unsloth, PEFT, and QLoRA fine-tuning where memory is constrained. | `optimizer = ApexW(model.parameters(), lr=2e-4, precision="8bit")` |
| **3. Hybrid ApexW** | **2 bytes (2D) + 4 bytes (1D)** | **Pretraining Transformers from scratch.** Combines the Muon optimizer on 2D weight matrices with ApexW on 1D normalization layers and embeddings. | `optimizer = ApexW(model, lr=1e-3, use_muon=True)` |

### Understanding the 3 Modes

#### Mode 1: ApexW Compact (Default)
Standard AdamW stores both moments in 32-bit float (`float32`). ApexW Compact stores them in 16-bit (`bfloat16` or dynamically scaled `float16`). 

During each update step, the values are temporarily loaded into 32-bit registers to calculate math without precision loss, then stored back in 16-bit. This cuts optimizer state memory in half (from 8 bytes to 4 bytes per parameter) with negligible impact on final loss.

#### Mode 2: ApexW 8-Bit (`precision="8bit"`)
When fine-tuning large language models on consumer GPUs (such as an RTX 3060, 4060, or 4090), even 4 bytes per parameter can lead to Out-Of-Memory (OOM) errors.

Mode 2 compresses states down to **2 bytes per parameter** (INT8 for momentum, UINT8 for variance):
* **Why not standard 8-bit AdamW?** Traditional 8-bit optimizers (like `bitsandbytes`) use linear quantization for variance. If a single gradient in a block is large, the scale factor becomes large, which can round small gradients to zero. Dividing by zero (or a tiny number) can cause step sizes to spike unpredictably, occasionally destabilizing training.
* **How ApexW handles it:** ApexW uses **logarithmic quantization** for variance. Similar to the decibel scale, logarithmic intervals represent relative proportions rather than absolute steps. This allows the optimizer to track both very large and very small variance values accurately without underflowing to zero.
* **Stochastic rounding:** When storing momentum in 8-bit, ApexW adds a tiny amount of pseudo-random dithering before rounding. This prevents tiny gradient updates from always rounding down to zero over thousands of steps.

#### Mode 3: Hybrid ApexW (`use_muon=True`)
*Muon* is a specialized optimizer developed for 2D weight matrices (such as the linear projection layers in Transformer attention blocks). It uses matrix orthogonalization (via Newton-Schulz iteration) to produce balanced updates across all singular vectors.

However, 1D parameters (such as LayerNorm scales and biases) and embedding tables do not have 2D matrix structure and require standard AdamW-style scaling. When you pass `use_muon=True` or use `HybridApexW(model)`, ApexW automatically inspects your model and routes:
* **2D linear weights** $\rightarrow$ Muon (fast orthogonal updates)
* **1D layers & embeddings** $\rightarrow$ ApexW Compact (standard adaptive learning rates)

> [!TIP]
> **Want 8-Bit or Hybrid as the permanent default in your repo?**
> If you are working on a dedicated QLoRA or pretraining project and want a specific mode to be the global default without typing extra flags in every script:
> Open [`apexw/optimizer.py`](./apexw/optimizer.py):
> * **To make 8-Bit the default everywhere:** On **Line 1301**, change `precision: Union[str, torch.dtype] = "compact"` to `"8bit"`.
> * **To make Hybrid Muon the default everywhere:** On **Line 1311**, change `use_muon: bool = False` to `True`.
> 
> You can still explicitly override them anytime in your code (e.g. `ApexW(model, precision="compact")`).

---

## How ApexW Defaults Differ from AdamW

ApexW's default hyperparameters were chosen based on modern training practices rather than strict historical adherence to PyTorch's defaults:

| Hyperparameter | ApexW Default | PyTorch `AdamW` Default | Rationale |
|---|:---:|:---:|---|
| `lr` | `1e-3` | `1e-3` | Standard learning rate. |
| `betas` | `(0.9, 0.99)` | `(0.9, 0.999)` | `beta2=0.99` adapts faster to recent gradient changes (~100 step window vs. ~1,000 steps) and protects 16-bit buffers from second-moment stagnation. |
| `eps` | `1e-8` | `1e-8` | Numerical stability denominator constant. |
| `weight_decay` | `1e-2` | `1e-2` | Decoupled weight decay. |
| **1D Weight Decay** | **Excluded** | Included | Standard practice in modern models (e.g., LLaMA, nanoGPT) excludes LayerNorm and bias vectors from weight decay to avoid under-regularizing representation scale. |


---

## Recommended Training Practice: Learning-Rate Warmup

Warmup is a standard best practice in modern deep learning. During warmup, the learning rate starts small and gradually ramps up to its target value over the first few steps.

This prevents aggressive initial updates when weights are random and gradients are noisy. A common rule of thumb is a warmup of **10 to 100 steps** for short runs, or **1% to 5% of total steps** for longer training:

```python
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from apexw import ApexW

optimizer = ApexW(model.parameters(), lr=3e-4, precision="compact")

total_steps = epochs * len(train_loader)
warmup_steps = max(10, min(100, int(0.03 * total_steps)))
decay_steps = max(1, total_steps - warmup_steps)

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],
)

for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

---

## Practical Examples & Integrations

### 1. Mixed Precision (AMP)
ApexW works natively with PyTorch's Automatic Mixed Precision (`torch.autocast`):

```python
# BF16 Autocast (recommended for modern GPUs — no GradScaler needed)
for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()

# FP16 Autocast (use GradScaler as recommended by PyTorch)
scaler = torch.cuda.amp.GradScaler()
for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(inputs), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. Hugging Face Trainer Integration
You can register ApexW as an optimizer option for Hugging Face `Trainer`:

```python
from transformers import Trainer, TrainingArguments
from apexw.hf import register_hf_trainer_optimizer

# Register apexw, apexw_8bit, and hybrid_apexw with Hugging Face
register_hf_trainer_optimizer()

training_args = TrainingArguments(
    output_dir="./results",
    optim="apexw",           # Options: "apexw", "apexw_8bit", or "hybrid_apexw"
    learning_rate=5e-5,
    warmup_ratio=0.03,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### 3. PEFT & QLoRA Fine-Tuning (Unsloth / TRL)
In Parameter-Efficient Fine-Tuning (PEFT) and QLoRA, base model weights are frozen in 4-bit, and small adapter layers (LoRA) are trained. 

* **Supports any LoRA rank ($r=8, 16, 64, 128+$):** Whether you train with standard low ranks or high ranks ($r=64$ or $r=128$), ApexW tracks and updates adapter parameters with the same 50% (Compact) or 75% (8-Bit) state memory cut.
* **Automatic frozen parameter filtering:** Passing your model directly to ApexW automatically ignores frozen base weights and only registers active LoRA adapters:

```python
from apexw import ApexW

# Passing `model` directly automatically finds and optimizes only trainable parameters (e.g. LoRA adapters)
optimizer = ApexW(model, lr=2e-4, precision="8bit")

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

### 4. Multi-GPU & Distributed Training (How Multi-Chip Works)

A common question is: *Does ApexW work on multiple GPUs automatically, what is `ApexWZeRO`, and does it scale to 64, 128, or more GPUs?*

#### Does standard ApexW work across multiple GPUs automatically?
**Yes.** If you are using standard PyTorch multi-GPU tools, `ApexW` works out of the box with zero extra configuration:

* **PyTorch DDP (`DistributedDataParallel`):** DDP runs a copy of your script on each GPU. In the backward pass, PyTorch automatically synchronizes (averages) gradients across all GPUs. Each GPU then calls `optimizer.step()` locally with `ApexW`. It works automatically.
* **PyTorch FSDP (`FullyShardedDataParallel`) & DeepSpeed:** FSDP automatically shards your model weights and gradients across your GPUs. Because each GPU only receives its local slice of the model, standard `ApexW` automatically only allocates optimizer memory for that local slice.
* **Hugging Face Trainer / Accelerate:** When launching with `accelerate launch` or `torchrun`, simply specify `optim="apexw"` (or `optim="apexw_8bit"`). Multi-GPU synchronization is handled automatically under the hood.

#### Does ApexW scale to 64, 128, or more GPUs?
**Yes.** There is no rank limit:
* With **FSDP, DeepSpeed, or DDP**, standard `ApexW` scales across 64, 128, 256, or more GPUs seamlessly.
* With **`ApexWZeRO`**, optimizer states are partitioned across all ranks. As you scale to 64 or 128 GPUs, optimizer state memory per GPU shrinks dramatically:
  * 1 GPU: ~4 bytes / parameter
  * 8 GPUs: ~0.5 bytes / parameter
  * 64 GPUs: ~0.06 bytes / parameter (128× less memory than standard AdamW)
  * 128 GPUs: ~0.03 bytes / parameter

#### What is `ApexWZeRO` then?
In standard PyTorch DDP, although gradients are split across mini-batches, **every GPU keeps a full duplicate copy of the entire optimizer state**:
* If your model's optimizer state requires 4 GB, on 4 GPUs you are using **16 GB of memory total** across the node storing the exact same numbers four times.
* **ZeRO Stage 1 (Zero Redundancy Optimizer)** solves this by slicing the optimizer states across the GPUs: GPU 0 only tracks the first 25% of weights, GPU 1 tracks the next 25%, etc. Each GPU updates its own shard and then shares the updated weights (`all_gather`). On 4 GPUs, each GPU only uses **1/4th** of the optimizer state memory.

Normally, using ZeRO-1 requires installing Microsoft's `deepspeed` library, writing JSON configuration files, and changing your codebase.

**`ApexWZeRO` is an optional, self-contained ZeRO-1 engine built directly into ApexW with zero external dependencies.** You only need it if you are writing custom PyTorch DDP scripts and want ZeRO-1 memory sharding without installing DeepSpeed:

```python
from apexw import ApexWZeRO

# Optional: Only needed for custom PyTorch DDP scripts without FSDP or DeepSpeed
optimizer = ApexWZeRO(
    model.parameters(),
    lr=1e-3,
    precision="compact",
    world_size=torch.distributed.get_world_size(),
    rank=torch.distributed.get_rank(),
)
```

#### Multi-GPU Quick Reference

| Your Multi-GPU Setup | What to Use | Is it Automatic? |
|---|---|:---:|
| **Single GPU or CPU** | `ApexW(model.parameters())` | **Automatic** |
| **Hugging Face / Accelerate / Unsloth** | `optim="apexw"` or `optim="apexw_8bit"` | **Automatic** (Trainer handles multi-GPU) |
| **PyTorch DDP (`DistributedDataParallel`)** | `ApexW(model.parameters())` | **Automatic** (DDP handles communication; each GPU gets ApexW's memory savings) |
| **PyTorch FSDP or DeepSpeed ZeRO-3** | `ApexW(model.parameters())` | **Automatic** (FSDP/DeepSpeed handles sharding) |
| **Custom DDP script wanting ZeRO-1 without DeepSpeed** | `ApexWZeRO(model.parameters())` | **Manual opt-in** (gives you ZeRO-1 state sharding with zero third-party packages) |

---

## Optional Features for Advanced Users

> [!NOTE]
> **Most users should stick to the defaults.** The features below are intended for researchers and specialized experimental setups:

* `clip_grad_norm=float`: In-optimizer gradient clipping (computes the global gradient norm and clips inside the optimizer kernel).
* `cautious=True`: Cautious update masking (updates parameters only when the current gradient agrees in sign with the momentum buffer).
* `grad_centralization=True`: Centers gradients to zero mean before updating (useful in some convolutional architectures).
* `beta3=float`, `alpha=float`: AdEMAMix slow momentum (adds an ultra-slow third EMA buffer to track long-term gradient trends).
* `lion=True`: Uses sign-based updates (from Google's Lion optimizer) rather than magnitude scaling.

---

## Hardware and Backend Support

ApexW detects your hardware and automatically selects the most efficient execution path:

| Device | Execution Paths | Notes |
|---|---|---|
| **NVIDIA CUDA** | OpenAI Triton, PyTorch Foreach, or C++/CUDA | Fuses momentum, variance, and weight updates into single SRAM passes. |
| **AMD ROCm** | OpenAI Triton or PyTorch Foreach | Fully supported via PyTorch ROCm. |
| **Apple Silicon (MPS)** | Vectorized Multi-Tensor operations | Native Apple Metal GPU execution. |
| **Google TPU / XLA** | PyTorch/XLA backend | Available when `torch_xla` is present. |
| **CPU** | PyTorch Foreach multi-tensor operations | Works everywhere without CUDA dependencies. |

---

## Benchmark Results

### 1. GPU: Unsloth QLoRA Fine-Tuning (Llama-3.2-1B, 500 Steps)
Measured during 4-bit quantized base model instruction fine-tuning on an NVIDIA GPU:

| Optimizer | Learning Rate | Peak VRAM | Steady Steps/Sec | Final Loss | Dependencies |
|---|:---:|:---:|:---:|:---:|---|
| **8-Bit AdamW (BitsAndBytes)** | 2e-4 | 2.352 GB | 4.147 | 1.3892 | Requires `bitsandbytes`, CUDA only |
| **ApexW Compact (Default)** | 2e-4 | 2.373 GB | **4.294** | 1.3932 | Pure PyTorch + Triton |
| **ApexW 8-Bit** | 2e-4 | **2.352 GB** | **4.168** | **1.3888** | Pure PyTorch + Triton (no external packages) |

### 2. CPU: Multi-Domain Benchmark
Unlike BitsAndBytes (which is CUDA-only and fails on CPU), ApexW runs natively on CPU:

| Workload | Optimizer | Step Latency | State Memory | Final Loss | State Memory Saved |
|---|---|:---:|:---:|:---:|:---:|
| **LoRA Fine-Tuning** | PyTorch AdamW (FP32) | 2.59 ms | 1.00 MB | 6.3251 | Baseline (0%) |
| | **ApexW Compact (Default)** | 2.70 ms | 0.50 MB | 6.3251 | **50.0%** |
| | **ApexW 8-bit** | 14.81 ms | 0.25 MB | 6.3801 | **74.6%** |
| | *8-Bit AdamW (BitsAndBytes)* | *Unsupported* | *N/A (CUDA only)* | *N/A* | *Not available on CPU* |
| **Transformer / DiT** | PyTorch AdamW (FP32) | 11.50 ms | 25.99 MB | 6.2879 | Baseline (0%) |
| | **ApexW Compact (Default)** | 14.64 ms | 12.99 MB | 6.2877 | **50.0%** |
| | **ApexW 8-bit** | 68.81 ms | 6.62 MB | 6.2873 | **74.5%** |

*Note: Benchmarks run on local test harnesses; step times can vary depending on hardware, batch size, and background processes.*

---

## Measuring Optimizer State Memory

To inspect the memory your optimizer states consume during a run:

```python
# Run at least one optimizer step so states are allocated
optimizer.print_summary()
```

This prints a breakdown comparing ApexW's state footprint against standard FP32 AdamW:

```text
Optimizer State Memory:
  - ApexW Current State : 12.99 MB
  - PyTorch AdamW Equiv : 25.99 MB
  - Memory Reduction    : 50.0% (13.00 MB saved)
```

---

## Saving and Loading Checkpoints

ApexW follows standard PyTorch checkpointing conventions:

```python
# Save optimizer state
torch.save(optimizer.state_dict(), "optimizer.pt")

# Load optimizer state
optimizer.load_state_dict(torch.load("optimizer.pt", map_location="cpu"))
```

---

## Running Tests

From the repository root:

```bash
pytest tests/
```

## Frequently Asked Questions (FAQ)

#### 1. Do I need to manually install Triton?
**No.** If you have an NVIDIA GPU on Linux, `pip install torch` automatically includes Triton as an internal dependency. On non-Triton platforms (Apple Silicon Metal/MPS, CPU, Windows, or Google TPU), ApexW automatically switches to PyTorch's native C++ multi-tensor engine (`torch._foreach_*`). You never need to install Triton separately.

#### 2. Do I need to use `torch.compile`?
**No.** ApexW runs out of the box in standard PyTorch eager mode with zero startup delay. `torch.compile` is 100% optional.

#### 3. Can ApexW scale to 64, 128, or more GPUs?
**Yes.** Standard ApexW has no rank limit when used with PyTorch FSDP, DeepSpeed, or DDP. When using `ApexWZeRO`, optimizer states are partitioned across all ranks, reducing memory per GPU down to ~0.06 bytes/parameter at 64 GPUs and ~0.03 bytes/parameter at 128 GPUs.

#### 4. Can ApexW train with high LoRA adapter ranks ($r=64, 128+$)?
**Yes.** Whether you train with small ranks ($r=8, 16$) or high ranks ($r=64, 128, 256$), ApexW treats them as standard PyTorch linear weights and delivers the same 50% (Compact) or 75% (8-Bit) state memory savings.

#### 5. Why is `optimizer.py` kept as a single unified file instead of being split into many small files?
During development, we explored refactoring `optimizer.py` into separate small files (splitting out quantization helpers, execution plans, and step engines). However, in PyTorch custom optimizers, doing this introduced severe subtle bugs:
* **Broken Checkpoint Loading:** PyTorch `state_dict()` references class namespaces. Moving parameter proxy classes caused saved checkpoints to fail on reload.
* **Circular Import Errors:** `ApexW`, `FastExecutionPlan`, and `ApexWZeRO` depend on mutual tensor and group definitions.
* **In-Place State Mutation Bugs:** Passing tensor references across multiple abstract helper layers caused subtle view-vs-copy bugs where optimizer state updates were silently dropped.
* **`torch.compile` Tracing Failures:** Extra abstraction layers caused PyTorch Dynamo graph capture to fail and drop into slow Python fallbacks.

To guarantee zero crashes, rock-solid numerical stability, and pass all 160 unit tests across PyTorch versions, `optimizer.py` is intentionally kept as a unified, self-contained engine—similar to how PyTorch officially maintains `torch/optim/adamw.py`.

#### 6. Which lines in `optimizer.py` control the library's default mode?
If you want to permanently change the default mode for your entire project so you don't have to specify flags in every script:
* **Line 1301:** `precision: Union[str, torch.dtype] = "compact"` $\rightarrow$ change `"compact"` to `"8bit"` to make **ApexW 8-Bit** the default everywhere.
* **Line 1311:** `use_muon: bool = False` $\rightarrow$ change to `True` to make **Hybrid Muon** the default everywhere.
You can still explicitly override them per-run in Python anytime (e.g. `ApexW(model, precision="compact")`).other thing are also located there

---

## License

Apache 2.0. See [LICENSE](./LICENSE) for details.
