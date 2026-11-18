> **Note: comparing ApexW with AdamW?**
>
> ApexW's default settings are a little different from PyTorch's AdamW (mainly `betas` and weight decay on LayerNorm/bias weights). If you run both with their defaults, you're not testing the same thing, and ApexW can look slightly worse in a short run.
>
> To make the test fair, add one flag:
>
> ```python
> optimizer = ApexW(model.parameters(), lr=6e-4, weight_decay=0.1, adamw_compat=True)
> ```
>
> That makes ApexW use the same betas and weight decay as AdamW. Also compare peak GPU memory (`torch.cuda.max_memory_allocated()`), since saving memory is the whole point.
>
> Want to change `betas` yourself? See [How ApexW Defaults Differ from Standard AdamW](#how-apexw-defaults-differ-from-standard-adamw). It also lists the exact line numbers.


---

# ApexW — Train Bigger Models on Less Memory

**ApexW saves you 50% to 90% of your GPU memory — by simply replacing one line of code.**

```python
# Before (standard PyTorch)
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-3)

# After (that's the ONLY change you need to make)
from apexw import ApexW
optimizer = ApexW(model.parameters(), lr=1e-3)
```

Your training loop, your model, your data — **nothing else changes**. You just get a lot more free memory.

### Quick pick

| Mode | One-liner | Memory | Best for |
|------|-----------|--------|---------|
| **Compact** (default) | `ApexW(model.parameters(), lr=1e-3)` | 2.5 B/param | General use, any hardware |
| **8-bit** | `ApexW(model.parameters(), lr=1e-3, precision="8bit")` | 0.754 B/param | Consumer GPUs, LoRA, large models |
| **Hybrid Muon** | `ApexW(model, lr=1e-3, use_muon=True)` | ~4.0 B/param | Transformer pretraining from scratch |

Details for each mode are in [Part 2](#part-2-how-to-use-apexw).

---

## Who Is This For?

This README is written for **everyone.** It's split into three parts:

| Part | Who it's for | What you'll learn |
|------|-------------|------------------|
| **[Part 1: The Basics](#part-1-the-basics)** | Complete beginners | What is VRAM? What is an optimizer? Why does it eat so much memory? |
| **[Part 2: How to Use ApexW](#part-2-how-to-use-apexw)** | Practitioners | Exact code for every real use case |
| **[Part 3: Internals for Researchers](#part-3-internals-for-researchers)** | Researchers / developers | Algorithms, codebase map, how to modify defaults |

**You don't need to read all three.** Find your level and start there.

---

## Table of Contents

- [Install ApexW](#install-apexw)
- [Part 1: The Basics](#part-1-the-basics)
- [Part 2: How to Use ApexW](#part-2-how-to-use-apexw)
  - [Picking your mode](#picking-your-mode-the-60-second-guide)
  - [Mode 1: Compact (Default)](#mode-1-compact-the-default)
  - [Mode 2: 8-Bit](#mode-2-8-bit-maximum-memory-savings)
  - [Mode 3: Hybrid Muon](#mode-3-hybrid-muon-for-transformer-pretraining)
  - [Training Recipes](#training-recipes)
  - [All settings explained](#all-settings-explained)
  - [Checking your memory savings](#checking-your-memory-savings)
  - [How ApexW defaults differ from AdamW](#how-apexw-defaults-differ-from-standard-adamw)
- [Part 3: Internals for Researchers](#part-3-internals-for-researchers)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Install ApexW

Open your terminal and run:

```bash
pip install apexw
```

That's it. Done. ✅

If you want the very latest version directly from source:

```bash
git clone https://github.com/apexw/apexw.git
cd apexw
pip install -e .
```

**You need:**
- Python 3.8 or newer
- PyTorch 2.0 or newer

You do **not** need a C++ compiler, a CUDA toolkit installation, or any extra setup steps.

---

## Part 1: The Basics

<details>
<summary><strong>📖 Click to expand beginner guide (What is VRAM? What is an optimizer?)</strong></summary>

<br>

> This section explains everything from scratch — what VRAM is, what an optimizer does, and why AdamW wastes so much memory. If you already know how optimizers work, skip straight to [Part 2: How to Use ApexW](#part-2-how-to-use-apexw).

---

## What is VRAM and Why Does It Run Out?

Your graphics card (GPU) has its own private memory called **VRAM** (think of it like RAM, but just for the GPU). It's very fast, but it's also limited.

Here's how much VRAM common GPUs have:

| GPU | VRAM |
|-----|------|
| RTX 3060 | 12 GB |
| RTX 4090 | 24 GB |
| A100 (data center) | 80 GB |

When you train an AI model, everything the model needs goes into VRAM — the model itself, the data being processed, and importantly, **extra numbers that the optimizer keeps track of**. When VRAM fills up, PyTorch crashes with an error that looks like this:

```
RuntimeError: CUDA out of memory.
```

If you've seen that error, there's a very good chance the **optimizer** is using way more memory than you think — and that's exactly what ApexW fixes.

---

## What is a Neural Network, Really?

Let's say you want to train an AI to recognize pictures of cats. You feed it thousands of images, and it needs to learn from its mistakes.

Inside a neural network there are millions — sometimes *billions* — of numbers called **weights** or **parameters**. Think of each one as a tiny dial on a huge control panel. When the AI makes a wrong prediction, something needs to go in and carefully adjust those dials to make the AI more accurate next time.

That "something" is called the **optimizer**.

---

## What is an Optimizer and What Does It Store?

The optimizer is the part of training that actually updates all those dials (parameters) after every batch of data. It answers two questions:

**1. Which direction should I adjust the dial?**  
Something called a *gradient* tells us this. Gradients are computed automatically by PyTorch and tell the optimizer: "dial #5,432,100 should go slightly up, dial #5,432,101 should go slightly down..."

**2. How far should I move each dial?**  
Just moving every dial by the same amount every time doesn't work well in practice. Smart optimizers keep notes about what's been happening recently:

- **Momentum** — Like how a rolling ball keeps moving in the same direction even after you stop pushing it. The optimizer remembers which direction each dial has been moving lately, so it doesn't flip-flop back and forth every step. If dial #42 has been consistently going up for the past 100 steps, it'll keep going up confidently instead of second-guessing itself.

- **Variance** — Think of this as a measure of how calm or chaotic each dial's movements have been. If a dial has been jumping all over the place (high variance), the optimizer takes smaller, more careful steps. If a dial has been moving smoothly and predictably (low variance), the optimizer takes bigger, more confident steps.

**The problem:** AdamW keeps both of these numbers (momentum and variance) for **every single parameter** in the model. And it stores them in full 32-bit precision, which means 4 bytes each.

So for every parameter, AdamW needs:
- 4 bytes for the parameter itself (4 bytes in FP32, or 2 bytes in BF16/FP16)
- 4 bytes for momentum
- 4 bytes for variance
- = **12 bytes total per parameter** (or **8 bytes just for the optimizer state**)

---

## Why Does AdamW Eat So Much Memory?

Let's put real numbers on this. AdamW needs 8 bytes per parameter just for optimizer state:

| Model | Parameters | AdamW optimizer state only |
|-------|-----------|---------------------------|
| Small model (110M) | 110 million | ~880 MB |
| Medium LLM (1B) | 1 billion | ~8 GB |
| LLaMA-7B | 7 billion | **~56 GB** |
| LLaMA-70B | 70 billion | **~560 GB** |

For a 7B model, **the optimizer alone needs 56 GB** — before even counting the model weights (another ~14 GB in 16-bit), activations, or gradients. A single RTX 4090 only has 24 GB total. The math just doesn't work.

This is exactly why training large models on consumer hardware with standard AdamW feels impossible.

---

## How Does ApexW Fix This?

ApexW keeps the same information (momentum and variance) but stores it in much smaller formats. It uses three main tricks:

### 🔧 Trick 1: Smaller Number Formats

Standard AdamW stores momentum in 32-bit floats (4 bytes each). ApexW uses 16-bit format instead (2 bytes) — that's already half the size, with barely any accuracy loss.

In 8-bit mode, it goes even further: momentum is stored using just **4 bits** (half a byte!) using a clever format called NF4 that's specifically designed to represent optimizer momentum values without losing important information.

### 🔧 Trick 2: Neighboring Parameters Share Variance

Here's something interesting: parameters that are right next to each other in a weight matrix almost always have *very* similar variance values. If parameter #5000 and parameter #5001 are neighbors in the same layer, their gradient noise looks nearly identical.

ApexW takes advantage of this by storing **one shared variance value for every 4 neighboring parameters**. Instead of 4 variance values, you need 1. That's a 75% reduction in variance storage right there.

### 🔧 Trick 3: Free Memory as You Go (8-bit Mode)

In a normal training step, gradients for the whole model pile up during `loss.backward()` and stay in memory until `optimizer.step()` is called. For large models this causes a big memory spike at the very end of the backward pass.

In 8-bit mode, ApexW updates each layer **as soon as that layer finishes its backward pass**, then immediately frees the gradient. Memory is released little by little as the backward pass runs, instead of everything building up at once.

---

## My First ApexW Training Loop

Here's a complete, working example with every line explained — perfect if you're brand new:

```python
import torch
import torch.nn as nn
from apexw import ApexW  # 1️⃣ Import ApexW instead of AdamW

# Your model — this can be ANY model you're working with.
# Here we build a simple one as an example.
model = nn.Sequential(
    nn.Linear(784, 256),   # Input layer: takes 784 numbers, outputs 256
    nn.ReLU(),              # Activation function (adds non-linearity)
    nn.Linear(256, 10),    # Output layer: 10 classes (e.g. digits 0-9)
).cuda()                   # .cuda() moves the model to your GPU

# 2️⃣ Create the optimizer — this is the ONLY change from AdamW
optimizer = ApexW(model.parameters(), lr=1e-3)
#                 ↑                   ↑
#                 Pass your model     Learning rate: how big each step is
#                 parameters          (1e-3 = 0.001, a common default)

# Loss function: measures how wrong the model's predictions are
criterion = nn.CrossEntropyLoss()

# 3️⃣ Training loop — this is completely standard PyTorch.
#    Nothing here needs to change just because you switched to ApexW.
for epoch in range(10):  # Train for 10 full passes over the dataset
    for inputs, targets in dataloader:
        inputs = inputs.cuda()    # Move input data to GPU
        targets = targets.cuda()  # Move labels to GPU

        optimizer.zero_grad(set_to_none=True)  # Step A: Clear gradients from last step (set_to_none=True clears them quickly)
        outputs = model(inputs)        # Step B: Run the model (forward pass)
        loss = criterion(outputs, targets)  # Step C: Calculate how wrong we were
        loss.backward()                # Step D: Figure out how to fix it (backprop)
        optimizer.step()               # Step E: Update the model weights ← ApexW does its thing here

    print(f"Epoch {epoch+1}/10 — Loss: {loss.item():.4f}")

# 4️⃣ After training, see exactly how much memory you saved
optimizer.print_summary()
```

You'll see output like this:

```
Optimizer State Memory:
  - ApexW Current State : 69.90 MB
  - PyTorch AdamW Equiv : 742.00 MB
  - Memory Reduction    : 90.6% (672.10 MB saved)
```

**That's really it.** The only things that changed from using AdamW were:
1. `from apexw import ApexW` instead of `from torch.optim import AdamW`
2. `ApexW(...)` instead of `AdamW(...)`

</details>

---

## Part 2: How to Use ApexW

---

### Picking Your Mode: The 60-Second Guide

ApexW has three modes. Here's how to choose:

```
Am I pretraining a Transformer (GPT, LLaMA, BERT-style) from scratch?
  ✅ Yes → Use Hybrid Muon mode
  ❌ No  ↓

Is my GPU small (less than 24 GB VRAM), or am I fine-tuning a large model?
  ✅ Yes → Use 8-bit mode
  ❌ No  ↓

Use Compact mode (the default — no extra code needed)
```

| Mode | One-liner | Memory | Best for |
|------|-----------|--------|---------|
| **Compact** (default) | `ApexW(model.parameters(), lr=1e-3)` | 2.5 B/param | General use, any hardware |
| **8-bit** | `ApexW(model.parameters(), lr=1e-3, precision="8bit")` | 0.754 B/param | Consumer GPUs, LoRA, large models |
| **Hybrid Muon** | `ApexW(model, lr=1e-3, use_muon=True)` | ~4.0 B/param | Transformer pretraining from scratch |

---

## Mode 1: Compact (The Default)

```python
# Compact is the default — you don't need to write anything extra
optimizer = ApexW(model.parameters(), lr=1e-3)

# Or be explicit:
optimizer = ApexW(model.parameters(), lr=1e-3, precision="compact")
```

**Memory:** 2.50 bytes per parameter — 68.75% less than standard AdamW.

**Hardware:** Everything — NVIDIA, AMD, Apple Silicon Macs, Google TPUs, CPUs.

**When to use it:** This is the safe, sensible default. Start here. Only switch to a different mode if you have a specific reason (not enough VRAM, or pretraining from scratch).

---

## Mode 2: 8-Bit (Maximum Memory Savings)

Add `precision="8bit"` — that's the only change:

```python
# Before (compact, the default):
optimizer = ApexW(model.parameters(), lr=1e-3)

# After (8-bit — just add one argument):
optimizer = ApexW(model.parameters(), lr=1e-3, precision="8bit")
#                                               ^^^^^^^^^^^^^^^^
```

**Memory:** 0.754 bytes per parameter — 90.6% less than AdamW, 63% less than BitsAndBytes 8-bit.

**Hardware:** NVIDIA GPUs, AMD GPUs, Apple Silicon Macs, CPUs. No precompiled C++ binaries needed.

**When to use 8-bit:**
- You have a consumer GPU with limited VRAM (RTX 3060, 4070, 4090, etc.)
- You're using Google Colab or Kaggle with a free GPU
- You're fine-tuning a large model (7B, 13B, 70B parameters)
- You're doing LoRA or QLoRA fine-tuning
- You keep hitting out-of-memory errors with compact mode

> **Learning rate tip:** For fine-tuning in 8-bit mode, `lr=2e-4` is a solid starting point. For pretraining, `lr=1e-3` is fine.

> **Global default shortcut:** Want 8-bit to be the default across your entire project without passing `precision="8bit"` everywhere? In `apexw/optimizer.py` at **Line 1629**, change `precision: Union[str, torch.dtype] = "compact"` to `"8bit"`.

---

## Mode 3: Hybrid Muon (For Transformer Pretraining)

Add `use_muon=True` and pass the **model itself** (not `model.parameters()`):

```python
# Before (standard AdamW or compact mode):
optimizer = ApexW(model.parameters(), lr=1e-3)

# After (Hybrid Muon):
optimizer = ApexW(model, lr=1e-3, use_muon=True)
#                 ↑                           ↑
#          Pass model directly,        Add this flag
#          NOT model.parameters()
```

### What Hybrid Actually Does: Muon + Your Chosen Precision

When you turn on Hybrid mode, ApexW automatically partitions your model into two specialized engines:
1. **2D Hidden Weight Matrices** (linear layers, attention projections, MLP blocks) $\rightarrow$ updated with **ApexMuon** using Newton-Schulz matrix orthogonalization for cleaner, faster-converging step directions.
2. **1D Norms, Biases & Embeddings** (LayerNorm, RMSNorm, bias vectors, embedding tables) $\rightarrow$ updated with **ApexW**.

**Crucially, Hybrid respects whichever `precision` you choose:**
- **Hybrid Compact (Default):**
  ```python
  optimizer = ApexW(model, lr=1e-3, use_muon=True)
  # Or explicitly:
  optimizer = ApexW(model, lr=1e-3, use_muon=True, precision="compact")
  ```
  Muon handles 2D matrices, and ApexW handles 1D/embeddings in compact 16-bit (~4.0 B/param).
- **Hybrid 8-Bit (Maximum Hybrid Memory Savings):**
  ```python
  optimizer = ApexW(model, lr=1e-3, use_muon=True, precision="8bit")
  ```
  Muon updates 2D matrices, and all 1D/embedding states are quantized to **8-bit**, slashing memory further!
- **Hybrid Full:**
  ```python
  optimizer = ApexW(model, lr=1e-3, use_muon=True, precision="full")
  ```
  Muon for 2D matrices + standard full FP32 states for 1D layers.

> **Why is `compact` the default in Hybrid mode?**  
> In Transformers, **over 98% of parameters are 2D weight matrices** (handled by Muon). Only ~1% are 1D layers (LayerNorm, RMSNorm scales, biases).  
> - Quantizing 1D norms to 8-bit saves almost no memory (a few megabytes), but LayerNorm/RMSNorm scales are sensitive to precision during training.  
> - Therefore, **Hybrid ApexW intentionally defaults to `compact` (16-bit) for 1D layers and embeddings** to ensure rock-solid numerical stability.  
> - It will **only** switch 1D layers and embeddings to 8-bit if you explicitly pass `precision="8bit"` (useful if your model has a massive 128k-token embedding table).

### Where to Set `use_muon=True`

You have three easy ways to enable Hybrid Muon:
1. **In your training script:**
   Pass `use_muon=True` to `ApexW(model, ..., use_muon=True)`, or import `HybridApexW` directly:
   ```python
   from apexw import HybridApexW
   optimizer = HybridApexW(model, lr=1e-3)
   ```
2. **As a global default across your repository:**
   Open `apexw/optimizer.py` at **Line 1639** inside `ApexW.__init__`, and change:
   ```python
   use_muon: bool = False,
   ```
   to:
   ```python
   use_muon: bool = True,
   ```
   Now every `ApexW(model, ...)` call will automatically run in Hybrid Muon mode by default!
3. **In Hugging Face Trainer:**
   Set `optim="hybrid_apexw"` in `TrainingArguments`.

**Why use Hybrid Muon?**  
Memory isn't the primary reason — **faster convergence** is. When pretraining a Transformer from scratch, Muon's orthogonalized update steps often reach your target loss in significantly fewer steps than standard AdamW.

> **Note:** You may see `precision="ultra"` in the codebase. This is an old prototype (3.0 B/param) that has been fully superseded by `8bit` (0.754 B/param). Don't use it.

---

## Training Recipes

Ready-to-use code for the most common situations.

---

### 1. Standard PyTorch Loop

Pass either `model.parameters()` or the model itself. When you pass the model directly, ApexW automatically skips frozen parameters (`requires_grad=False`) — great for fine-tuning.

```python
import torch
import torch.nn as nn
from apexw import ApexW

model = MyModel().cuda()

# Option A — classic style
optimizer = ApexW(model.parameters(), lr=1e-3)

# Option B — pass model directly (skips frozen params automatically)
optimizer = ApexW(model, lr=1e-3)

for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)  # set_to_none=True is faster than zeroing
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

---

### 2. Switching from AdamW in an Existing Project

Here's the before/after for every mode — copy the one you want:

```python
# ─── BEFORE (AdamW) ───────────────────────────────────────────────────────
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)


# ─── AFTER: Compact mode (default, 68% less memory) ──────────────────────
from apexw import ApexW
optimizer = ApexW(model.parameters(), lr=1e-4, weight_decay=1e-2)


# ─── AFTER: 8-bit mode (90% less memory, best for large models / LoRA) ───
from apexw import ApexW
optimizer = ApexW(model.parameters(), lr=1e-4, weight_decay=1e-2, precision="8bit")


# ─── AFTER: Hybrid Muon (faster convergence when pretraining Transformers) ─
from apexw import ApexW
optimizer = ApexW(model, lr=1e-4, weight_decay=1e-2, use_muon=True)
#                 ↑ pass model, not model.parameters(), for Muon mode
```

---

### 3. Hugging Face Trainer and SFTTrainer

Call `register_hf_trainer_optimizer()` once at the top of your script, then use a string in `TrainingArguments` to pick your mode:

```python
from transformers import Trainer, TrainingArguments
from apexw.hf import register_hf_trainer_optimizer

# Call this ONCE at the top of your script
register_hf_trainer_optimizer()

training_args = TrainingArguments(
    output_dir="./checkpoints",
    optim="apexw_8bit",          # ← change this string to switch modes
    #      "apexw"        → compact mode
    #      "apexw_8bit"   → 8-bit mode (most popular for fine-tuning)
    #      "hybrid_apexw" → Hybrid Muon mode
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_ratio=0.03,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

Or use `ApexWTrainer` — a Trainer subclass with Hybrid Muon built in:

```python
from apexw.hf import ApexWTrainer
from transformers import TrainingArguments

trainer = ApexWTrainer(
    model=model,
    args=TrainingArguments(output_dir="./out", learning_rate=5e-5),
    train_dataset=train_ds,
    use_muon=True,   # ← Hybrid Muon on/off
    muon_lr=2e-2,    # ← optional: separate LR for Muon layers
)
trainer.train()
```

---

### 4. LoRA and QLoRA Fine-Tuning

Pass the model directly — ApexW detects active LoRA adapter layers and skips the frozen base weights entirely. The frozen base model weights take up **zero optimizer memory**.

```python
from peft import get_peft_model, LoraConfig
from apexw import ApexW

model = get_peft_model(
    base_model,
    LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj"])
)

# Pass model directly — skips frozen base weights automatically
# 8-bit is recommended for LoRA on consumer GPUs
optimizer = ApexW(model, lr=2e-4, precision="8bit")

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

---

### 5. Mixed Precision (AMP / Autocast)

ApexW works with PyTorch's automatic mixed precision out of the box. No extra setup needed.

**BF16 — for RTX 30xx / 40xx, A100, H100 (recommended, no GradScaler needed):**

```python
import torch
from apexw import ApexW

optimizer = ApexW(model.parameters(), lr=1e-3)

for inputs, targets in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

**FP16 — for older GPUs (V100, T4). Requires GradScaler:**

```python
import torch
from apexw import ApexW

optimizer = ApexW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")  # Prevents FP16 underflow

for inputs, targets in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(inputs), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

You can combine AMP with any ApexW mode — just add `precision="8bit"` or `use_muon=True` as normal.

---

### 6. Gradient Clipping

Gradient clipping prevents training instability when gradients suddenly get very large. ApexW supports in-optimizer clipping so you don't need a separate `clip_grad_norm_` call:

```python
# Add clip_grad_norm — clipping happens automatically inside optimizer.step()
optimizer = ApexW(model.parameters(), lr=1e-3, clip_grad_norm=1.0)

for inputs, targets in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()  # ← clipping happens here
```

Or use PyTorch's standard approach between `backward()` and `step()` — both work fine:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### 7. Gradient Accumulation

When your batch size doesn't fit in memory, accumulate gradients across smaller mini-batches before stepping:

```python
from apexw import ApexW

accumulation_steps = 4  # Equivalent batch size = mini_batch × 4
optimizer = ApexW(model.parameters(), lr=1e-3, precision="8bit")

for step, (inputs, targets) in enumerate(dataloader):
    loss = criterion(model(inputs), targets) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

---

### 8. Parameter Groups (Different Settings per Layer)

Same format as standard PyTorch — pass a list of group dicts:

```python
from apexw import ApexW

optimizer = ApexW(
    [
        # Backbone: tiny LR (preserve pre-trained features)
        {"params": model.backbone.parameters(), "lr": 1e-5, "weight_decay": 1e-2},
        # Head: larger LR (learning the new task)
        {"params": model.head.parameters(), "lr": 1e-3, "weight_decay": 0.0},
    ],
    precision="compact",
)
```

You can also set different precision per group:

```python
optimizer = ApexW(
    [
        {"params": model.big_layers.parameters(), "precision": "8bit"},
        {"params": model.small_layers.parameters(), "precision": "compact"},
    ],
)
```

---

### 9. LLM-Style Parameter Groups

Modern LLM recipes (LLaMA-3, DeepSeek-V3, Qwen-2.5) separate parameters into decayed vs. non-decayed groups. The `build_llm_param_groups` helper is useful when you need this grouping **explicitly** — for example, to pass the groups to a learning rate scheduler that handles them separately, or to override `no_decay_embeddings`:

```python
from apexw import ApexW, build_llm_param_groups

param_groups = build_llm_param_groups(
    model,
    weight_decay=0.1,              # Applied to 2D weight matrices
    no_decay_embeddings=True,      # Also skip decay on embedding tables
)

# Pass groups to both the optimizer and your LR scheduler
optimizer = ApexW(param_groups, lr=3e-4, precision="8bit")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
```

> **Note:** ApexW's `decay_norm_only=True` default already skips 1D norms and biases from weight decay automatically. Use `build_llm_param_groups` when you need explicit groups — e.g. to also exclude embeddings, or to wire them separately into a scheduler.

---

### 10. Fused Backward Pass (Extra Memory Savings for 8-bit)

Normally gradients accumulate during the entire `loss.backward()` call and are only freed after `optimizer.step()`. For large models this causes a big memory spike.

`enable_fused_backward` makes ApexW update each layer *during* the backward pass and free its gradient immediately — flattening that spike:

```python
from apexw import ApexW

optimizer = ApexW(model, lr=1e-3, precision="8bit")

# Enable once after creating the optimizer
optimizer.enable_fused_backward(model=model, grad_accum_steps=1)

# Training loop is unchanged — hooks fire automatically
for inputs, targets in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(inputs), targets)
    loss.backward()   # ← each layer is stepped and freed as backward runs
    optimizer.step()  # ← handles any remaining layers
```

> **Note:** In 8-bit mode on large models (≥ 8 params, ≥ 512K total elements), ApexW enables a version of this automatically. `enable_fused_backward` lets you turn it on explicitly for any mode.

---

### 11. Activation Checkpointing

ApexW saves optimizer memory. **Activation checkpointing** saves a completely different kind of memory — the intermediate values your model stores during the forward pass.

If you're still running out of memory after enabling 8-bit mode, combine both:

```python
from apexw import ApexW, apply_activation_checkpointing

model = MyDeepModel().cuda()

# Discard activations during forward, recompute them during backward
# Saves 60-80% of activation memory at the cost of ~30% slower backward
n_layers = apply_activation_checkpointing(
    model,
    layer_types=(TransformerBlock,),  # ← your repeating layer type here
)
print(f"Checkpointed {n_layers} layers")

optimizer = ApexW(model, lr=1e-3, precision="8bit")
```

---

## All Settings Explained

| Setting | Default | What it does |
|---------|---------|-------------|
| `lr` | `1e-3` | **Learning rate.** How big each update step is. `1e-3` to start; `1e-4` or `2e-4` for fine-tuning. |
| `precision` | `"compact"` | **Mode.** `"compact"` = default; `"8bit"` = max savings; `"full"` = same as standard AdamW. |
| `use_muon` | `False` | **Hybrid Muon.** `True` for Transformer pretraining. Must pass `model`, not `model.parameters()`. |
| `weight_decay` | `1e-2` | **Regularization.** Keeps weights from getting too large. Try `0.1` for LLMs. |
| `betas` | `(0.9, 0.99)` (compact) / `(0.9, 0.999)` (8bit, full) | **Momentum / variance decay.** Controls how fast the optimizer forgets old history. Rarely needs changing. |
| `eps` | `1e-8` | **Stability constant.** Prevents division by zero. Never needs changing. |
| `decay_norm_only` | `True` | **Smart weight decay.** Automatically skips decay for LayerNorm, RMSNorm, and bias layers. Leave on. |
| `adamw_compat` | `False` | **Match AdamW's hyperparameter behavior.** Sets `betas=(0.9, 0.999)` (if you didn't pass your own) and applies weight decay to all parameters, including 1D layers. It does **not** change `precision`, so in `compact`/`8bit` the optimizer state is still stored in a smaller form. For math that matches AdamW as closely as possible, also use `precision="full"`. |
| `clip_grad_norm` | `None` | **Gradient clipping.** e.g. `1.0` — clips inside `optimizer.step()`. |
| `amsgrad` | `False` | AMSGrad variant. Tracks max variance for more conservative steps. Rarely used. |
| `nesterov` | `False` | Nesterov momentum variant. |
| `cautious` | `False` | Zero out updates that disagree with the gradient direction. Can help generalization. |
| `grad_centralization` | `False` | Subtract mean gradient before updating. Can help some architectures. |
| `lion` | `False` | Sign-based Lion updates. Very different behavior — only use if you know what you're doing. |
| `foreach` | `None` (auto) | Multi-tensor engine. Auto-enables on CUDA. Set `False` to disable. |
| `capturable` | `False` | CUDA graph capture compatibility. |
| `maximize` | `False` | Maximize instead of minimize (unusual). |
| `backend` | `None` (auto) | Force a specific backend: `"triton"`, `"cuda"`, `"mps"`, `"foreach"`, `"xla"`. Leave `None`. |
| `stochastic_rounding` | `True` | Keeps quantization errors unbiased over time. Leave on. |
| `block_size` | `None` → 256 | 8-bit block size. Smaller = more precise, slightly more memory. |
| `threshold` | `65536` | Min elements for Quad-V variance sharing. |
| `auto_pack` | `False` | Pack many tiny parameters into one buffer. Helps models with thousands of small tensors. |
| `amp` | `None` (auto) | Mixed precision hint. Auto-detected from model dtype. |

---

## Checking Your Memory Savings

After at least one training step:

```python
optimizer.step()
optimizer.print_summary()
```

Output:
```
Optimizer State Memory:
  - ApexW Current State : 69.90 MB
  - PyTorch AdamW Equiv : 742.00 MB
  - Memory Reduction    : 90.6% (672.10 MB saved)
```

Or print the optimizer object:

```python
print(optimizer)
# ApexW(lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, precision='8bit',
#       params=185,220,864, state_memory=140.42MB, vram_saved=90.6%)
```

---

## How ApexW Defaults Differ from Standard AdamW

ApexW doesn't use exactly the same defaults as `torch.optim.AdamW`. Two things are different:

| Setting | ApexW | PyTorch AdamW | Why |
|---------|:-----:|:-------------:|-----|
| `betas` | `(0.9, 0.99)` in compact mode, `(0.9, 0.999)` in `8bit` and `full` | `(0.9, 0.999)` | With `beta2=0.99` the optimizer adapts to new gradients in about 100 steps instead of about 1,000. In compact mode (16-bit storage) a `beta2` very close to 1 can make the stored variance get stuck, so a lower value is the safer default. |
| Weight decay on 1D weights | **Off** (skipped) | On | LayerNorm, RMSNorm and biases usually shouldn't be decayed. LLaMA, nanoGPT and DeepSeek do the same. |

If you want AdamW's betas and weight decay behavior:

```python
optimizer = ApexW(model.parameters(), lr=1e-3, adamw_compat=True)
# adamw_compat=True switches betas to (0.9, 0.999) and turns weight decay back on for 1D layers
```

### Changing betas

**For one run (easiest):** pass `betas` when you create the optimizer. Whatever you pass is used as-is.

```python
optimizer = ApexW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))
```

**For every run in your project:** edit `apexw/optimizer.py`.

| Line | What's there | What to do |
|------|--------------|-----------|
| **1650** | `betas = (0.9, 0.999) if precision in ("8bit", "full") else (0.9, 0.99)` | Change the two tuples. The first is used for `8bit` and `full`, the second for `compact`. |
| 1615 | `betas: ... = None` in `ApexW.__init__` | Leave it as `None`. That's what lets line 1650 pick the default for you. |
| 252 in `apexw/hybrid.py` | Same rule, used by Hybrid Muon mode | Change it there too if you use `use_muon=True`. |

> **Heads up:** in compact mode, if `beta2` is 0.995 or higher, ApexW keeps the variance in 32-bit to stay accurate. That works fine, but it uses more memory than the 2.5 bytes per parameter number at the top of this README.


---

## Part 3: Internals for Researchers

<details>
<summary><strong>🔬 Click to expand deep dive (quantization math, hardware routing, codebase map)</strong></summary>

<br>

> This section covers algorithms and internals. It assumes familiarity with optimization algorithms and PyTorch.

---

## How the Quantization Actually Works

### Compact Mode (2.50 B/param)

Momentum is stored in **bfloat16** (or float16 on older hardware) — 2 bytes per parameter.

Variance uses **Quad-V sharing**: one variance value is stored per 4 adjacent parameters. Adjacent parameters along the input channel dimension experience correlated gradient noise, so this shared value captures variance accurately while cutting storage by 75%.

Total: 2.00 (momentum) + 0.50 (variance) = **2.50 bytes/param**

### 8-Bit Mode (0.754 B/param)

**Momentum — NF4 4-bit (0.50 B/param):** Momentum values are quantized into 4-bit NormalFloat4, whose non-uniform bin spacing closely matches the expected normal distribution of Adam momentum values. Two values packed per byte. Each 256-element block gets a separate float32 scale factor.

**Variance — Quad logarithmic uint8 (0.25 B/param):** One shared value per 4 parameters, stored in a 256-bin log grid spanning $2^{-24} \approx 6 \times 10^{-8}$ to $1.0$. Log spacing gives ~6.7% relative precision at any scale.

**Block scales (~0.004 B/param):** One float32 scale per 256-element block.

Total: 0.50 + 0.25 + 0.004 ≈ **0.754 bytes/param**

### Fused Nesterov-SNR Lookahead

Both CUDA and Triton kernels compute a Nesterov lookahead in FP32 registers without extra buffer allocation:

$$m_{\text{look}} = \beta_1 \cdot m_t + (1 - \beta_1) \cdot g_t$$

A sign-agreement (SNR) mask zeroes out the update where the lookahead and raw gradient disagree in sign. This sharpens convergence without additional memory.

### EMA-MAX Scale Smoothing

Per-block quantization scales use smoothed EMA:

$$s_t = \max(\text{abs\_max}(m_t),\ s_{t-1} \times 0.9)$$

This prevents quantization resolution collapse after gradient spikes — a common failure mode in naive 8-bit optimizers.

### Stochastic Rounding

Rather than always rounding to the nearest quantization bin (which creates systematic bias), ApexW uses **stochastic rounding**: each value rounds to floor or ceiling with probability proportional to its distance from each. This keeps rounding errors mean-zero across steps.

---

## In-Backward Layer Stepping

**The problem:** All gradients accumulate during `loss.backward()` and stay in memory until `optimizer.step()`. For a 7B model in fp32, that's ~28 GB of gradient memory all at once.

**The solution:** ApexW registers a `post_accumulate_grad_hook` on each parameter. When a layer finishes its backward pass:

1. That layer's weights are updated in-place
2. That gradient is freed: `p.grad = None`
3. Memory is reclaimed before the next layer's gradient is allocated

This activates automatically in 8-bit mode for models with ≥ 8 parameters and ≥ 512K total elements. For other modes, call `optimizer.enable_fused_backward(model=model)` explicitly.

---

## Hardware Routing

ApexW auto-detects hardware and routes to the fastest available backend:

| Hardware | Primary Backend | Fallback |
|----------|----------------|---------|
| NVIDIA CUDA (SM 7.0+) | Triton JIT kernel | Custom C++/CUDA JIT, then `_foreach_*` |
| AMD ROCm | Triton kernel | PyTorch `_foreach_*` |
| Apple Silicon MPS | Vectorized `_foreach_*` | PyTorch `_foreach_*` |
| Google TPU (XLA) | PyTorch/XLA | — |
| CPU | PyTorch `_foreach_*` | — |

BF16 hardware capability is cached per device at startup (one driver API call), so it costs nothing on subsequent steps.

---

## Distributed Training

### DDP and FSDP — No Changes Needed

```python
# DDP
optimizer = ApexW(model.parameters(), lr=1e-3)

# FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model)
optimizer = ApexW(model.parameters(), lr=1e-3)
```

### ZeRO-1 State Sharding — No DeepSpeed Required

```python
from apexw import ApexWZeRO

# Each GPU stores only 1/world_size of optimizer state
optimizer = ApexWZeRO(model.parameters(), lr=1e-3, precision="compact")
```

Combined memory per GPU (compact + ZeRO):

| GPUs | Bytes/param |
|------|------------|
| 1 | 2.50 |
| 4 | ~0.63 |
| 8 | ~0.31 |
| 64 | ~0.04 |
| 128 | ~0.02 |

> **Production-Ready & Built-in:** `ApexWZeRO` is a complete, native ZeRO-1 state sharder built directly into ApexW with zero external dependencies (no DeepSpeed required). It supports PyTorch DDP, `torch.distributed` ProcessGroups, and seamless checkpoint resharding across different GPU cluster sizes via `reshard_zero_checkpoint`.

---

## Codebase Map

### `apexw/optimizer.py` — Main Optimizer

Key locations by symbol name and exact line numbers:

| What | Where to find it |
|------|-----------------|
| **Change default precision** | **Line 1629** (`ApexW.__init__`, parameter `precision=`): change `"compact"` to `"8bit"` to make 8-bit the global default across your whole repository |
| **Change default betas** | `ApexW.__init__`, parameter `betas=` |
| **Change default weight decay** | `ApexW.__init__`, parameter `weight_decay=` |
| **1D norm protection** | `ApexW.__init__`, parameter `decay_norm_only=` |
| **Change Quad-V threshold** | `ApexW.__init__`, parameter `threshold=` |
| **Enable Muon by default** | **Line 1639** (`ApexW.__init__`, parameter `use_muon=`): change `False` to `True` to make Hybrid mode the global default |
| **8-bit runtime acceleration** | `ApexW._enable_8bit_runtime_acceleration()` |
| **In-backward fused stepping** | `ApexW.enable_fused_backward()` |
| **Memory summary** | `ApexW.print_summary()` |
| **Checkpoint save** | `ApexW.state_dict()` |
| **Checkpoint load** | `ApexW.load_state_dict()` |
| **Main step dispatch** | `ApexW.step()` |
| **LLM param group helper** | `build_llm_param_groups()` |

### `apexw/hybrid.py` — HybridApexW

Key parameters in `HybridApexW.__init__`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lr` | `1e-3` | Base LR for 1D/norm/embedding layers |
| `muon_lr` | `None` → inherits `lr` | Separate LR for 2D Muon layers |
| `scale_mode` | `"rms_matched"` | `0.2 * sqrt(max(m,n))` to match AdamW RMS at drop-in LR |
| `precision` | `"compact"` | **Line 233**: Precision for 1D/norm layers — set to `"8bit"` for max savings |
| `min_muon_dim` | `32` | Min matrix dimension to route to Muon |
| `exclude_embeddings` | `True` | Heuristic to skip embedding tables from Muon routing |

### `apexw/quantization.py` — Quantization Core

| Function | What it does |
|----------|-------------|
| `_quantize_blockwise_int8` | Blockwise INT8 with EMA-MAX scaling + stochastic rounding |
| `_dequantize_blockwise_int8` | INT8 → float32 |
| `_get_nf4_m_lut` | NF4 16-bin lookup table |
| `_quantize_blockwise_int4` | NF4 4-bit momentum (2 values per byte) |
| `_dequantize_blockwise_int4` | NF4 4-bit dequantization |
| `_get_v_log_lut` | 24-octave log variance lookup table |
| `_quantize_blockwise_uint8` | Log-UINT8 variance quantization |
| `_copy_fp32_to_16bit_sr_` | In-place FP32 → 16-bit with stochastic rounding |

### `apexw/state.py` — State Management

| Component | What it does |
|-----------|-------------|
| `_normalize_precision` | Parses `"8bit"`, `"compact"`, `"full"`, `"bf16"`, torch dtypes → canonical form |
| `_get_compact_dtype` | Returns `bfloat16` on Ampere+/TPU, `float16` on older CUDA, `float32` on CPU/MPS |
| `StateDictWrapper` | Proxy dict for PyTorch checkpoint compatibility |

### `apexw/zero.py` — ZeRO-1 Sharding

| Component | What it does |
|-----------|-------------|
| `ApexWZeRO` | Native ZeRO Stage-1 — shards `m` and `v` across ranks |
| `reshard_zero_checkpoint` | Convert checkpoint between different world sizes |
| `ZeROStateAdapter` | Mix ZeRO-sharded and non-sharded state dicts |

### `apexw/checkpointing.py` — Activation Checkpointing

| Component | What it does |
|-----------|-------------|
| `CheckpointWrapper` | Wraps a module to recompute activations on backward |
| `apply_activation_checkpointing` | Applies `CheckpointWrapper` to matching layer types automatically |
| `get_memory_breakdown` | Returns VRAM breakdown: weights, optimizer states, activations |

### `apexw/hf.py` — Hugging Face Integration

| Component | What it does |
|-----------|-------------|
| `register_hf_trainer_optimizer` | Patches HF Trainer to recognize `"apexw"`, `"apexw_8bit"`, `"hybrid_apexw"` strings |
| `get_apexw_optimizer` | Constructs the right optimizer from a model and training args |
| `ApexWTrainer` | Trainer subclass with Hybrid Muon built in |

</details>

---

## FAQ

**Q: I already have AdamW in my code. What's the minimum change?**

```python
# Old:
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-3)

# New:
from apexw import ApexW
optimizer = ApexW(model.parameters(), lr=1e-3)
```
That's it. Everything else stays the same.

---

**Q: How do I switch to 8-bit mode?**

Add `precision="8bit"`:
```python
optimizer = ApexW(model.parameters(), lr=1e-3, precision="8bit")
```

---

**Q: How do I use Hybrid Muon, and can I combine it with 8-bit?**

Yes! Hybrid mode routes 2D weight matrices to **ApexMuon** and 1D norms/biases/embeddings to **ApexW**, and it respects whichever precision mode you choose:
```python
# Hybrid with compact 16-bit states (default, ~4.0 B/param):
optimizer = ApexW(model, lr=1e-3, use_muon=True)

# Hybrid with 8-bit quantized states (maximum hybrid memory savings):
optimizer = ApexW(model, lr=1e-3, use_muon=True, precision="8bit")
```
*Note:* Always pass `model` directly (not `model.parameters()`) so ApexW can inspect layer names and shapes for automatic routing.

---

**Q: When does Hybrid ApexW use compact vs. 8-bit? Can I keep 1D layers in compact?**

**Compact is already the default for Hybrid mode!**  
When you call `ApexW(model, lr=1e-3, use_muon=True)`, ApexW routes 2D weight matrices to Muon, and automatically keeps the 1D normalization layers (LayerNorm, RMSNorm) and embeddings in **compact (16-bit) ApexW**.

Even if you changed `apexw/optimizer.py` Line 1629 to default to `"8bit"` for regular training, `HybridApexW` independently defaults to `precision="compact"` (defined in `apexw/hybrid.py` at Line 233). 1D layers will **only** use 8-bit if you explicitly pass `precision="8bit"`.

---

**Q: Where can I enable Hybrid Muon as the global default across my codebase?**

Open `apexw/optimizer.py`, go to **Line 1639** inside `ApexW.__init__`, and change:
```python
use_muon: bool = False,
```
to:
```python
use_muon: bool = True,
```
Now every `ApexW(model, ...)` in your project will run in Hybrid Muon mode automatically.

---

**Q: How do I make 8-bit the global default across a whole codebase?**

Open `apexw/optimizer.py`, go to **Line 1629** inside `ApexW.__init__`, and change:
```python
precision: Union[str, torch.dtype] = "compact",
```
to:
```python
precision: Union[str, torch.dtype] = "8bit",
```
Every `ApexW(...)` call in your project will now use 8-bit by default without needing to pass `precision="8bit"` everywhere.

---

**Q: Will ApexW train as well as AdamW?**

Yes. Quantization errors are small and statistically unbiased thanks to stochastic rounding. ApexW's test suite includes a Rosenbrock convergence test verifying numerical equivalence with PyTorch AdamW in full precision mode.

---

**Q: How is ApexW 8-bit different from BitsAndBytes 8-bit?**

| | ApexW 8-bit | BitsAndBytes 8-bit |
|---|---|---|
| Memory | 0.754 B/param | 2.004 B/param |
| Platforms | CUDA, ROCm, MPS, CPU | CUDA only |
| Peak VRAM | Layer-by-layer freed during backward | Freed only after full backward |
| Install | `pip install apexw` | Needs precompiled binaries |

---

**Q: Do I need Triton or a C++ compiler?**

No. Triton is bundled with standard PyTorch on Linux and auto-detected. The custom CUDA kernel is optional — ApexW falls back to PyTorch's built-in `_foreach_*` multi-tensor operations automatically.

---

**Q: Can I use ApexW with `torch.compile`?**

Yes. The in-backward hooks are aware of CUDA graph capture and disable themselves when needed.

---

**Q: What does `adamw_compat=True` do?**

Makes ApexW use AdamW's hyperparameter behavior: it sets `betas=(0.9, 0.999)` (if you didn't pass your own) and applies weight decay to all parameters, including 1D norms and biases. It does **not** change how the optimizer state is stored. In `compact` and `8bit` mode the state is still compressed, so results will be close to AdamW but not bit-for-bit identical. For the closest match, combine it with `precision="full"`.

---

**Q: How do I run the tests?**

```bash
pip install pytest
pytest tests/
```

---

## Acknowledgements

ApexW was built by its contributors. The techniques it uses — NormalFloat4 quantization, Nesterov lookahead, Newton-Schulz matrix orthogonalization, ZeRO state sharding, stochastic rounding — come from peer-reviewed research across the ML community.

**We used AI assistance** to help improve the ApexW project itself — including reviewing algorithmic choices, strengthening edge cases, and making this documentation clear and accessible to everyone, from beginners to researchers.

---

## License

Apache License 2.0. See [LICENSE](./LICENSE) for details.
