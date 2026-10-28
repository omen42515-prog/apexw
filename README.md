# ApexW

**ApexW** is a memory-efficient, PyTorch-compatible optimizer alternative to `torch.optim.AdamW`. By storing first and second optimizer moments ($m_t$ and $v_t$) in compact 16-bit representations (`bfloat16` or `float16`) with dynamic range scaling, ApexW cuts optimizer state memory by ~50% (4 bytes per parameter instead of 8 bytes) while maintaining numerical precision and training convergence.

ApexW runs on pure PyTorch across CPU, NVIDIA CUDA, Apple MPS, AMD ROCm, and Intel XPU, with an optional JIT-compiled C++/CUDA fused kernel for NVIDIA GPUs (SM 7.0+).

---

## 5-Minute Onboarding

### 1. Installation

Install in your existing PyTorch environment without breaking your CUDA setup:

```bash
cd apexw
pip install -r requirements.txt
pip install -e .
```

### 2. Instant 1-Minute Quickstart

Run the self-contained quickstart script (eager mode, runs in under a second):

```bash
python example.py
```

### 3. Basic Training Loop

ApexW adopts the standard PyTorch optimizer interface:

```python
import torch
import torch.nn as nn
from apexw import ApexW

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10)).to(device)

# Drop-in optimizer replacement
optimizer = ApexW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()

# Inspect memory savings with one line
optimizer.print_summary()
```

### 4. Diagnostics & Verification Runner

ApexW includes a built-in diagnostic runner to verify your environment, report the active backend, test convergence, and display state memory savings:

```bash
# Quick eager verification
python run.py

# Verify with PyTorch 2.x torch.compile()
python run.py --compile
```

Sample output:
```text
======================================================================
 ApexW Verification Runner (v1.2.0)
======================================================================
 Python        : 3.12.3
 PyTorch       : 2.14.1
 Target Device : cuda (NVIDIA A100 / T4)
 Active Backend: triton (auto)
======================================================================
...
 Optimizer State Memory  :
  - PyTorch AdamW States : 3.06 MB  (8 bytes/parameter)
  - ApexW Compact States : 1.54 MB  (4 bytes/parameter)
  - Memory Saved         : 1.52 MB  (49.6% reduction)
======================================================================
```

---

## Behavioral Differences vs PyTorch AdamW

While ApexW follows PyTorch optimizer conventions, its default settings are tuned for modern deep learning and memory efficiency:

| Setting | ApexW Default | PyTorch AdamW Default | Why it matters | How to match standard AdamW |
| :--- | :--- | :--- | :--- | :--- |
| **`betas`** | `(0.9, 0.95)` | `(0.9, 0.999)` | `beta2=0.95` stays safely within the bf16 quantisation window at full 50% VRAM savings. `beta2=0.999` triggers automatic fp32 escalation of `v_t` (25% savings). | `betas=(0.9, 0.999)` |
| **`precision`** | `"auto"` | Full FP32 (8 bytes/param) | Tensors $\ge 64\text{K}$ elements or 2D+ matrices receive compact 16-bit states, cutting state memory by 50%. 1D norms remain in FP32. | `precision="full"` |
| **`decay_norm_only`** | `True` | `False` | Automatically excludes 1D tensors (LayerNorm/RMSNorm weights, biases) from weight decay, matching modern LLM best practices. | `decay_norm_only=False` |
| **Dynamic Range Scaling** | `True` | N/A | Monitors variance magnitude and dynamically scales 16-bit second moments to prevent underflow. | `dynamic_scaling=False` |

> [!IMPORTANT]
> **Beta2 × Compact Precision Compatibility**
>
> When `beta2 >= 0.995` (including the legacy AdamW value of **0.999**) with compact precision, the per-step EMA contribution `(1 - beta2) * g²` falls below the bf16 quantisation step of the accumulated `v_t`. This causes `v_t` to silently stagnate — the second moment freezes and stops tracking gradient statistics, inflating effective learning rate.
>
> **ApexW automatically detects this and escalates `exp_avg_sq` to fp32** while keeping `exp_avg` in compact 16-bit. This costs 2 extra bytes per parameter but remains **25% cheaper than AdamW** (6 vs 8 bytes).
>
> | `beta2` | bf16 EMA margin | Effect | ApexW action |
> | :--- | :--- | :--- | :--- |
> | **0.95** ⬅ *default* | **6.4×** | None | compact bf16 (50% savings) |
> | 0.98 | **2.6×** | Very mild | compact bf16 (50% savings) |
> | 0.99 | **1.3×** | Mild | compact bf16 + warning |
> | 0.995 | **0.6×** | Significant | `v` auto-escalated to **fp32** (25% savings) |
> | 0.999 | **0.1×** | **Severe** | `v` auto-escalated to **fp32** (25% savings) |
>
> **ApexW's default `betas=(0.9, 0.95)` is pre-tuned to avoid these warnings entirely** — you get full 50% VRAM savings out of the box with no silent fallbacks.
> LLM recipes using `beta2=0.95` (e.g., Chinchilla, Llama, Mistral recipes) are naturally compatible. If you must use `beta2=0.999`, automatic fp32 escalation keeps correctness intact.
> To suppress warnings and keep all states in fp32: use `precision="full"`.

> [!NOTE]
> If you require exact numerical equivalence to PyTorch AdamW for debugging:
> ```python
> # Explicitly match AdamW's defaults (beta2=0.999, no norm-only decay)
> optimizer = ApexW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), precision="full", decay_norm_only=False)
> ```


---

## Cross-Hardware Acceleration Architecture

ApexW eliminates proprietary CUDA lock-in by providing specialized, zero-compromise acceleration engines across every major AI accelerator:

| Hardware Architecture | Primary Acceleration Engine | How It Achieves Peak Performance | Dynamic Memory Reduction |
| :--- | :--- | :--- | :---: |
| **NVIDIA GPUs**<br>*(A100, H100, L4, RTX 4090)* | **OpenAI Triton** or **Custom CUDA C++** (`csrc/`) | JIT-compiled into 128-bit vectorized GPU machine code (PTX/SASS); stochastic rounding hash | **~50% cut** (BF16/FP16) |
| **AMD GPUs (ROCm)**<br>*(MI300X, MI250, RX 7900)* | **OpenAI Triton** (`apexw/triton_kernel.py`) | Triton compiles Python kernels down to AMD GCN / CDNA ISA via LLVM HIP backend | **~50% cut** (BF16/FP16) |
| **Apple Silicon**<br>*(M1 / M2 / M3 / M4 Macs)* | **Native Vectorized Multi-Tensor** (`_multi_tensor`) | PyTorch lowers batched `torch._foreach_*` calls directly to Apple Metal Performance Shaders (MPS) | **~50% cut** (Auto BF16/FP16) |
| **Google Cloud TPUs**<br>*(v2, v3, v4, v5e, v5p, v6e)* | **PyTorch XLA** (`apexw/xla_support.py`) | Compiles entire forward + backward + optimizer step globally into fused TPU ASIC machine instructions | **~50% cut** (BF16/FP32) |
| **CPUs & Other Accelerators** | **PyTorch Foreach & Chunking** | Bounded memory chunking (<16 MB) with vectorized C++ ATen multi-tensor execution | **~50% cut** (BF16/FP32) |

---

### 1. OpenAI Triton (NVIDIA CUDA & AMD ROCm)

Triton allows writing high-performance GPU kernels directly in Python that automatically compile down to native machine code on both **NVIDIA** and **AMD ROCm** architectures:

```python
# ApexW automatically detects and dispatches to Triton on CUDA and ROCm
optimizer = ApexW(model.parameters(), lr=1e-3, backend="triton")
```

**Why it matters:**
- **No CUDA compiler (`nvcc`) required**: Triton generates GPU assembly on the fly via LLVM.
- **AMD ROCm Native**: Runs on AMD Instinct and Radeon GPUs with zero kernel rewrites.
- **Mixed-Precision & Dynamic Scaling**: Fuses 16-bit variance scaling, momentum EMA, and weight decay in a single GPU pass.

> [!NOTE]
> When the optional custom C++/CUDA fused kernel (in `csrc/`) is loaded instead of Triton,
> dynamic range `scale_v` factors are **not** passed into the kernel — the kernel applies
> bias correction from the integer `step` value alone. This is sufficient for `bfloat16` and
> `float32` states. For `float16` compact states on machines where only the C++/CUDA extension
> is available (no Triton), set `dynamic_scaling=False` or use `precision="full"` to avoid
> any potential underflow past step ~1000. In practice, if Triton is installed it takes
> priority and handles float16 scaling correctly in-register.

---

### 2. Apple Silicon (MPS Metal Multi-Tensor Fallback)

On macOS, custom C++/CUDA extensions cannot run. ApexW utilizes PyTorch's native vectorized **Multi-Tensor** (`_multi_tensor` / `_foreach_*`) engine, which PyTorch maps directly to **Apple Metal Performance Shaders (MPS)**:

```python
# Automatic on Apple Silicon Mac:
device = torch.device("mps")
model = model.to(device)

optimizer = ApexW(model.parameters(), lr=1e-3)  # Uses native vectorized Metal multi-tensor ops
```

**Why it matters:**
- **Hardware Metal Fusion**: Dispatches batched GPU commands to Apple Silicon unified memory (UAM).
- **Metal Precision Safety**: Automatically detects if the host Mac's Metal version supports `bfloat16`, smoothly selecting `float16` with dynamic range scaling if needed.
- **Bounded Chunking**: Keeps transient memory allocations under 16 MB to prevent unified memory pressure.

---

### 3. Google Cloud TPUs (PyTorch XLA Global Graph Compilation)

On Google TPUs, running custom C++/CUDA kernels or CPU-device synchronizations destroys performance. ApexW implements a dedicated, zero-barrier PyTorch XLA pathway:

```python
import torch_xla.core.xla_model as xm
from apexw import ApexW

device = xm.xla_device()
model = model.to(device)
optimizer = ApexW(model.parameters(), lr=1e-3, backend="xla")

for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
    xm.mark_step()  # XLA fuses forward, backward, and optimizer into single TPU ASIC execution
```

**Why it matters:**
- **Zero Host Synchronization Barriers**: Step counters and bias corrections are computed entirely as on-device tensors (no `.item()` calls).
- **Global ASIC Kernel Fusion**: Google's XLA compiler fuses the entire mathematical graph into TPU Matrix Multiply and Vector processing units.

> [!NOTE]
> The XLA path does **not** apply dynamic range rescaling to 16-bit variance states.
> Injecting the Python-level `scale_v` factor would require a `.item()` host read that creates a synchronization barrier and breaks XLA's HLO graph fusion.
> In practice this is not a problem because the recommended TPU state dtype is `bfloat16`,
> which shares float32's 8-bit exponent and is effectively immune to the underflow
> that `float16` (5-bit exponent) can hit during long training runs.
> If you use the XLA backend with `float16` states, prefer a larger `eps` (e.g., `1e-6`) as a precaution.

---

## Checkpointing & State Persistence

ApexW automatically serializes its dynamic scaling factors alongside optimizer moment buffers in `state_dict()`:

```python
# Save checkpoint
torch.save({"model": model.state_dict(), "opt": optimizer.state_dict()}, "checkpoint.pt")

# Resume training: dynamic scaling factors are fully restored
ckpt = torch.load("checkpoint.pt")
optimizer = ApexW(model.parameters(), lr=1e-3)
optimizer.load_state_dict(ckpt["opt"])
```

---

## Memory Footprint Comparison

| Architecture | Parameter Count | Standard AdamW States | ApexW Compact States | State VRAM Saved |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet-18** | 11.2M | 85.6 MB | **42.8 MB** | 42.8 MB (50%) |
| **BERT-Large** | 340.0M | 2.53 GB | **1.27 GB** | 1.26 GB (50%) |
| **GPT-2 XL** | 1.5B | 11.18 GB | **5.59 GB** | 5.59 GB (50%) |
| **LLaMA-3 (8B)** | 8.0B | 59.60 GB | **29.80 GB** | 29.80 GB (50%) |
| **LLaMA-3 (70B)** | 70.0B | 521.54 GB | **260.77 GB** | 260.77 GB (50%) |

---

## Acceleration Guides

### A. How to Win in Small Variants: Parameter Flattening (`auto_pack=True`) & `torch.compile()`

When a model has many small tensors (e.g. hundreds of small MLP layers, heads, or embeddings), running separate optimizer kernels for each parameter introduces severe sequential launch overhead. 

ApexW solves this natively with **Parameter Flattening (`auto_pack=True`)**, **Adaptive Block Sizing**, and **`torch.compile()`**:

```python
import torch
import torch.nn as nn
from apexw import ApexW

# 1. Compile model to fuse packing, activations, and backward kernels
compiled_model = torch.compile(model)

# 2. Enable native auto_pack to group small parameters into contiguous flat buffers
optimizer = ApexW(compiled_model.parameters(), lr=1e-3, auto_pack=True, pack_threshold=50000)

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(compiled_model(x), y)
    loss.backward()
    optimizer.step()
```

#### Performance Comparison: Small-Scale Variant

| Configuration | Avg Step Time (ms) | Peak VRAM (MB) | Status |
| :--- | :--- | :--- | :--- |
| **PyTorch AdamW Fused** | 16.46 ms | 369.55 MB | Baseline |
| **ApexW (`auto_pack=True`)** | 16.75 ms | 369.46 MB | Almost Matches Fused |
| **ApexW (`auto_pack=True` + `torch.compile`)** | **13.13 ms** ⚡ | **43.24 MB** 📉 | **Absolute Victory 🏆 (20% faster, ~88% less VRAM)** |

#### Why this combination dominates:
1. **Speed Win (20% Faster than AdamW Fused)**: `torch.compile()` automatically fuses the PyTorch execution graph, stripping away Python orchestration overhead and kernel launch latency.
2. **Memory Sweep (~88% VRAM Saved)**: While native uncompiled configurations peaked at ~369 MB, the compiled ApexW configuration reduced peak memory to just **43.24 MB**.
3. **Adaptive Block Size**: ApexW dynamically scales kernel block sizes ($\min(\max(64, \text{pow2}(N)), 1024)$) to optimize warp occupancy on small dimensions.
4. **Zero TorchDynamo Graph Breaks**: Fully integrated without tensor scalar conversions (`.item()`), ensuring unbroken compilation traces.

### B. PyTorch CUDA Graphs (`capturable=True`)

For small models where CPU-to-GPU launch overhead dominates:

```python
optimizer = ApexW(model.parameters(), lr=1e-3, capturable=True)

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()

# Replay with zero CPU launch latency
for x, y in dataloader:
    inputs.copy_(x)
    targets.copy_(y)
    g.replay()
```

### C. Activation Checkpointing for Vision & CNNs (Solving the Activation Bottleneck)

In activation-heavy architectures (deep 2D/3D CNNs, U-Nets, and high-resolution Vision Transformers), forward activations dominate peak VRAM, masking optimizer savings. ApexW integrates native activation checkpointing to slash activation memory by 60–80%:

```python
from apexw import ApexW, apply_activation_checkpointing, get_memory_breakdown

# Checkpoint candidate residual/transformer blocks in one line
num_wrapped = apply_activation_checkpointing(model)

# Inspect exact parameter, optimizer, and activation memory breakdown
diagnosis = get_memory_breakdown(model, optimizer, sample_inputs, sample_targets, criterion)
print(diagnosis["recommendation"])
```

### D. PyTorch Automatic Mixed Precision (AMP) & GradScaler

ApexW provides zero-overhead integration with PyTorch's `torch.amp.GradScaler`:
- **Fused In-Register Unscaling**: The OpenAI Triton kernel unscales gradients directly upon memory load into GPU registers, eliminating auxiliary gradient allocations.
- **In-Place Multi-Tensor Unscaling**: Vectorized foreach unscaling executes strictly in-place with early NaN/Inf detection, eliminating auxiliary peak memory spikes.

```python
scaler = torch.amp.GradScaler("cuda")
optimizer = ApexW(model.parameters(), lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### E. Distributed Training: ApexWZeRO *(Experimental)*

> [!WARNING]
> `ApexWZeRO` is an experimental research implementation of ZeRO Stage-1 state sharding. For production distributed training, using PyTorch FSDP (`torch.distributed.fsdp.FullyShardedDataParallel`) or DeepSpeed ZeRO with ApexW as the base optimizer is strongly recommended.

```python
from apexw import ApexWZeRO

# Experimental ZeRO-1 state sharding
optimizer = ApexWZeRO(model.parameters(), lr=1e-3, world_size=world_size, rank=rank)
```

## 🔮 Complete Optimizer Configuration Matrix

When designing or training neural networks you can end up with highly unconventional architectures. Use this matrix to pick the right ApexW settings without trial and error.

> [!NOTE]
> **New to optimizers? Read this first.**
> An *optimizer* is the part of training that updates your model's weights after each backward pass. ApexW is a drop-in replacement for `torch.optim.AdamW` with memory and speed improvements built in. The three knobs below control how those improvements are applied — the defaults work well for most models, so only reach for these when you hit a specific bottleneck.

---

### The Three Knobs Explained (Plain English)

| Knob | What it does | When to change it |
| :--- | :--- | :--- |
| **`auto_pack`** | Groups your model's many tiny weight tensors into one big contiguous buffer so the GPU runs a *single* kernel instead of hundreds of tiny ones. Think of it like packing 200 small boxes into one shipping pallet — one truck trip vs. 200 trips. | Enable when your model has **lots of small layers** (e.g. a deep MLP, many attention heads). Disable for models with just 1–3 huge layers. |
| **`pack_threshold`** | The maximum size (in number of elements) a parameter can be and still get packed. Parameters larger than this limit are left alone. | Use when your model mixes giant tensors (like an embedding table with millions of entries) alongside many small ones. Set the threshold just below the giant tensor's size so only the small ones get packed. |
| **`torch.compile`** | Asks PyTorch's compiler to trace your model and fuse operations into optimized machine code. Removes Python overhead entirely. | Enable for GPU training whenever your input shapes don't change batch-to-batch. Avoid on CPU — compile overhead outweighs the savings. |

---

### Which Architecture Type Are You?

Answer these questions to find your row in the table below:

- **How many layers does your model have?**
  - Hundreds of small ones → leaning toward #1 (Spindle) or #3 (Chimera)
  - Just 1–3 but each is massive → #2 (Pillar)
  - A handful, but each is very wide → #4 (Shallow-Wide)
- **Do you have a huge embedding table** (vocabulary, positional, etc.)?
  - Yes, plus many small layers too → #3 (Chimera)
  - Yes, and that's basically the whole model → #2 (Pillar)
- **Are you GPU-limited on memory?** → #5 (Edge Device)
- **Are you training on CPU only?** → #6 (CPU Fallback)

---

### Quick-Reference Table

| # | Architecture | `torch.compile` | `auto_pack` | `pack_threshold` |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Mega-Deep Spindle** — 200 + tiny layers, small params | ✅ | ✅ `True` | default |
| 2 | **Monolith Pillar** — 1–2 massive layers (huge embeddings) | ✅ | ❌ `False` | — |
| 3 | **Hybrid Chimera** — giant embeddings + hundreds of tiny blocks | ✅ | ✅ `True` | low (e.g. `50_000`) |
| 4 | **Extreme Shallow-Wide** — few layers, massive parallel paths | optional | ❌ `False` | — |
| 5 | **Memory-Starved Edge Device** — limited VRAM | ✅ | ❌ `False` | — |
| 6 | **CPU Fallback** — no GPU available | ❌ | ❌ `False` | — |

---

### Scenario Details

#### 1 — The Mega-Deep Spindle *(200+ tiny layers)*

**Real-world examples:** Very deep MLPs, recursive models, NeRF-style networks, hyper-networks with hundreds of small submodules, tiny-attention transformers with many thin blocks.

**What's happening without `auto_pack`:** Each small weight tensor gets its own GPU kernel launch. Launching 200 kernels sequentially for 200 tiny layers takes ~4 ms of pure launch overhead before any actual math runs — that's often *more* time than the math itself.

> [!TIP]
> **Maximum Speed (up to 47% faster):** `auto_pack` consolidates those 200 kernel launches into 1. Pair with `torch.compile` to also remove Python loop overhead and you get the fastest possible step time for this architecture type. If you skip `auto_pack` here, you leave speed on the table for free.

```python
model = DeepSpindleModel()   # your 200+ tiny-layer model
optimizer = ApexW(model.parameters(), lr=1e-3, auto_pack=True)
compiled_model = torch.compile(model)

# Training loop (unchanged from standard PyTorch)
for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(compiled_model(x), y)
    loss.backward()
    optimizer.step()
```

---

#### 2 — The Monolith Pillar *(1–2 massive layers)*

**Real-world examples:** A standalone language model head (`nn.Linear(4096, 32000)`), a single giant embedding table, a lone huge projection matrix in a state-space model.

**What's happening if you enable `auto_pack` here:** There's nothing to pack (only 1–2 tensors). `auto_pack` would copy a 500 MB embedding into a temporary buffer, do the exact same math, then copy it back — burning extra VRAM for zero gain.

> [!TIP]
> **Fastest speed + maximum memory savings:** With only one or two large kernels, GPU launch overhead is already negligible — one launch is one launch. Keep `auto_pack=False` and let `torch.compile` fuse the forward + backward + optimizer step into one optimized graph.

```python
model = MonolithModel()   # e.g. a single nn.Linear(4096, 32000) LM head
optimizer = ApexW(model.parameters(), lr=1e-3, auto_pack=False)
compiled_model = torch.compile(model)

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(compiled_model(x), y)
    loss.backward()
    optimizer.step()
```

---

#### 3 — The Hybrid Chimera *(huge embedding + many tiny blocks)*

**Real-world examples:** A transformer with a large vocabulary embedding (`nn.Embedding(50000, 512)`) *plus* 24 small attention/FFN blocks, a multi-modal model with a giant visual encoder and a small text decoder.

**The problem:** If you enable `auto_pack` without a threshold, it tries to pack *everything* — including the 50 M-element embedding table. That wastes VRAM. If you disable it entirely, the 24 tiny blocks lose their speed benefit.

> [!TIP]
> **Optimal balance:** Set `pack_threshold` to a value *smaller* than the embedding's element count but *larger* than any small block's element count. This way `auto_pack` only sees and fuses the small layers, leaving the giant embedding completely untouched.

```python
# How to find a good threshold:
#   print(max(p.numel() for p in model.small_blocks.parameters()))  # e.g. 12 288
#   print(min(p.numel() for p in model.embedding.parameters()))     # e.g. 25 600 000
# → any value between 12 288 and 25 600 000 works; 50 000 is a safe conservative pick

optimizer = ApexW(
    model.parameters(),
    lr=1e-3,
    auto_pack=True,
    pack_threshold=50_000,   # params with numel < 50 K get packed; embedding stays free
)
compiled_model = torch.compile(model)

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(compiled_model(x), y)
    loss.backward()
    optimizer.step()
```

---

#### 4 — The Extreme Shallow-Wide *(few layers, massive parallel paths)*

**Real-world examples:** A 2-layer MLP where each layer is `nn.Linear(8192, 8192)`, a wide ResNet block, a single multi-head attention layer with 128 heads operating in parallel.

**Why packing doesn't help here:** Wide layers are already parallelised across thousands of GPU cores in a single kernel — the GPU is already working at maximum capacity per layer. Packing a 67 M-element tensor would require a large temporary copy and still run the same number of kernels.

> [!NOTE]
> **Low overhead mode:** The bottleneck here is raw compute, not kernel-launch overhead. Focus tuning effort on batch size, mixed precision, and learning-rate schedule rather than optimizer knobs. `torch.compile` is worth trying — measure with and without for your exact shape.

```python
model = ShallowWideModel()   # e.g. nn.Sequential(nn.Linear(8192, 8192), nn.Linear(8192, 8192))
optimizer = ApexW(model.parameters(), lr=1e-3, auto_pack=False)
# Try both — torch.compile helps here when shapes are static
compiled_model = torch.compile(model)

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(compiled_model(x), y)
    loss.backward()
    optimizer.step()
```

---

#### 5 — The Memory-Starved Edge Device *(limited VRAM, e.g. ≤ 8 GB GPU)*

**Real-world examples:** Training on an RTX 3060 (8 GB), a Jetson AGX, a colab free-tier T4 with a large model, or any setup where you're already hitting CUDA out-of-memory errors.

**Why `auto_pack` makes things worse here:** `auto_pack` creates a temporary flat buffer to hold the packed tensors. On a memory-starved device that temporary allocation can be the straw that breaks the camel's back and triggers an OOM. Turning it off removes that overhead entirely.

> [!IMPORTANT]
> **Peak VRAM reduction (~20% additional savings on top of ApexW's baseline 50%):** `torch.compile` with `precision="compact"` is the winning combo — the compiler aggressively reuses intermediate activation buffers, and compact bf16 states cut optimizer memory in half. No packing needed.

```python
model = LightweightModel().to("cuda")

# precision="compact" → optimizer states in bf16 (50% VRAM savings vs AdamW)
# auto_pack=False     → no temporary pack buffer allocation
optimizer = ApexW(model.parameters(), lr=1e-3, precision="compact", auto_pack=False)
compiled_model = torch.compile(model)

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):   # also activations in bf16
        loss = criterion(compiled_model(x), y)
    loss.backward()
    optimizer.step()
```

---

#### 6 — The CPU Fallback *(no GPU available)*

**Real-world examples:** Running on a laptop without discrete GPU, a GitHub Actions CI runner, a Raspberry Pi, or any environment where `torch.cuda.is_available()` returns `False`.

**Why `auto_pack` and `torch.compile` don't help on CPU:**
- `auto_pack` is designed around GPU memory alignment and coalesced kernel launches — concepts that don't map to CPU execution. On CPU it just adds Python overhead with no upside.
- `torch.compile` (Dynamo) on CPU introduces compilation latency upfront and for small models the compiled version is often *slower* than eager due to graph dispatch overhead.

> [!NOTE]
> **Eager mode is optimal on CPU:** PyTorch's foreach path (which ApexW uses automatically on CPU) is already the fastest option — it batches operations into vectorized ATen C++ calls without any overhead. Just use the default settings and let ApexW detect the CPU backend automatically.

```python
model = AnyModel().to("cpu")

# ApexW auto-detects CPU and switches to the foreach path — no config needed
optimizer = ApexW(model.parameters(), lr=1e-3)
# ❌ Do NOT do: torch.compile(model) — adds overhead, rarely helps on CPU

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

---


## Repository Structure

```
apexw/
├── apexw/                   # Core Python package
│   ├── __init__.py          # Package exports: ApexW, ApexWZeRO, Checkpointing, Hardware utils
│   ├── optimizer.py         # ApexW optimizer implementation
│   ├── checkpointing.py     # Activation checkpointing & memory profiling utilities
│   ├── triton_kernel.py     # OpenAI Triton fused kernel backend (NVIDIA CUDA & AMD ROCm)
│   ├── xla_support.py       # Google TPU PyTorch XLA graph compilation utilities
│   ├── flatten.py           # Parameter flattening & grouping for Many-Small models
│   ├── zero.py              # ApexWZeRO distributed state sharder (Experimental)
│   ├── cuda_extension.py    # JIT loader for custom CUDA fused kernels (SM 7.0+)
│   └── csrc/                # C++/CUDA source files (packaged in wheels)
│       ├── apexw_cuda.cpp
│       └── apexw_cuda_kernel.cu
├── tests/                   # Comprehensive unit test suite (93 passing, 1 skipped)
│   ├── __init__.py
│   └── test_apexw.py
├── benchmarks/              # Performance, throughput, and scaling benchmarks
│   ├── benchmark_small_mlp_flattening.py  # 100-layer small MLP parameter flattening benchmark
│   ├── benchmark_flattened_many_small.py  # 2,000 parameter flattening benchmark
│   ├── benchmark.py
│   ├── benchmark_vision_cnn.py    # CNN activation bottleneck & checkpointing benchmark
│   ├── benchmark_compile.py
│   ├── benchmark_mlp.py
│   ├── benchmark_scale.py
│   ├── benchmark_ffn_large.py
│   ├── benchmark_small_cuda_graph.py
│   ├── benchmark_colab_resnet18.py     # Real GPU benchmark run on Google Colab
│   ├── benchmark_colab_statistical.py  # Statistical convergence comparison on Colab
│   ├── results_colab_resnet18.txt      # Recorded Colab benchmark results
│   └── results_colab_statistical.txt
├── example.py               # 1-minute quickstart script
├── MANIFEST.in              # Source distribution manifest
├── LICENSE                  # Apache 2.0
├── pyproject.toml           # Standard PEP 517 build configuration
├── setup.py                 # Setuptools installer
├── requirements.txt         # Core dependencies
├── run.py                   # One-command verification and diagnostics (--compile supported)
└── README.md
```

---

## Testing

Run the full test suite:

```bash
pytest tests/
```

Expected output: **93 passed, 1 skipped** (the skipped test requires an active CUDA device for `torch.cuda.CUDAGraph` capture).

You can also run tests the old-fashioned way with `unittest`:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).
