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
| **`precision`** | `"auto"` | Full FP32 (8 bytes/param) | Tensors $\ge 64\text{K}$ elements or 2D+ matrices receive compact 16-bit states, cutting state memory by 50%. 1D norms remain in FP32. | `precision="full"` |
| **`decay_norm_only`** | `True` | `False` | Automatically excludes 1D tensors (LayerNorm/RMSNorm weights, biases) from weight decay, matching modern LLM best practices. | `decay_norm_only=False` |
| **Dynamic Range Scaling** | `True` | N/A | Monitors variance magnitude and dynamically scales 16-bit second moments to prevent underflow. | `dynamic_scaling=False` |

> [!NOTE]
> If you require exact numerical equivalence to PyTorch AdamW for debugging:
> ```python
> optimizer = ApexW(model.parameters(), lr=1e-3, precision="full", decay_norm_only=False)
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

### A. PyTorch 2.x `torch.compile()` (Recommended)

PyTorch 2.x automatically fuses model operations with zero graph breaks:

```python
compiled_model = torch.compile(model)

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(compiled_model(x), y)
    loss.backward()
    optimizer.step()
```

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
│   ├── zero.py              # ApexWZeRO distributed state sharder (Experimental)
│   ├── cuda_extension.py    # JIT loader for custom CUDA fused kernels (SM 7.0+)
│   └── csrc/                # C++/CUDA source files (packaged in wheels)
│       ├── apexw_cuda.cpp
│       └── apexw_cuda_kernel.cu
├── tests/                   # Comprehensive unit test suite (68 passing, 1 skipped)
│   ├── __init__.py
│   └── test_apexw.py
├── benchmarks/              # Performance, throughput, and scaling benchmarks
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

Expected output: **68 passed, 1 skipped** (the skipped test requires an active CUDA device for `torch.cuda.CUDAGraph` capture).

You can also run tests the old-fashioned way with `unittest`:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).
