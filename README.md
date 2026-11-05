# ApexW

A fast, memory-efficient drop-in replacement for PyTorch `torch.optim.AdamW`.

Train larger models, use bigger batch sizes, and avoid out-of-memory errors on your GPU without sacrificing model quality.

---

## The Hidden Memory Cost of AdamW

When training a deep learning model, GPU VRAM is divided among three primary components:
1. **Model Weights** (the parameters of your neural network)
2. **Gradients** (computed during backpropagation)
3. **Optimizer States** (historical tracking statistics used to compute parameter updates)

While weights and activations receive the most attention, standard `AdamW` is often the largest single memory consumer in your training setup:
- For every parameter in your model, standard `AdamW` tracks **two 32-bit floating-point numbers** ($m$ for momentum, and $v$ for second-moment variance).
- This requires **8 bytes per parameter** in optimizer state buffers alone — twice as much memory as 16-bit model weights!
- For a 1-billion-parameter model (2 GB in FP16/BF16), standard AdamW requires **8 GB of GPU VRAM exclusively for optimizer states**.

```
[ Model Weights: 2 GB ] [ Gradients: 2 GB ] [ Standard AdamW States: 8 GB ]
                                            └── 2x to 4x your model weight memory!
```

### How ApexW Solves This

ApexW stores historical tracking statistics in compact 16-bit precision (`bfloat16` or `float16`) with dynamic scaling to prevent numerical underflow.

Crucially, **all mathematical updates are computed in full 32-bit registers**. Your model trains with the exact same numerical quality and convergence as standard AdamW, but the optimizer uses **half the memory** (4 bytes per parameter instead of 8 bytes).

Best of all: it is a **drop-in replacement**. You don't need to change your training loop, loss calculation, or learning rate schedulers.

> [!NOTE]
> **For Researchers Running Strict Baseline Comparisons:**
> ApexW defaults to `betas=(0.9, 0.99)` because a 100-step variance window provides optimal numerical stability in 16-bit states. If you are publishing academic benchmarks and need **exact hyperparameter parity** with vanilla PyTorch `AdamW` (`beta2=0.999`), simply pass `adamw_compat=True`. This guarantees that differences in benchmark results reflect genuine optimizer logic rather than hyperparameter changes.

---

## 30-Second Quickstart

### Installation

```bash
pip install apexw
```

### Usage

Just swap out `torch.optim.AdamW` for `ApexW`:

```python
import torch
import torch.nn as nn
from apexw import ApexW

model = MyModel().cuda()

# Before: optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# After:
optimizer = ApexW(model.parameters(), lr=1e-3, precision="compact")

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

---

## Why `precision="compact"` Gives the Best Results

In our real-world benchmarks on Transformer models, passing `precision="compact"` delivers the best balance of speed, memory reduction, and training loss.

Here is why:
- **16-bit Storage**: Stores both momentum ($m$) and variance ($v$) in 16-bit memory buffers.
- **Cross-Precision Execution**: Even if your model weights are standard 32-bit floats (`float32`), ApexW's custom GPU kernels load the 16-bit states, promote them to 32-bit inside fast GPU SRAM registers, compute the update cleanly with dynamic underflow protection, and write back 16-bit states in a single memory pass.
- **Result**: You cut optimizer state memory in half while keeping the mathematical precision of full 32-bit arithmetic.

```python
# Recommended setting for maximum memory savings:
optimizer = ApexW(model.parameters(), lr=1e-3, precision="compact")
```

---

## The 4 Modes of ApexW

ApexW comes with 4 straightforward modes depending on your training setup:

### 1. Compact Mode (`precision="compact"`) [Recommended]
Best for everyday training (LLMs, vision models, diffusion networks, and fine-tuning). Cuts optimizer states to 16-bit and protects against underflow using dynamic scaling.

```python
optimizer = ApexW(model.parameters(), lr=1e-3, precision="compact")
```

### 2. Exact AdamW Compatibility Mode (`adamw_compat=True`)
Use this mode if you are publishing a research paper or running controlled benchmark comparisons against vanilla PyTorch `AdamW`. It enforces `beta2 = 0.999`, decays all parameters (including 1D norms), and runs in bit-for-bit hyperparameter parity with PyTorch defaults.

```python
# 100% identical hyperparameter behavior to torch.optim.AdamW
optimizer = ApexW(model.parameters(), lr=1e-3, adamw_compat=True)
```

### 3. Muon Hybrid Mode (`use_muon=True`) [Experimental]
Designed for Transformer and LLM pretraining. Routes 2D weight matrices (attention projections, MLP layers) to 5th-order Newton-Schulz orthogonal momentum updates (Muon), while keeping 1D normalization layers (LayerNorm, RMSNorm) and embeddings on standard AdamW.

```python
# Hybrid Muon pretraining mode
optimizer = ApexW(model.parameters(), lr=1e-3, use_muon=True)
```

### 4. Full Precision Mode (`precision="full"`)
Stores all states in full 32-bit floating point (`float32`), identical to standard PyTorch AdamW, but executes through ApexW's pre-cached execution plan for reduced Python dispatch overhead.

```python
optimizer = ApexW(model.parameters(), lr=1e-3, precision="full")
```

---

## Common Integrations

### Hugging Face `Trainer`

Use the built-in factory function to plug directly into the Hugging Face `Trainer`:

```python
from transformers import Trainer, TrainingArguments
from apexw.hf import get_apexw_optimizer

model = AutoModelForCausalLM.from_pretrained("my-model")

# One line to create the optimizer
optimizer = get_apexw_optimizer(model, lr=5e-5, precision="compact")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    optimizers=(optimizer, None),  # Pass optimizer directly to Trainer
)
trainer.train()
```

### PyTorch LR Schedulers

ApexW works with every standard PyTorch learning rate scheduler without modification:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = ApexW(model.parameters(), lr=1e-3, precision="compact")
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

for epoch in range(epochs):
    # train ...
    optimizer.step()
    scheduler.step()
```

### Automatic Mixed Precision (`torch.cuda.amp.GradScaler`)

ApexW has native AMP scaling support:

```python
scaler = torch.cuda.amp.GradScaler()

for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### PyTorch 2.x `torch.compile()`

ApexW is fully compatible with `torch.compile()`:

```python
model = torch.compile(model)
optimizer = ApexW(model.parameters(), lr=1e-3, precision="compact")
```

---

## Hardware Support

ApexW runs on whatever hardware you have:

| Hardware | How it Runs |
| :--- | :--- |
| **NVIDIA GPUs** (RTX 3090/4090, A100, H100, T4, V100) | OpenAI Triton fused GPU kernels and custom C++/CUDA extensions. |
| **AMD GPUs** (ROCm) | OpenAI Triton ROCm backend and PyTorch multi-tensor operations. |
| **Apple Silicon** (M1 / M2 / M3 / M4 Macs) | Native vectorized Multi-Tensor operations via Apple Metal (MPS). |
| **Google Cloud TPUs** (v2 through v6e) | Native PyTorch/XLA (`torch_xla`) with zero CPU-TPU synchronization stalls. |
| **Standard CPU** | Fast PyTorch multi-tensor C++ foreach operations. |

---

## Key Parameters

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `lr` | `1e-3` | Learning rate. Use the exact same values you use with AdamW. |
| `precision` | `"auto"` | State storage precision: `"compact"` (recommended for memory savings), `"auto"`, or `"full"`. |
| `betas` | `(0.9, 0.99)` | Momentum coefficients. `beta2 = 0.99` provides a stable 100-step smoothing window in 16-bit. |
| `weight_decay` | `1e-2` | Decoupled weight decay factor. |
| `decay_norm_only` | `True` | Protects 1D normalization weights (LayerNorm, RMSNorm) and biases from weight decay. |
| `adamw_compat` | `False` | Sets `betas=(0.9, 0.999)` and `decay_norm_only=False` to match PyTorch AdamW defaults exactly. |
| `use_muon` | `False` | Enables Newton-Schulz orthogonal momentum updates on 2D weight matrices (Experimental). |

---

## Frequently Asked Questions (FAQ)

### Will my model lose accuracy?
No. While optimizer states are stored in 16-bit to save memory, the gradient updates themselves are computed in 32-bit precision inside GPU registers. In extensive testing across Transformer and vision benchmarks, convergence and final loss match standard AdamW.

### Why does ApexW use `beta2 = 0.99` by default instead of `0.999`?
Standard AdamW uses `beta2 = 0.999`, which tracks variance over an exponential 1,000-step window. In 16-bit floating point, the per-step gradient contribution `(1 - beta2) = 0.001` can fall below the precision resolution of 16-bit floats, causing variance buffers to stagnate. Setting `beta2 = 0.99` provides an optimal 100-step window that keeps all updates well within 16-bit precision while converging to identical loss.

**For rigorous academic baseline comparisons:** If you are running controlled benchmarks against vanilla PyTorch `AdamW` and want zero hyperparameter discrepancy, simply pass `adamw_compat=True`:
```python
# Exact 1:1 bit-for-bit parity with vanilla PyTorch AdamW defaults (beta2=0.999)
optimizer = ApexW(model.parameters(), lr=1e-3, adamw_compat=True)
```
This enforces `beta2 = 0.999`, applies weight decay to all parameters (including 1D norms), and automatically maintains variance states in 32-bit so that any benchmark differences reflect genuine optimizer behavior rather than hyperparameter changes.

### Can I save and resume checkpoints?
Yes. Calling `torch.save(optimizer.state_dict(), "checkpoint.pt")` and `optimizer.load_state_dict(checkpoint)` works exactly like PyTorch. Furthermore, you can load an existing PyTorch `AdamW` checkpoint directly into `ApexW` without any conversion script.

### How do I check how much VRAM I saved?
ApexW has a built-in helper:

```python
optimizer.print_summary()
```

This prints a clear breakdown comparing ApexW's actual state memory against native PyTorch AdamW.

---

## Running the Tests

To verify your installation across all backends:

```bash
python3 -m pytest -q
```


---

## License

Apache 2.0 — see [LICENSE](./LICENSE).
