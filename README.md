# ApexW

ApexW is an AdamW-style PyTorch optimizer with configurable optimizer-state precision and fused or multi-tensor execution paths. It is designed to reduce optimizer-state memory while remaining usable across different model architectures and training setups.

ApexW works at the parameter level. It does not need to identify a model as a CNN, Transformer, RNN, MLP, or state-space model to update its parameters. Actual speed and memory use depend on tensor shapes, dtypes, device, backend, and the rest of the training workload.

## Why optimizer-state memory matters

Standard AdamW keeps two optimizer-state tensors for each parameter: the first moment (`exp_avg`) and second moment (`exp_avg_sq`). With both stored in FP32, that is approximately **8 bytes per parameter**. For one billion parameters, the two states alone occupy about 8 GB (decimal), before weights, gradients, activations, or temporary buffers are counted.

ApexW can store these states in a compact 16-bit format. When both moments are 16-bit, the standard two-moment state uses about **4 bytes per parameter**. There are important qualifications:

- With `precision="compact"`, ApexW uses compact state storage where the parameter dtype and device support it.
- When `beta2 >= 0.995`, ApexW promotes the second-moment state to FP32 to avoid losing small EMA updates. In that case, the two moments use about 6 bytes per parameter rather than 4.
- `precision="auto"` chooses state precision by parameter, device, and execution context. For example, pure-FP32 CUDA training can use FP32 states for the native fused path, so `auto` does not always reduce state memory.
- Small tensors may remain in FP32 under `auto`. AMSGrad and optional extra optimizer states also change the total.

The figures above describe optimizer states only. They are not a promise to reduce total training VRAM by the same percentage. Check the actual allocation for your model with `optimizer.print_summary()` after the optimizer has initialized its states.

## Installation

```bash
pip install apexw
```

To install a local checkout from the repository root:

```bash
pip install -e .
```

## Quickstart

```python
import torch
from apexw import ApexW

model = MyModel().to("cuda")
optimizer = ApexW(
    model.parameters(),
    lr=1e-3,
    precision="compact",
)

for x, y in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

Use `precision="compact"` when optimizer-state memory is the priority. Keep `precision="auto"` to let ApexW choose state precision for each parameter, or use `precision="full"` when full-precision optimizer states are important. The best choice depends on the model, hardware, and training objective; compact mode is not automatically the fastest option for every workload.

## Using ApexW across model families

ApexW applies parameter updates without model-family-specific rules, so the same optimizer can be used with CNNs, Transformers/ViTs, RNNs/LSTMs/GRUs, MLPs/FFNs, and SSMs such as Mamba or S4. That does not mean every architecture will have the same performance profile:

| Model family | What to keep in mind |
|---|---|
| Transformers and ViTs | Large matrix parameters can make optimizer-state savings useful. Tune the learning rate and warmup for model size, batch size, and sequence length. |
| CNNs | Small networks can be sensitive to per-step launch and Python overhead. Benchmark after warm-up iterations; do not assume kernel speed from one short run. |
| RNNs, LSTMs, and GRUs | Watch gradient stability on long sequences. Gradient clipping can be useful when justified by the training run. |
| MLPs and FFNs | Wide dense layers are handled as ordinary parameter tensors; memory savings depend on parameter count and chosen state precision. |
| SSMs such as Mamba and S4 | ApexW updates their parameters like other tensors. Long-sequence activation and scan memory is separate from optimizer-state memory. |

ApexW does not make the forward or backward pass faster by itself. End-to-end speed depends on the model kernels, data pipeline, batch shape, optimizer implementation, and hardware.

## Recommended training practice: use learning-rate warmup

A short learning-rate warmup is a good starting point for **most training workloads**, not just CNNs. It can be useful for Transformers and ViTs, CNNs, RNNs/LSTMs/GRUs, MLPs, and SSMs such as Mamba. Warmup is part of the learning-rate schedule, not a special ApexW-only feature.

During warmup, the learning rate rises gradually from a lower value to the target learning rate. This can make the first updates less aggressive, particularly when training from scratch, using a relatively high learning rate, or working with large batches. ApexW does not require a fixed warmup length, and a warmup schedule cannot compensate for an unsuitable learning rate or other training problems.

A practical starting point is **10–100 optimizer updates**, or roughly **1–5% of the planned updates** for longer runs. Treat that as a starting range, not a universal rule:

- Use fewer warmup updates for very short experiments; a 100-step warmup can consume too much of a short run.
- Longer or larger-scale training may benefit from a longer warmup.
- Tune the warmup length together with the peak learning rate, batch size, scheduler, and model.
- Apply the schedule per optimizer update. With gradient accumulation, count optimizer updates rather than raw microbatches.

Here is a linear warmup followed by cosine decay. Adjust `warmup_steps`, `start_factor`, and the peak learning rate for your run.

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from apexw import ApexW

optimizer = ApexW(model.parameters(), lr=3e-4, precision="compact")

total_steps = epochs * len(train_loader)  # assumes one optimizer update per batch
warmup_steps = max(1, min(100, round(0.03 * total_steps)))
decay_steps = max(1, total_steps - warmup_steps)

warmup = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_steps,
)
decay = CosineAnnealingLR(optimizer, T_max=decay_steps)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, decay],
    milestones=[warmup_steps],
)

for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

An early loss spike by itself does not establish that the optimizer is over-correcting. Initialization, learning rate, batch order, gradient scale, mixed-precision overflow, and random seed can all affect the first few updates. Check the learning-rate schedule and gradient/loss values, then compare smoothed training and validation metrics over a meaningful portion of the run. Do not select an optimizer from the single lowest training-loss point alone.

## Choosing a state-precision mode

| Mode | Behavior | When to use it |
|---|---|---|
| `"auto"` (default) | Chooses state precision based on parameter size, dtype, device, and execution context. | A general starting point when you want ApexW to choose per parameter. |
| `"compact"` | Uses 16-bit states where supported; the second moment is promoted to FP32 when `beta2 >= 0.995`. | When reducing optimizer-state memory is a priority. |
| `"full"` | Keeps optimizer moments in FP32 for ordinary FP32 parameters. | Numerical comparisons or runs where full-precision states are preferred. |

Example:

```python
optimizer = ApexW(model.parameters(), lr=3e-4, precision="compact")
```

For compact storage, BF16 is used where supported because it has a wider exponent range than FP16. FP16 variance states use dynamic range scaling when enabled; BF16 does not need that scaling. Full-precision updates and accumulation are used where the selected execution path provides them, but compact state storage is still a numerical approximation to full-FP32 optimizer state. Validate convergence on your own workload.

## Defaults and AdamW comparisons

ApexW's defaults are not identical to PyTorch AdamW's defaults:

| Setting | ApexW default | PyTorch `AdamW` default |
|---|---:|---:|
| `lr` | `1e-3` | `1e-3` |
| `betas` | `(0.9, 0.99)` | `(0.9, 0.999)` |
| `eps` | `1e-8` | `1e-8` |
| `weight_decay` | `1e-2` | `1e-2` |
| Weight decay on 1D parameters | Excluded by default | Included unless parameter groups exclude them |

ApexW's default `beta2=0.99` responds more quickly to recent squared gradients than `beta2=0.999`. The rough EMA time scales are about 100 and 1,000 updates respectively; these are characteristic scales, not hard windows. Choose the value for the task rather than assuming one setting is best for every architecture.

To align the main defaults used in a PyTorch AdamW comparison:

```python
optimizer = ApexW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,
    adamw_compat=True,
    precision="full",
)
```

`adamw_compat=True` disables ApexW's default exclusion of 1D parameters from weight decay and changes beta2 to `0.999` when `betas` is `(0.9, 0.99)` or `(0.9, 0.95)`. Use `precision="full"` for full-FP32 moment storage in a closer numerical comparison. This aligns important optimizer settings; it does **not** promise bit-for-bit identical results, since implementation and kernel operation order can still differ.

For a fair benchmark, report both (1) matched hyperparameters and (2) separately tuned configurations. Keep the model, data order, batch size, precision, number of optimizer updates, learning-rate schedule, and evaluation procedure consistent. Compare validation quality and time-to-target as well as training time and memory.

## Optional optimizer features

ApexW also exposes optional update rules and controls. Leave them disabled unless you have a reason to test them, and tune them independently:

- `cautious=True`: applies a directional update mask.
- `grad_centralization=True`: centralizes gradients for eligible weight tensors.
- `clip_grad_norm=...`: enables in-optimizer gradient-norm clipping.
- `beta3` / three-value `betas`: enables the additional slow first-moment path used by AdEMAMix; `alpha` controls its weight.
- `lion=True`: selects sign-based Lion-style updates.
- `amsgrad=True`: enables AMSGrad state and update behavior.
- `nesterov=True`: enables the Nesterov momentum variant.
- `auto_pack=True`: packs eligible small parameter tensors; it is not suitable for every model, especially when parameters in a packed group receive gradients conditionally.
- `fused`, `foreach`, `backend`, `capturable`, and `dynamic_scaling`: control execution and state handling. Defaults are generally the safest starting point.

The constructor also accepts standard optimizer options such as `maximize`. See `ApexW`'s signature and docstring for the complete option list and constraints.

### Muon and HybridApexW

ApexW includes optional Muon optimizers for workloads where orthogonalized matrix updates are worth evaluating. `ApexMuon` is the standalone Muon optimizer. `HybridApexW` routes eligible 2D hidden weight matrices to Muon and keeps vectors, normalization parameters, and excluded embeddings on the AdamW-style ApexW path.

The shortcut below constructs the hybrid optimizer through `ApexW`:

```python
optimizer = ApexW(model, lr=1e-3, use_muon=True)
```

Muon and hybrid configurations are separate optimization choices, not simply faster ApexW settings. Treat them as experimental for your workload and compare convergence, stability, throughput, and memory against an appropriate baseline.

## Mixed precision

For BF16 autocast, a `GradScaler` is usually not needed:

```python
for x, y in train_loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

For FP16 autocast, use a gradient scaler as recommended by PyTorch:

```python
scaler = torch.cuda.amp.GradScaler()

for x, y in train_loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Integrations

### PyTorch learning-rate schedulers

ApexW uses the standard optimizer interface, so it can be paired with PyTorch learning-rate schedulers. Call `scheduler.step()` once per optimizer update, in the order shown in the warmup example above.

### Hugging Face Trainer

Use `use_muon=False` when you specifically want ApexW rather than the hybrid Muon optimizer. When using Hugging Face's built-in scheduler, set `warmup_ratio` or `warmup_steps` in `TrainingArguments` instead of adding a second warmup scheduler:

```python
from apexw.hf import get_apexw_optimizer

optimizer = get_apexw_optimizer(
    model,
    lr=5e-5,
    precision="compact",
    use_muon=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(optimizer, None),
)
trainer.train()
```

### `torch.compile`

ApexW can be used with compiled PyTorch models. Compatibility and performance depend on the PyTorch version, backend, and options in use, so test the exact configuration you plan to run.

## Backend and hardware notes

With `backend="auto"` (the default), ApexW selects an available execution path and can fall back when a specialized kernel is not usable. Depending on the installed PyTorch build and device, paths include:

| Device | Available execution paths |
|---|---|
| NVIDIA CUDA | Triton, custom C++/CUDA, or PyTorch multi-tensor operations |
| AMD ROCm | Triton or PyTorch multi-tensor operations |
| Apple MPS | PyTorch multi-tensor operations |
| Google TPU / XLA | PyTorch/XLA path when `torch_xla` is installed |
| CPU | PyTorch multi-tensor operations |

Availability varies with device, dtype, PyTorch build, and installed dependencies. `backend="auto"` is the recommended starting point; forcing a backend is mainly useful for testing and benchmarking.

## Measuring state memory

```python
# Run at least one optimizer step first so the state tensors have been created.
optimizer.print_summary()
```

This reports ApexW's allocated optimizer-state memory against an FP32 two-moment AdamW estimate. It does not include model parameters, gradients, activations, CUDA workspaces, or other training allocations. For total VRAM, also measure peak device memory during a representative forward/backward/update step.

## Benchmarking ApexW fairly

A first run can be slower than later runs because CUDA initialization, Triton compilation, extension loading, allocator setup, and execution-plan creation may be included in the first timing. For useful comparisons:

1. Run warm-up iterations before starting the timer.
2. Synchronize CUDA immediately before and after the timed region.
3. Repeat the measurement and report a median or distribution, not a single run.
4. Keep batch shape, sequence length, data, precision, and update count fixed.
5. Report final validation metrics, time-to-target, peak VRAM, and optimizer-state memory alongside steps per second.
6. Benchmark both standard AdamW and `AdamW(fused=True)` when the device supports it.

For optimizer research, use multiple seeds when practical. A lower minimum training loss from one run is not enough to establish better convergence or generalization.

## Saving and loading

ApexW follows the PyTorch optimizer checkpoint interface:

```python
torch.save(optimizer.state_dict(), "optimizer.pt")

# After recreating the model and optimizer:
optimizer.load_state_dict(torch.load("optimizer.pt", map_location="cpu"))
```

Load the model checkpoint and recreate the optimizer with compatible parameter groups before loading its state. As with other optimizers, resuming the full training run also requires restoring the model, scheduler, scaler (when used), and relevant random-number-generator state.

## Running the tests

From the repository root:

```bash
python -m pytest -q
```

## License

Apache 2.0. See [LICENSE](./LICENSE).
