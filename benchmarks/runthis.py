import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from apexw import ApexW

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

BATCH_SIZE = 16
SEQ_LEN = 512
VOCAB_SIZE = 50304  # Padded to multiple of 128 for GPU efficiency
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 6      # Mini 124M block structure for single-GPU test
NUM_STEPS = 100
TARGET_LOSS = 5.20  # Fast early-convergence target metric

print(f"Running Modded-NanoGPT Speedrun Benchmark on {DEVICE} ({DTYPE})")

# -----------------------------------------------------------------------------
# Modded-NanoGPT Style Transformer Architecture (Rotary + QK-Norm)
# -----------------------------------------------------------------------------
class QKNormCausalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose for PyTorch SDPA: (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)

class ModdedBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_1 = nn.RMSNorm(dim)
        self.attn = QKNormCausalAttention(dim, num_heads)
        self.ln_2 = nn.RMSNorm(dim)
        # Modded-NanoGPT uses ReLU^2 (squared ReLU) for MLP activation
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.ReLU(),
            nn.Linear(4 * dim, dim, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.mlp[0](self.ln_2(x))
        x = x + self.mlp[2](h * h)  # Squared ReLU activation
        return x

class ModdedNanoGPT(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([ModdedBlock(dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying

    def forward(self, idx, targets=None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------------------------------------------------------------
# Data Synthetic Generator (Deterministic Tokens)
# -----------------------------------------------------------------------------
def get_batch():
    torch.manual_seed(42)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    return x, y

# -----------------------------------------------------------------------------
# Rigorous Benchmark Execution Engine
# -----------------------------------------------------------------------------
def run_modded_nanogpt_speedrun(opt_name, opt_factory):
    print(f"\n--- Testing: {opt_name} ---")
    torch.manual_seed(1337)

    # FIXED: Convert model directly to training precision (DTYPE) to avoid cast mismatches!
    model = ModdedNanoGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE).to(dtype=DTYPE)
    optimizer = opt_factory(model.parameters())
    scaler = torch.amp.GradScaler('cuda', enabled=(DTYPE == torch.float16))

    # Pre-generate batches
    x, y = get_batch()

    # 1. Warmup Steps (Excludes JIT compilation overhead from timers)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(DEVICE, dtype=DTYPE):
            _, loss = model(x, y)
        if DTYPE == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()

    # 2. Benchmark Loop with CUDA Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_time = time.perf_counter()
    start_event.record()

    time_to_target_loss = None
    final_loss = 0.0

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(DEVICE, dtype=DTYPE):
            _, loss = model(x, y)

        if DTYPE == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        final_loss = loss_val

        if time_to_target_loss is None and loss_val <= TARGET_LOSS:
            time_to_target_loss = time.perf_counter() - start_time

    end_event.record()
    torch.cuda.synchronize()

    total_cuda_ms = start_event.elapsed_time(end_event)
    total_wall_sec = time.perf_counter() - start_time

    tokens_processed = BATCH_SIZE * SEQ_LEN * NUM_STEPS
    tok_per_sec = tokens_processed / total_wall_sec
    ms_per_step = total_cuda_ms / NUM_STEPS
    peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

    print(f"  Step Time    : {ms_per_step:.2f} ms/step")
    print(f"  Throughput   : {tok_per_sec:,.0f} tokens/sec")
    print(f"  Final Loss   : {final_loss:.4f}")
    print(f"  Peak VRAM    : {peak_mem_mb:.2f} MB")

    return {
        "ms_per_step": ms_per_step,
        "tok_per_sec": tok_per_sec,
        "final_loss": final_loss,
        "peak_mem_mb": peak_mem_mb
    }

# -----------------------------------------------------------------------------
# Configuration Matrix
# -----------------------------------------------------------------------------
optimizers = {
    "PyTorch AdamW (Standard)": lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1),
    "PyTorch AdamW (Fused)": lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1, fused=True),
    "ApexW (Default)": lambda params: ApexW(params, lr=1e-3, weight_decay=0.1, auto_pack=False),
    "ApexW (Optimized Auto-Pack)": lambda params: ApexW(params, lr=1e-3, weight_decay=0.1, auto_pack=True),
}

results = {}
for name, factory in optimizers.items():
    results[name] = run_modded_nanogpt_speedrun(name, factory)

# -----------------------------------------------------------------------------
# Final Markdown Summary Table Generator
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print(f"{'Optimizer':<30} | {'Step Time (ms)':<15} | {'Tokens/Sec':<12} | {'Peak VRAM (MB)':<14}")
print("-" * 80)
for opt_name, res in results.items():
    print(f"{opt_name:<30} | {res['ms_per_step']:<15.2f} | {res['tok_per_sec']:<12,.0f} | {res['peak_mem_mb']:<14.1f}")
print("="*80)
