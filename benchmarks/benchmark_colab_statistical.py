import numpy as np
import scipy.stats as stats
import torch
import copy
import gc
import time
from optimizer import ApexW
from lion_pytorch import Lion

# Configure experimental parameters
SEEDS = [42, 101, 2023, 777, 999]
MODEL_DIM = 1024
HIDDEN_DIM = MODEL_DIM * 4
BATCH_SIZE = 64
STEPS = 100
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dynamic data generator
def get_synthetic_data(seed, num_examples=2048):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    x = torch.randn(num_examples, MODEL_DIM, generator=gen)
    w = torch.randn(MODEL_DIM, MODEL_DIM, generator=gen) / (MODEL_DIM ** 0.5)
    y = x @ w
    y = (y - y.mean(dim=-1, keepdim=True)) / (y.var(dim=-1, keepdim=True) + 1e-5).sqrt()
    return x.to(DEVICE), y.to(DEVICE)

# Model definition
class TargetMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(MODEL_DIM, HIDDEN_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(HIDDEN_DIM, MODEL_DIM),
            torch.nn.LayerNorm(MODEL_DIM)
        )
    def forward(self, x):
        return self.net(x)

# Optimizer factories
opt_factories = {
    "ApexW": lambda m: ApexW(m.parameters(), lr=LR, weight_decay=1e-2),
    "AdamW Fused": lambda m: torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-2, fused=True),
    "Lion": lambda m: Lion(m.parameters(), lr=1e-4, weight_decay=1e-2)
}

stats_results = {opt: {"steps_per_sec": [], "val_loss": []} for opt in opt_factories}
criterion = torch.nn.MSELoss()

print(f"[*] Starting Multi-Seed Benchmark across {len(SEEDS)} seeds...")
for seed in SEEDS:
    print(f"\n>>> RUNNING SEED: {seed} <<<")
    train_x, train_y = get_synthetic_data(seed, 2048)
    val_x, val_y = get_synthetic_data(seed + 500, 512)
    
    for name, factory in opt_factories.items():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        model = TargetMLP().to(DEVICE)
        optimizer = factory(model)
        
        # Warmup
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[:64]), train_y[:64])
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # Profile Loop
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[64:128]), train_y[64:128])
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()
            
        stats_results[name]["steps_per_sec"].append(STEPS / elapsed)
        stats_results[name]["val_loss"].append(val_loss)
        
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

# Calculate Statistics
print("\n" + "="*90)
print("STATISTICAL METRICS OVER 5 INDEPENDENT SEEDS (Mean, StdDev, 95% Confidence Interval)")
print("="*90)

for opt, metrics in stats_results.items():
    print(f"\n[Optimizer: {opt}]")
    for metric_name, values in metrics.items():
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        sem = std / np.sqrt(len(arr))
        ci = stats.t.interval(0.95, df=len(arr)-1, loc=mean, scale=sem)
        print(f"  * {metric_name:<15}: Mean = {mean:.4f} | Std = {std:.4f} | 95% CI = ({ci[0]:.4f}, {ci[1]:.4f})")

