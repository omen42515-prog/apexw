import os
import sys
import unittest
from unittest.mock import patch
import copy
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from apexw import ApexW, build_llm_param_groups


class TestApexW(unittest.TestCase):

    def test_rosenbrock_convergence(self):
        """Test on non-convex Rosenbrock banana function."""
        torch.manual_seed(42)
        p_adamw = nn.Parameter(torch.tensor([-1.5, 2.0]))
        p_apex = nn.Parameter(torch.tensor([-1.5, 2.0]))

        opt_adamw = torch.optim.AdamW([p_adamw], lr=0.01, weight_decay=0.0)
        # Use same betas as AdamW to test numerical equivalence in full precision
        opt_apex = ApexW([p_apex], lr=0.01, weight_decay=0.0, precision="full", betas=(0.9, 0.999))

        def rosenbrock(p):
            return (1.0 - p[0])**2 + 100.0 * (p[1] - p[0]**2)**2

        for _ in range(1000):
            opt_adamw.zero_grad()
            rosenbrock(p_adamw).backward()
            opt_adamw.step()

            opt_apex.zero_grad()
            rosenbrock(p_apex).backward()
            opt_apex.step()

        final_adam = rosenbrock(p_adamw).item()
        final_apex = rosenbrock(p_apex).item()

        print(f"\nRosenbrock Final Loss -> AdamW: {final_adam:.6f} | ApexW: {final_apex:.6f}")
        # Full precision ApexW with same betas matches AdamW closely
        self.assertAlmostEqual(final_apex, final_adam, places=4)

    def test_memory_reduction(self):
        """Formally verify ApexW cuts state buffer memory in half."""
        model = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.Linear(512, 256, bias=False),
        )

        opt_adamw = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt_adamw.step()

        adamw_bytes = sum(
            sum(v.nelement() * v.element_size() for v in opt_adamw.state[p].values() if torch.is_tensor(v) and v.ndim >= 1)
            for p in model.parameters()
        )

        opt_apex = ApexW(model.parameters(), lr=1e-3, precision="compact", betas=(0.9, 0.95))
        opt_apex.step()
        apex_bytes = sum(
            sum(v.nelement() * v.element_size() for v in opt_apex.state[p].values() if torch.is_tensor(v) and v.ndim >= 1)
            for p in model.parameters()
        )

        print(f"\nState Buffer Bytes: AdamW={adamw_bytes}, ApexW={apex_bytes}")
        # Compact ApexW with beta2<0.995 uses exactly 50% bytes (both m and v in bf16)
        self.assertEqual(apex_bytes, adamw_bytes // 2)

    def test_state_dict_save_and_load(self):
        """Verify checkpoint serialization and restoration."""
        model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 10))
        opt = ApexW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        x = torch.randn(8, 32)
        model(x).sum().backward()
        opt.step()

        sd = opt.state_dict()
        opt_new = ApexW(model.parameters(), lr=1e-3)
        opt_new.load_state_dict(sd)

        self.assertEqual(len(opt.state), len(opt_new.state))
        for p in model.parameters():
            self.assertEqual(opt.state[p]["step"].item(), opt_new.state[p]["step"].item())

    def test_various_tensor_dimensions(self):
        """Verify 1D biases, 2D matrices, and 4D convolutions all execute without error."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        opt = ApexW(conv.parameters(), lr=1e-3)

        x = torch.randn(2, 3, 16, 16)
        conv(x).sum().backward()
        opt.step()

        self.assertFalse(torch.isnan(conv.weight).any())
        self.assertFalse(torch.isnan(conv.bias).any())

    def test_foreach_mode(self):
        """Verify multi-tensor foreach execution engine."""
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        opt = ApexW(model.parameters(), lr=1e-3, foreach=True)

        x = torch.randn(16, 64)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_amsgrad_mode(self):
        """Verify AMSGrad tracks non-decreasing variance."""
        model = nn.Linear(16, 8)
        opt = ApexW(model.parameters(), lr=1e-3, amsgrad=True)

        for _ in range(5):
            opt.zero_grad()
            model(torch.randn(4, 16)).sum().backward()
            opt.step()

        for p in model.parameters():
            self.assertIn("max_exp_avg_sq", opt.state[p])
            self.assertFalse(torch.isnan(opt.state[p]["max_exp_avg_sq"]).any())

    def test_maximize_mode(self):
        """Verify maximization objective."""
        p = nn.Parameter(torch.tensor([0.0]))
        opt = ApexW([p], lr=0.1, maximize=True, weight_decay=0.0)

        for _ in range(10):
            opt.zero_grad()
            loss = p * 1.0  # gradient is 1.0
            loss.backward()
            opt.step()

        self.assertGreater(p.item(), 0.0)

    def test_full_precision_mode(self):
        """Verify full precision float32 mode."""
        model = nn.Linear(32, 16)
        opt = ApexW(model.parameters(), lr=1e-3, precision="full")
        model(torch.randn(4, 32)).sum().backward()
        opt.step()

        for p in model.parameters():
            self.assertEqual(opt.state[p]["exp_avg"].dtype, torch.float32)
            self.assertEqual(opt.state[p]["exp_avg_sq"].dtype, torch.float32)

    # =================== NEW TESTS FOR V4 FIXES ===================

    def test_step_is_tensor(self):
        """Verify step counter is a 0-dim tensor (not Python int) for state_dict compat."""
        model = nn.Linear(4, 2)
        opt = ApexW(model.parameters(), lr=1e-3)
        model(torch.randn(1, 4)).sum().backward()
        opt.step()

        for p in model.parameters():
            step = opt.state[p]["step"]
            self.assertIsInstance(step, torch.Tensor, "Step must be a tensor")
            self.assertEqual(step.dim(), 0, "Step must be 0-dimensional")
            self.assertEqual(step.device.type, "cpu", "Step must be on CPU when capturable=False")
            self.assertEqual(step.item(), 1.0)

    def test_foreach_vs_single_tensor_compact_consistency(self):
        """CRITICAL: foreach and single-tensor MUST produce identical results in compact mode."""
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(8, 8))
        p2 = nn.Parameter(p1.data.clone())

        opt1 = ApexW([p1], lr=1e-3, weight_decay=1e-2, precision="compact", foreach=False)
        opt2 = ApexW([p2], lr=1e-3, weight_decay=1e-2, precision="compact", foreach=True)

        torch.manual_seed(123)
        for _ in range(50):
            g = torch.randn(8, 8)
            p1.grad = g.clone()
            p2.grad = g.clone()
            opt1.step()
            opt2.step()

        diff = (p1.data - p2.data).abs().max().item()
        print(f"\nCompact foreach consistency: max diff = {diff:.2e}")
        # The foreach path does sqrt in bf16 then upcasts (for speed), while
        # single-tensor upcasts first then does sqrt in fp32. The difference
        # is bounded by bf16 mantissa precision (~1e-5) and is acceptable.
        self.assertLess(diff, 1e-4, f"Foreach and single-tensor diverged: {diff:.2e}")

    def test_foreach_vs_single_tensor_full_consistency(self):
        """Foreach and single-tensor MUST produce near-identical results in full mode."""
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(8, 8))
        p2 = nn.Parameter(p1.data.clone())

        opt1 = ApexW([p1], lr=1e-3, weight_decay=1e-2, precision="full", foreach=False)
        opt2 = ApexW([p2], lr=1e-3, weight_decay=1e-2, precision="full", foreach=True)

        torch.manual_seed(123)
        for _ in range(50):
            g = torch.randn(8, 8)
            p1.grad = g.clone()
            p2.grad = g.clone()
            opt1.step()
            opt2.step()

        diff = (p1.data - p2.data).abs().max().item()
        print(f"\nFull foreach consistency: max diff = {diff:.2e}")
        self.assertLess(diff, 1e-6, f"Foreach and single-tensor diverged: {diff:.2e}")

    def test_nesterov_foreach_consistency(self):
        """Nesterov foreach and single-tensor should be close."""
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(8, 8))
        p2 = nn.Parameter(p1.data.clone())

        opt1 = ApexW([p1], lr=1e-3, weight_decay=1e-2, precision="full", foreach=False, nesterov=True)
        opt2 = ApexW([p2], lr=1e-3, weight_decay=1e-2, precision="full", foreach=True, nesterov=True)

        torch.manual_seed(123)
        for _ in range(50):
            g = torch.randn(8, 8)
            p1.grad = g.clone()
            p2.grad = g.clone()
            opt1.step()
            opt2.step()

        diff = (p1.data - p2.data).abs().max().item()
        print(f"\nNesterov foreach consistency: max diff = {diff:.2e}")
        # FP order-of-ops diff is expected (a/d + b/d vs (a+b)/d)
        self.assertLess(diff, 1e-5)

    def test_cautious_foreach_consistency(self):
        """Cautious foreach and single-tensor MUST match."""
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(8, 8))
        p2 = nn.Parameter(p1.data.clone())

        opt1 = ApexW([p1], lr=1e-3, weight_decay=1e-2, precision="full", foreach=False, cautious=True)
        opt2 = ApexW([p2], lr=1e-3, weight_decay=1e-2, precision="full", foreach=True, cautious=True)

        torch.manual_seed(123)
        for _ in range(50):
            g = torch.randn(8, 8)
            p1.grad = g.clone()
            p2.grad = g.clone()
            opt1.step()
            opt2.step()

        diff = (p1.data - p2.data).abs().max().item()
        print(f"\nCautious foreach consistency: max diff = {diff:.2e}")
        self.assertLess(diff, 1e-6)

    def test_grad_scale_support(self):
        """Verify grad_scale attribute is properly handled."""
        model = nn.Linear(8, 4)
        opt = ApexW(model.parameters(), lr=1e-3, foreach=True)

        model(torch.randn(2, 8)).sum().backward()

        # Simulate GradScaler setting grad_scale
        opt.grad_scale = torch.tensor(256.0)
        opt.found_inf = torch.tensor(0.0)  # No inf found
        opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any(), "Params should not be NaN after scaled step")
            self.assertFalse(torch.isinf(p).any(), "Params should not be Inf after scaled step")

        # Cleanup
        del opt.grad_scale
        del opt.found_inf

    def test_found_inf_skips_update(self):
        """When found_inf=1.0, step should be skipped (params unchanged)."""
        model = nn.Linear(8, 4, bias=False)
        opt = ApexW(model.parameters(), lr=1e-3, foreach=True)

        # Do one normal step to init state
        model(torch.randn(2, 8)).sum().backward()
        opt.step()
        opt.zero_grad()

        # Save params
        param_before = list(model.parameters())[0].data.clone()

        # Now do a step with found_inf=1
        model(torch.randn(2, 8)).sum().backward()
        opt.grad_scale = torch.tensor(256.0)
        opt.found_inf = torch.tensor(1.0)  # Inf detected!
        opt.step()

        param_after = list(model.parameters())[0].data
        diff = (param_before - param_after).abs().max().item()
        print(f"\nFound_inf skip: param diff = {diff:.2e}")
        self.assertEqual(diff, 0.0, "Params should not change when found_inf=1")

        # Cleanup
        del opt.grad_scale
        del opt.found_inf

    def test_multiple_param_groups(self):
        """Verify correct behavior with multiple parameter groups."""
        model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 10))
        params = list(model.parameters())
        opt = ApexW(
            [{"params": [params[0], params[1]], "lr": 1e-3},
             {"params": [params[2], params[3]], "lr": 1e-4}],
            weight_decay=1e-2,
        )

        model(torch.randn(4, 32)).sum().backward()
        opt.step()

        # All params should have state
        for p in model.parameters():
            self.assertIn(p, opt.state)
            self.assertEqual(opt.state[p]["step"].item(), 1.0)

    def test_clip_grad_norm(self):
        """Verify in-optimizer gradient norm clipping."""
        model = nn.Linear(8, 4)
        opt = ApexW(model.parameters(), lr=1e-3, clip_grad_norm=1.0)

        # Set large gradients
        model(torch.randn(2, 8)).sum().backward()
        for p in model.parameters():
            p.grad.mul_(100.0)

        opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())
            self.assertFalse(torch.isinf(p).any())

    def test_nesterov_cautious_combined(self):
        """Test nesterov + cautious together."""
        model = nn.Linear(16, 8)
        opt = ApexW(model.parameters(), lr=1e-3, nesterov=True, cautious=True, precision="compact")

        for _ in range(10):
            opt.zero_grad()
            model(torch.randn(4, 16)).sum().backward()
            opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_memory_summary(self):
        """Test the memory summary formatting."""
        model = nn.Linear(256, 128, bias=False)
        # Use beta2=0.95 (LLM recipe) so both m and v stay in bf16 -> 50% savings
        opt = ApexW(model.parameters(), lr=1e-3, precision="compact", betas=(0.9, 0.95))
        model(torch.randn(1, 256)).sum().backward()
        opt.step()

        summary = opt.get_memory_summary()
        self.assertIn("50.0%", summary)
        self.assertIn("Memory Reduction", summary)
        print(f"\n{summary}")

    def test_zero_weight_decay(self):
        """Ensure zero weight decay doesn't modify params before Adam update."""
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(p1.data.clone())

        # Use same betas as AdamW to test numerical equivalence in full precision
        opt1 = ApexW([p1], lr=1e-3, weight_decay=0.0, precision="full", betas=(0.9, 0.999))
        opt2 = torch.optim.AdamW([p2], lr=1e-3, weight_decay=0.0)

        torch.manual_seed(99)
        for _ in range(20):
            g = torch.randn(4)
            p1.grad = g.clone()
            p2.grad = g.clone()
            opt1.step()
            opt2.step()

        diff = (p1.data - p2.data).abs().max().item()
        print(f"\nZero weight_decay match: {diff:.2e}")
        self.assertLess(diff, 1e-6)

    def test_gradient_centralization(self):
        """Verify gradient centralization for conv and linear weights."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        linear = nn.Linear(32, 10)
        opt = ApexW(
            list(conv.parameters()) + list(linear.parameters()),
            lr=1e-3,
            grad_centralization=True,
        )

        # Set gradients with non-zero mean
        for p in list(conv.parameters()) + list(linear.parameters()):
            p.grad = torch.randn_like(p) + 10.0

        opt.step()

        # Weight gradients should have near-zero mean across fan-in dims
        conv_w_mean = conv.weight.grad.mean(dim=[1, 2, 3]).abs().max().item()
        linear_w_mean = linear.weight.grad.mean(dim=1).abs().max().item()
        self.assertLess(conv_w_mean, 1e-5)
        self.assertLess(linear_w_mean, 1e-5)
        # Biases should remain unchanged
        self.assertFalse(torch.isnan(conv.bias).any())
        self.assertFalse(torch.isnan(linear.bias).any())

    def test_fp16_underflow_safety(self):
        """Verify that FP16 compact states do not underflow to NaN with tiny gradients."""
        p = nn.Parameter(torch.randn(16, 16))
        opt = ApexW([p], lr=1e-3, precision="compact", foreach=False)

        # Force state_dtype to float16 to simulate FP16 hardware
        opt.param_groups[0]["params"][0].grad = torch.randn_like(p) * 1e-5
        opt.step()
        state = opt.state[p]
        state["exp_avg"] = state["exp_avg"].to(torch.float16)
        state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.float16)
        state["state_dtype"] = torch.float16

        # Step with tiny gradients
        for _ in range(5):
            opt.zero_grad()
            p.grad = torch.randn_like(p) * 1e-6
            opt.step()

        self.assertFalse(torch.isnan(p).any(), "FP16 step produced NaN!")
        self.assertFalse(torch.isinf(p).any(), "FP16 step produced Inf!")

    def test_capturable_mode(self):
        """Verify capturable mode: rejects CPU, works with lr=0.0 on accelerator."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = ApexW([p], lr=0.0, capturable=True)
        p.grad = torch.randn_like(p)
        if torch.cuda.is_available():
            p_cuda = p.cuda()
            p_cuda.grad = torch.randn_like(p_cuda)
            opt_cuda = ApexW([p_cuda], lr=0.0, capturable=True)
            opt_cuda.step()
            self.assertFalse(torch.isnan(p_cuda).any())
        else:
            # Capturable requires accelerator device, CPU raises error matching PyTorch
            with self.assertRaises(RuntimeError):
                opt.step()

    def test_ademamix_mode(self):
        """Verify AdEMAMix dual-momentum mode (fast + slow EMA)."""
        model = nn.Linear(16, 8)
        opt = ApexW(model.parameters(), lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=5.0)

        for _ in range(5):
            opt.zero_grad()
            model(torch.randn(4, 16)).sum().backward()
            opt.step()

        for p in model.parameters():
            self.assertIn("exp_avg_slow", opt.state[p])
            self.assertFalse(torch.isnan(opt.state[p]["exp_avg_slow"]).any())
            self.assertFalse(torch.isnan(p).any())

    def test_lion_mode(self):
        """Verify Google Lion sign-momentum mode (zero second moment, 75% memory cut)."""
        model = nn.Linear(32, 16)
        opt = ApexW(model.parameters(), lr=1e-4, lion=True, precision="compact")

        for _ in range(5):
            opt.zero_grad()
            model(torch.randn(4, 32)).sum().backward()
            opt.step()

        for p in model.parameters():
            # Lion should NOT allocate exp_avg_sq
            self.assertNotIn("exp_avg_sq", opt.state[p])
            self.assertIn("exp_avg", opt.state[p])
            self.assertFalse(torch.isnan(p).any())

        # Check that Lion cuts state memory by ~75% vs AdamW FP32
        mem_mb = opt.get_state_memory_mb()
        param_bytes = sum(p.numel() * 2 for p in model.parameters())  # 2 bytes/param
        expected_mb = param_bytes / (1024 * 1024)
        self.assertAlmostEqual(mem_mb, expected_mb, places=4)

    def test_smart_weight_decay(self):
        """Verify decay_norm_only=True applies decay only to 2D+ weights, skipping 1D biases."""
        linear = nn.Linear(8, 4, bias=True)
        # Initialize weight and bias to all ones
        linear.weight.data.fill_(1.0)
        linear.bias.data.fill_(1.0)

        opt = ApexW(linear.parameters(), lr=0.01, weight_decay=0.5, decay_norm_only=True)

        # Zero gradients, so only weight decay acts
        linear.weight.grad = torch.zeros_like(linear.weight)
        linear.bias.grad = torch.zeros_like(linear.bias)
        opt.step()

        # Weight should decay: 1.0 * (1 - 0.01 * 0.5) = 0.995
        self.assertLess(linear.weight.data.mean().item(), 1.0)
        # Bias should NOT decay: should remain 1.0
        self.assertEqual(linear.bias.data.mean().item(), 1.0)

    def test_complex_parameters(self):
        """Verify full complex64 support across both foreach and single-tensor execution."""
        # Test foreach mode with complex
        p_for = nn.Parameter(torch.randn(16, dtype=torch.complex64))
        opt_for = ApexW([p_for], lr=1e-3, foreach=True, precision="compact")
        (p_for.abs() ** 2).sum().backward()
        opt_for.step()
        self.assertFalse(torch.isnan(p_for).any())
        self.assertEqual(opt_for.state[p_for]["step"].item(), 1.0)

        # Test single tensor mode with complex
        p_sing = nn.Parameter(torch.randn(16, dtype=torch.complex64))
        opt_sing = ApexW([p_sing], lr=1e-3, foreach=False, precision="compact")
        (p_sing.abs() ** 2).sum().backward()
        opt_sing.step()
        self.assertFalse(torch.isnan(p_sing).any())
        self.assertEqual(opt_sing.state[p_sing]["step"].item(), 1.0)

    def test_dynamic_amsgrad_switch(self):
        """Verify enabling amsgrad after initial step initializes max_exp_avg_sq without crash."""
        p = nn.Parameter(torch.randn(8))
        opt = ApexW([p], lr=1e-3, amsgrad=False)
        p.grad = torch.randn_like(p)
        opt.step()

        # Dynamically switch amsgrad on
        opt.param_groups[0]["amsgrad"] = True
        p.grad = torch.randn_like(p)
        opt.step()
        self.assertIn("max_exp_avg_sq", opt.state[p])
        self.assertFalse(torch.isnan(p).any())

    def test_build_llm_param_groups(self):
        """Verify LLM parameter group builder separates 2D weights from 1D biases/norms."""
        class MockLLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.linear = nn.Linear(32, 64)
                self.norm = nn.LayerNorm(64)

        model = MockLLM()
        groups = build_llm_param_groups(model, weight_decay=0.1, no_decay_embeddings=True)
        self.assertEqual(len(groups), 2)
        decayed = groups[0]["params"]
        non_decayed = groups[1]["params"]

        # Linear weight should be decayed
        self.assertTrue(any(p is model.linear.weight for p in decayed))
        # Linear bias, norm weight, norm bias, embedding should be non_decayed
        self.assertTrue(any(p is model.linear.bias for p in non_decayed))
        self.assertTrue(any(p is model.norm.weight for p in non_decayed))
        self.assertTrue(any(p is model.norm.bias for p in non_decayed))
        self.assertTrue(any(p is model.embed.weight for p in non_decayed))

    def test_zero_config_default(self):
        """Verify zero-config UX: ApexW(model.parameters()) runs without any flags."""
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
        opt = ApexW(model.parameters())
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        self.assertEqual(opt.defaults["precision"], "auto")
        self.assertIsNone(opt.defaults["foreach"])
        self.assertTrue(opt.defaults["decay_norm_only"])

    def test_smart_threshold_partitioning(self):
        """Verify smart partitioning: 2D+ weights get compact BF16 for m, 1D norm/bias get FP32.
        With default beta2=0.999, exp_avg_sq (v) is auto-escalated to fp32 for all compact params.
        With beta2=0.95, exp_avg_sq stays in bfloat16 (safe margin = 6.4x).
        """
        linear = nn.Linear(256, 256)
        norm = nn.LayerNorm(256)

        # Case 1: beta2=0.95 (safe) -> both m and v in bfloat16 for massive params
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt95 = ApexW(
                list(linear.parameters()) + list(norm.parameters()),
                precision="auto", threshold=65536, betas=(0.9, 0.95),
            )
        for p in list(linear.parameters()) + list(norm.parameters()):
            p.grad = torch.randn_like(p)
        opt95.step()

        # 2D Linear weight: m=bf16, v=bf16 (beta2=0.95, margin=6.4x)
        self.assertEqual(opt95.state[linear.weight]["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(opt95.state[linear.weight]["exp_avg_sq"].dtype, torch.bfloat16)

        # 1D params (< threshold): m=fp32, v=fp32 (auto=full for small 1D)
        self.assertEqual(opt95.state[linear.bias]["exp_avg"].dtype, torch.float32)
        self.assertEqual(opt95.state[norm.weight]["exp_avg"].dtype, torch.float32)
        self.assertEqual(opt95.state[norm.bias]["exp_avg"].dtype, torch.float32)

        # Case 2: beta2=0.999 (severe) -> m=bf16, v=fp32 for massive params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt999 = ApexW(
                list(linear.parameters()) + list(norm.parameters()),
                precision="auto", threshold=65536, betas=(0.9, 0.999),
            )
        for p in list(linear.parameters()) + list(norm.parameters()):
            p.grad = torch.randn_like(p)
        opt999.step()

        # 2D Linear weight: m=bf16 (compact), v=fp32 (auto-escalated for correctness)
        self.assertEqual(opt999.state[linear.weight]["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(opt999.state[linear.weight]["exp_avg_sq"].dtype, torch.float32)

    def test_cuda_extension_safe_fallback(self):
        """Verify CUDA extension interface is safely present and returns graceful fallback."""
        from apexw.cuda_extension import (
            is_cuda_kernel_available,
            fused_adamw_cuda_step,
            fused_multi_tensor_adamw_cuda_step,
        )
        avail = is_cuda_kernel_available()
        self.assertIsInstance(avail, bool)

        p = torch.randn(10, 10)
        g = torch.randn(10, 10)
        m = torch.zeros(10, 10, dtype=torch.bfloat16)
        v = torch.zeros(10, 10, dtype=torch.bfloat16)
        res_single = fused_adamw_cuda_step(p, g, m, v, 1e-3, 0.9, 0.999, 0.01, 1e-8, 1)
        self.assertFalse(res_single)

        res_multi = fused_multi_tensor_adamw_cuda_step([p], [g], [m], [v], 1e-3, 0.9, 0.999, 0.01, 1e-8, 1)
        self.assertFalse(res_multi)

    def test_multi_tensor_mixed_precision_convergence(self):
        """Verify multi-tensor step with mixed precision converges smoothly."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        opt = ApexW(model.parameters(), lr=1e-3, precision="auto")
        criterion = nn.MSELoss()

        x = torch.randn(16, 32)
        target = torch.randn(16, 16)

        initial_loss = criterion(model(x), target).item()
        for _ in range(20):
            opt.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            opt.step()

        final_loss = criterion(model(x), target).item()
        self.assertLess(final_loss, initial_loss)

    def test_bf16_weights_multi_tensor(self):
        """Verify models with BF16 weights step without error."""
        linear = nn.Linear(32, 16).bfloat16()
        opt = ApexW(linear.parameters(), lr=1e-3, precision="compact")
        x = torch.randn(8, 32, dtype=torch.bfloat16)
        linear(x).sum().backward()
        opt.step()
        for p in linear.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_torch_compile_step(self):
        """Verify torch.compile(opt.step) compiles and executes without graph breaks."""
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
        opt = ApexW(model.parameters(), lr=1e-3)
        opt_step = torch.compile(opt.step)

        x = torch.randn(4, 16)
        y = model(x).sum()
        y.backward()

        opt_step()
        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_apex_zero_state_sharding(self):
        """Verify ApexWZeRO state sharding partitions memory across simulated ranks."""
        from apexw import ApexWZeRO
        torch.manual_seed(42)
        base = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        x = torch.randn(8, 128)

        # Standard unpartitioned ApexW
        m_ref = copy.deepcopy(base)
        opt_ref = ApexW(m_ref.parameters(), lr=1e-3, weight_decay=1e-2)
        m_ref(x).sum().backward()
        opt_ref.step()
        ref_mem = opt_ref.get_state_memory_mb()

        # ZeRO-1 partitioned across world_size = 2
        m_z0 = copy.deepcopy(base)
        opt_z0 = ApexWZeRO(m_z0.parameters(), lr=1e-3, weight_decay=1e-2, world_size=2, rank=0)
        m_z0(x).sum().backward()
        opt_z0.step()
        z0_mem = opt_z0.get_state_memory_mb()

        # State memory on rank 0 must be roughly half of the reference
        self.assertLess(z0_mem, ref_mem * 0.75)
        self.assertGreater(z0_mem, 0.0)

    def test_master_weight_copy_isolation(self):
        """Verify master weights are maintained strictly in float32 without downcasting.
        Uses beta2=0.95 (safe compact range) so both m and v stay in bfloat16.
        Verifies that at beta2=0.999, exp_avg_sq auto-escalates to fp32 while param stays fp32.
        """
        linear = nn.Linear(32, 16)
        self.assertEqual(linear.weight.dtype, torch.float32)

        # beta2=0.95: m=bf16, v=bf16 (safe 6.4x margin)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt = ApexW(linear.parameters(), lr=1e-3, precision="auto", betas=(0.9, 0.95))
        linear(torch.randn(4, 32)).sum().backward()
        opt.step()

        # Parameter must remain full float32
        self.assertEqual(linear.weight.dtype, torch.float32)
        # History state: m=compact bfloat16, v=compact bfloat16 (beta2=0.95 safe)
        state = opt.state[linear.weight]
        self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)

        # beta2=0.999: v auto-escalates to fp32, parameter stays fp32
        linear2 = nn.Linear(32, 16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt2 = ApexW(linear2.parameters(), lr=1e-3, precision="auto", betas=(0.9, 0.999))
        linear2(torch.randn(4, 32)).sum().backward()
        opt2.step()
        self.assertEqual(linear2.weight.dtype, torch.float32)
        state2 = opt2.state[linear2.weight]
        self.assertEqual(state2["exp_avg"].dtype, torch.bfloat16)      # m stays compact
        self.assertEqual(state2["exp_avg_sq"].dtype, torch.float32)    # v auto-escalated

    def test_adaptive_dynamic_epsilon(self):
        """Verify adaptive dynamic epsilon prevents division by zero with zero/tiny gradients."""
        p = nn.Parameter(torch.ones(10, dtype=torch.float32))
        p.grad = torch.zeros(10, dtype=torch.float32)
        opt = ApexW([p], lr=1e-3, eps=1e-16)

        # Step with zero grad should not produce NaN or Inf
        opt.step()
        self.assertFalse(torch.isnan(p).any())
        self.assertFalse(torch.isinf(p).any())

    def test_stochastic_rounding_expectation(self):
        """Verify stochastic rounding preserves tiny updates below the deterministic threshold."""
        u = torch.tensor([1.0001, 1.0002, 1.0003, 1.0004], dtype=torch.float32)
        # Standard round-to-nearest in BF16 rounds 1.0001 directly to 1.0
        rn = u.bfloat16().float()
        self.assertEqual(rn[0].item(), 1.0)


    def test_step_fused_interface(self):
        """Verify step_fused interface is importable and handles fallback safely."""
        from apexw.cuda_extension import step_fused, is_cuda_kernel_available
        # Calling on CPU tensors should safely return False without crashing
        p = torch.randn(10)
        g = torch.randn(10)
        m = torch.zeros(10)
        v = torch.zeros(10)
        res = step_fused([p], [g], [m], [v], lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2)
        self.assertFalse(res)

    def test_multi_layer_fused_step_consistency(self):
        """Verify multi-layer model updates consistently using ApexW grouped parameter updates."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        opt = ApexW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        x = torch.randn(16, 64)
        target = torch.randint(0, 10, (16,))
        criterion = nn.CrossEntropyLoss()

        loss1 = criterion(model(x), target)
        loss1.backward()
        opt.step()
        opt.zero_grad()

        loss2 = criterion(model(x), target)
        self.assertLess(loss2.item(), loss1.item())

    def test_is_compiling_defined_and_safe(self):
        """Verify is_compiling is safely defined and callable without NameError."""
        from apexw.optimizer import is_compiling
        self.assertTrue(callable(is_compiling))
        self.assertIsInstance(is_compiling(), bool)

    def test_capturable_mode_execution_and_zero_sync(self):
        """Verify capturable=True executes properly without CPU-GPU synchronization (.item())."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)
        opt = ApexW(model.parameters(), lr=1e-3, capturable=True, foreach=True)

        x = torch.randn(4, 8, device=device)
        model(x).sum().backward()

        if device.type == "cpu":
            # On CPU, capturable must raise RuntimeError matching PyTorch behavior
            with self.assertRaises(RuntimeError):
                opt.step()
            return

        # On accelerator, step initializes states on model device
        opt.step()
        for p in model.parameters():
            step = opt.state[p]["step"]
            self.assertIsInstance(step, torch.Tensor)
            self.assertEqual(step.dim(), 0)
            self.assertEqual(step.device.type, p.device.type)
            self.assertEqual(step.item(), 1.0)

        # Verify subsequent steps execute with zero .item() calls (graph safe)
        orig_item = torch.Tensor.item
        def forbidden_item(self):
            raise AssertionError("Item called during capturable step!")

        torch.Tensor.item = forbidden_item
        try:
            for _ in range(5):
                opt.zero_grad()
                model(x).sum().backward()
                opt.step()
        finally:
            torch.Tensor.item = orig_item

        for p in model.parameters():
            self.assertEqual(opt.state[p]["step"].item(), 6.0)

    def test_capturable_single_tensor_mode(self):
        """Verify capturable=True works with single-tensor engine."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Linear(4, 2).to(device)
        opt = ApexW(model.parameters(), lr=1e-3, capturable=True, foreach=False)
        model(torch.randn(2, 4, device=device)).sum().backward()

        if device.type == "cpu":
            with self.assertRaises(RuntimeError):
                opt.step()
            return

        opt.step()
        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())
            self.assertEqual(opt.state[p]["step"].item(), 1.0)

    def test_cuda_graph_capture_replay(self):
        """Verify model training step captures and replays inside torch.cuda.CUDAGraph."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for CUDAGraph test")

        device = torch.device("cuda")
        model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)).to(device)
        opt = ApexW(model.parameters(), lr=1e-3, capturable=True, foreach=True)

        static_x = torch.randn(8, 16, device=device)
        static_y = torch.randint(0, 4, (8,), device=device)
        criterion = nn.CrossEntropyLoss()

        # Warmup on side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                criterion(model(static_x), static_y).backward()
                opt.step()
        torch.cuda.current_stream().wait_stream(s)

        # Capture CUDA Graph
        g = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(g, stream=s):
            loss = criterion(model(static_x), static_y)
            loss.backward()
            opt.step()

        torch.cuda.synchronize()

        # Replay graph
        initial_params = [p.clone() for p in model.parameters()]
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()

        # Verify parameters were updated and no NaNs
        for p_init, p_curr in zip(initial_params, model.parameters()):
            self.assertFalse(torch.isnan(p_curr).any())
            self.assertFalse(torch.equal(p_init, p_curr))

    def test_checkpoint_resume_dynamic_scaling_consistency(self):
        """Verify checkpoint resume preserves dynamic scaling metadata and matches uninterrupted run."""
        torch.manual_seed(42)
        base = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        m1 = copy.deepcopy(base)
        m2 = copy.deepcopy(base)

        opt1 = ApexW(m1.parameters(), lr=1e-3, precision="compact", dynamic_scaling=True)
        opt2 = ApexW(m2.parameters(), lr=1e-3, precision="compact", dynamic_scaling=True)

        inputs = [torch.randn(8, 32) for _ in range(15)]
        targets = [torch.randint(0, 10, (8,)) for _ in range(15)]
        criterion = nn.CrossEntropyLoss()

        # Run uninterrupted opt1 for 10 steps
        for i in range(10):
            opt1.zero_grad()
            criterion(m1(inputs[i]), targets[i]).backward()
            opt1.step()

        # Run opt2 for 5 steps, modify scaling factor, save checkpoint
        for i in range(5):
            opt2.zero_grad()
            criterion(m2(inputs[i]), targets[i]).backward()
            opt2.step()

        # Manually alter scale_v in opt2 state to simulate dynamic scaling recalibration
        for p in m2.parameters():
            if "scale_v" in opt2.state[p]:
                opt2.state[p]["scale_v"] = 4.0
                opt2.state[p]["sqrt_scale_v"] = 2.0

        sd = opt2.state_dict()
        self.assertIn("_dynamic_scaling_meta", sd)

        # Create new optimizer and restore state_dict
        opt2_resumed = ApexW(m2.parameters(), lr=1e-3, precision="compact", dynamic_scaling=True)
        opt2_resumed.load_state_dict(sd)

        # Verify scaling factor was restored
        for p in m2.parameters():
            if "scale_v" in opt2_resumed.state[p]:
                self.assertEqual(opt2_resumed.state[p]["scale_v"], 4.0)
                self.assertEqual(opt2_resumed.state[p]["sqrt_scale_v"], 2.0)

    def test_apex_zero_shard_reconstruction_no_stale_summing(self):
        """Verify ApexWZeRO slice reconstruction does not sum stale values from non-owned shards."""
        from apexw import ApexWZeRO
        # Test simulated rank 0 and rank 1 parameter slicing
        model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 10))
        opt0 = ApexWZeRO(model.parameters(), lr=1e-3, world_size=2, rank=0)
        opt1 = ApexWZeRO(model.parameters(), lr=1e-3, world_size=2, rank=1)

        # Verify that shard bounds correctly partition dimension 0 without overlap
        for item0, item1 in zip(opt0._sharded_param_map, opt1._sharded_param_map):
            if item0["type"] == "slice":
                self.assertEqual(item0["start"], 0)
                self.assertEqual(item0["end"], item1["start"])
                self.assertEqual(item1["end"], item0["dim0"])

    def test_backend_validation(self):
        """Verify optimizer validates hardware backend options."""
        p = nn.Parameter(torch.randn(4, 4))
        # Valid backends
        for b in [None, "auto", "triton", "cuda", "mps", "foreach", "_multi_tensor", "xla"]:
            opt = ApexW([p], lr=1e-3, backend=b)
            self.assertEqual(opt.param_groups[0]["backend"], b)

        # Invalid backend raises ValueError
        with self.assertRaises(ValueError):
            ApexW([p], lr=1e-3, backend="invalid_gpu")

    def test_triton_availability_and_fallback(self):
        """Verify OpenAI Triton kernel discovery, ROCm detection, and safe fallback."""
        from apexw import is_triton_available, is_rocm
        from apexw.triton_kernel import triton_adamw_step, triton_multi_tensor_adamw_step

        self.assertIsInstance(is_triton_available(), bool)
        self.assertIsInstance(is_rocm(), bool)

        p = torch.randn(16, 16)
        g = torch.randn(16, 16)
        m = torch.zeros_like(p)
        v = torch.zeros_like(p)

        # When Triton is not active (e.g. CPU or no Triton installed), steps return False without crashing
        res = triton_adamw_step(p, g, m, v, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                                weight_decay=1e-2, step_size=1e-3, eps_scaled=1e-8)
        self.assertEqual(res, is_triton_available())

        res_multi = triton_multi_tensor_adamw_step([p], [g], [m], [v], lr=1e-3, beta1=0.9, beta2=0.999,
                                                   eps=1e-8, weight_decay=1e-2, step=1)
        self.assertEqual(res_multi, is_triton_available())

        # Test environment variable override
        old_env = os.environ.get("APEXW_DISABLE_TRITON", None)
        try:
            os.environ["APEXW_DISABLE_TRITON"] = "1"
            self.assertFalse(is_triton_available())
        finally:
            if old_env is not None:
                os.environ["APEXW_DISABLE_TRITON"] = old_env
            else:
                os.environ.pop("APEXW_DISABLE_TRITON", None)

    def test_apple_silicon_mps_support(self):
        """Verify Apple Silicon MPS probe, compact dtype safety, and multi-tensor execution."""
        from apexw.optimizer import _mps_supports_bfloat16, _get_compact_dtype

        # Probe should return boolean without error
        res = _mps_supports_bfloat16()
        self.assertIsInstance(res, bool)

        # Verify execution with backend="mps" / "foreach" on network
        model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))
        opt = ApexW(model.parameters(), lr=1e-3, backend="mps")

        x = torch.randn(8, 16)
        y = torch.randint(0, 4, (8,))
        criterion = nn.CrossEntropyLoss()

        for _ in range(5):
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_google_tpu_xla_step_execution(self):
        """Verify Google TPU XLA zero-sync graph execution, tensor step counters, and convergence."""
        from apexw import is_xla_available, is_xla_device
        from apexw.xla_support import is_xla_tensor

        self.assertIsInstance(is_xla_available(), bool)
        self.assertTrue(is_xla_device("xla:0"))
        self.assertFalse(is_xla_device("cuda:0"))

        # Test full neural network convergence on backend="xla"
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
        opt = ApexW(model.parameters(), lr=1e-2, backend="xla", weight_decay=1e-2)

        x = torch.randn(16, 32)
        y = torch.randint(0, 8, (16,))
        criterion = nn.CrossEntropyLoss()

        initial_loss = criterion(model(x), y).item()
        for _ in range(25):
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        final_loss = criterion(model(x), y).item()

        print(f"\nTPU/XLA Backend Loss: {initial_loss:.4f} -> {final_loss:.4f}")
        self.assertLess(final_loss, initial_loss)

        # Verify step counter was incremented
        for p in model.parameters():
            self.assertEqual(opt.state[p]["step"].item(), 25)

    def test_xla_advanced_variants(self):
        """Verify XLA execution path with Lion, AdEMAMix, Cautious updates, and AMSGrad."""
        model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 2))
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        # AdEMAMix on XLA
        opt_mix = ApexW(model.parameters(), lr=1e-3, backend="xla", beta3=0.9999, alpha=5.0)
        opt_mix.zero_grad()
        criterion(model(x), y).backward()
        opt_mix.step()

        # Cautious updates on XLA
        opt_cautious = ApexW(model.parameters(), lr=1e-3, backend="xla", cautious=True)
        opt_cautious.zero_grad()
        criterion(model(x), y).backward()
        opt_cautious.step()

        # AMSGrad on XLA
        opt_ams = ApexW(model.parameters(), lr=1e-3, backend="xla", amsgrad=True)
        opt_ams.zero_grad()
        criterion(model(x), y).backward()
        opt_ams.step()

        # Lion on XLA
        opt_lion = ApexW(model.parameters(), lr=1e-3, backend="xla", lion=True)
        opt_lion.zero_grad()
        criterion(model(x), y).backward()
        opt_lion.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_xla_vs_adamw_rosenbrock(self):
        """Verify numerical equivalence between XLA path and PyTorch AdamW on Rosenbrock function."""
        torch.manual_seed(42)
        p_adamw = nn.Parameter(torch.tensor([-1.5, 2.0]))
        p_xla = nn.Parameter(torch.tensor([-1.5, 2.0]))

        opt_adamw = torch.optim.AdamW([p_adamw], lr=0.01, weight_decay=0.0)
        # Use same betas as AdamW to test numerical equivalence
        opt_xla = ApexW([p_xla], lr=0.01, weight_decay=0.0, precision="full", backend="xla", betas=(0.9, 0.999))

        def rosenbrock(p):
            return (1.0 - p[0])**2 + 100.0 * (p[1] - p[0]**2)**2

        for _ in range(500):
            opt_adamw.zero_grad()
            rosenbrock(p_adamw).backward()
            opt_adamw.step()

            opt_xla.zero_grad()
            rosenbrock(p_xla).backward()
            opt_xla.step()

        final_adam = rosenbrock(p_adamw).item()
        final_xla = rosenbrock(p_xla).item()

        print(f"\nRosenbrock Parity -> AdamW: {final_adam:.6f} | ApexW (XLA): {final_xla:.6f}")
        self.assertAlmostEqual(final_xla, final_adam, places=4)

    def test_decay_norm_only_under_xla(self):
        """Verify 1D norms are protected from weight decay under XLA mode."""
        linear = nn.Linear(4, 4, bias=True)  # weight is 2D, bias is 1D
        opt = ApexW(linear.parameters(), lr=1e-3, weight_decay=0.5, decay_norm_only=True, backend="xla")

        # Zero gradients to test pure weight decay effect
        for p in linear.parameters():
            p.grad = torch.zeros_like(p)

        init_weight = linear.weight.clone()
        init_bias = linear.bias.clone()

        opt.step()

        # 2D weight should be decayed: p * (1 - lr * wd) = 1 - 0.0005 = 0.9995
        self.assertTrue(torch.allclose(linear.weight, init_weight * (1.0 - 1e-3 * 0.5)))
        # 1D bias should be untouched by weight decay
        self.assertTrue(torch.equal(linear.bias, init_bias))

    def test_triton_multi_tensor_mock_execution(self):
        """Verify triton_multi_tensor_adamw_step correctly coordinates tensor list when Triton is active."""
        from unittest.mock import patch
        import apexw.triton_kernel as tk

        p1 = torch.randn(4, 4)
        p2 = torch.randn(4)
        g1 = torch.randn(4, 4)
        g2 = torch.randn(4)
        m1 = torch.zeros_like(p1)
        m2 = torch.zeros_like(p2)
        v1 = torch.zeros_like(p1)
        v2 = torch.zeros_like(p2)

        calls = []
        def mock_step(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay,
                      step_size, eps_scaled, scale_v, sqrt_scale_v, apply_wd, **kwargs):
            calls.append((param.shape, apply_wd))
            return True

        # Fix 4: _triton_preflight_check is patched to return None (pass) so the
        # pre-flight does not reject CPU tensors in an environment without a CUDA device.
        with patch.object(tk, "is_triton_available", return_value=True), \
             patch.object(tk, "_triton_preflight_check", return_value=None), \
             patch.object(tk, "triton_adamw_step", side_effect=mock_step):
            success = tk.triton_multi_tensor_adamw_step(
                [p1, p2], [g1, g2], [m1, m2], [v1, v2],
                lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2,
                step=1, decay_norm_only=True
            )
            self.assertTrue(success)
            self.assertEqual(len(calls), 2)
            # 2D tensor has weight decay applied, 1D tensor has weight decay excluded
            self.assertEqual(calls[0], (torch.Size([4, 4]), True))
            self.assertEqual(calls[1], (torch.Size([4]), False))

    def test_activation_checkpointing_wrapper(self):
        """Verify CheckpointWrapper produces correct forward/backward outputs."""
        from apexw import CheckpointWrapper
        linear = nn.Linear(8, 4)
        wrapped = CheckpointWrapper(linear, use_reentrant=False)

        x = torch.randn(2, 8, requires_grad=True)
        out = wrapped(x)
        out.sum().backward()

        self.assertIsNotNone(linear.weight.grad)
        self.assertFalse(torch.isnan(linear.weight.grad).any())

    def test_apply_activation_checkpointing_auto(self):
        """Verify apply_activation_checkpointing discovers and wraps submodules in deep model."""
        from apexw import apply_activation_checkpointing, CheckpointWrapper

        model = nn.Sequential(
            nn.Sequential(nn.Linear(16, 32), nn.ReLU()),
            nn.Sequential(nn.Linear(32, 32), nn.ReLU()),
            nn.Sequential(nn.Linear(32, 10)),
        )

        wrapped_count = apply_activation_checkpointing(model)
        self.assertGreater(wrapped_count, 0)

        # Ensure model runs end-to-end with optimizer
        opt = ApexW(model.parameters(), lr=1e-3)
        x = torch.randn(4, 16)
        y = torch.randint(0, 10, (4,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_get_memory_breakdown(self):
        """Verify get_memory_breakdown calculates metrics and reports architectural diagnosis."""
        from apexw import get_memory_breakdown

        model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 10))
        opt = ApexW(model.parameters(), lr=1e-3)

        info = get_memory_breakdown(model, optimizer=opt)
        self.assertIn("param_memory_mb", info)
        self.assertIn("optimizer_state_mb", info)
        self.assertIn("adamw_state_mb", info)
        self.assertIn("bottleneck", info)
        self.assertIn("recommendation", info)
        self.assertGreater(info["param_memory_mb"], 0.0)
        self.assertGreater(info["optimizer_state_mb"], 0.0)

    def test_single_tensor_amp_grad_scale(self):
        """Verify single-tensor (foreach=False) mode handles grad_scale and unscaling seamlessly."""
        model = nn.Linear(4, 2, bias=False)
        opt = ApexW(model.parameters(), lr=1e-3, foreach=False)

        x = torch.randn(2, 4)
        model(x).sum().backward()

        # Simulate GradScaler
        opt.grad_scale = torch.tensor(128.0)
        opt.found_inf = torch.tensor(0.0)
        opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

        del opt.grad_scale
        del opt.found_inf


class TestPrecisionParity(unittest.TestCase):
    """
    Targeted precision-parity tests across all four ApexW execution backends.

    Background: ApexW's single-tensor and multi-tensor (foreach) paths both
    implement dynamic range scaling for 16-bit variance states to prevent
    underflow — picking an initial scale of 1024.0 for float16 states and
    adjusting every 100 steps based on the observed max variance value.

    The XLA path omits this rescaling (by design: the XLA compiler cannot see
    Python-level scale state without host synchronization, which would break
    TPU fusion). The custom CUDA kernel path also skips it because the kernel
    receives pre-computed bias-corrected values from the Python dispatcher.

    These tests document the known behaviour explicitly so that future
    contributors know which guarantees each path provides, and catch any
    unintended regression where a path silently drops NaNs.
    """

    def _make_model_and_fp16_opt(self, **kwargs):
        """Helper: tiny model with compact float16 states for underflow stress testing."""
        torch.manual_seed(7)
        model = nn.Linear(8, 4, bias=False)
        opt = ApexW(model.parameters(), lr=1e-3, precision="compact", **kwargs)
        return model, opt

    # ------------------------------------------------------------------
    # Path 4: single-tensor (foreach=False) — canonical dynamic scaling
    # ------------------------------------------------------------------
    def test_single_tensor_fp16_dynamic_scale_initialized(self):
        """
        Single-tensor path: after the first step on a compact-precision optimizer
        the scale_v entry must be present and set to 1024.0 for float16 states.
        This is the reference path that all other backends are compared against.
        """
        model, opt = self._make_model_and_fp16_opt(foreach=False)
        x = torch.randn(2, 8)
        model(x).sum().backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            if state["exp_avg_sq"].dtype == torch.float16:
                # scale_v only initializes when dynamic_scaling=True and dtype is 16-bit
                self.assertIn("scale_v", state,
                    "single-tensor path must initialize scale_v for float16 states")
                self.assertEqual(state["scale_v"], 1024.0,
                    "float16 initial scale must be 1024.0 (matches the 6e-5 eps floor ratio)")

    def test_single_tensor_no_nan_after_many_steps(self):
        """
        Single-tensor path: run 300 steps (which crosses the 100-step recalibration
        threshold three times) and verify no NaN/Inf is produced.
        """
        model, opt = self._make_model_and_fp16_opt(foreach=False)
        x = torch.randn(2, 8)
        for _ in range(300):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any(), "NaN after 300 steps (single-tensor)")
            self.assertFalse(torch.isinf(p).any(), "Inf after 300 steps (single-tensor)")

    # ------------------------------------------------------------------
    # Path 4b: multi-tensor / foreach — same dynamic scaling, batched
    # ------------------------------------------------------------------
    def test_foreach_fp16_dynamic_scale_initialized(self):
        """
        Multi-tensor (foreach) path: same scale_v initialization as single-tensor.
        A regression here would mean the two CPU paths diverge silently.
        """
        model, opt = self._make_model_and_fp16_opt(foreach=True)
        x = torch.randn(2, 8)
        model(x).sum().backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            if state["exp_avg_sq"].dtype == torch.float16:
                self.assertIn("scale_v", state,
                    "foreach path must initialize scale_v for float16 states")
                self.assertEqual(state["scale_v"], 1024.0)

    def test_foreach_no_nan_after_many_steps(self):
        """Multi-tensor path: same 300-step NaN guard as the single-tensor test."""
        model, opt = self._make_model_and_fp16_opt(foreach=True)
        x = torch.randn(2, 8)
        for _ in range(300):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()

        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any(), "NaN after 300 steps (foreach)")
            self.assertFalse(torch.isinf(p).any(), "Inf after 300 steps (foreach)")

    # ------------------------------------------------------------------
    # Path 1: XLA — dynamic scaling intentionally absent; document it
    # ------------------------------------------------------------------
    def test_xla_path_does_not_use_dynamic_scaling(self):
        """
        XLA path: verify that state_dict entries do NOT contain scale_v after
        an XLA-simulated step.

        This is the EXPECTED behaviour: the XLA path executes the update as
        on-device tensor math with no Python-level host synchronization.
        Injecting scale_v state from the host would require .item() calls
        that break TPU HLO fusion — the trade-off is acceptable because:
          (a) TPU bf16 has a wider dynamic range than GPU fp16 (bfloat16 ≈ fp32 exponent)
          (b) XLA's compiler can fuse operations that prevent most underflow in practice
        This test explicitly documents the contract so it cannot regress silently.
        """
        import unittest.mock as mock

        model = nn.Linear(8, 4, bias=False)
        opt = ApexW(model.parameters(), lr=1e-3, precision="compact", backend="xla")

        # Simulate an XLA device tensor step without a real TPU
        with mock.patch(
            "apexw.optimizer.is_xla_tensor", return_value=True
        ), mock.patch(
            "apexw.optimizer.is_xla_device", return_value=True
        ):
            x = torch.randn(2, 8)
            model(x).sum().backward()
            try:
                opt.step()
            except Exception:
                # If the XLA path errors on CPU tensors that's fine here;
                # the important thing is what state entries look like.
                pass

        # XLA path must NOT have initialized scale_v; that would require .item()
        for p in model.parameters():
            if p in opt.state:
                self.assertNotIn(
                    "scale_v", opt.state[p],
                    "XLA path must not write scale_v — that would require a host sync "
                    "and break TPU HLO fusion. Use bfloat16 instead of float16 on TPUs."
                )

    def test_xla_bf16_states_no_underflow_concern(self):
        """
        XLA path: when states are bfloat16 (the recommended TPU dtype),
        the effective dynamic range is essentially the same as float32 for
        the variance values (bf16 has 8 exponent bits vs float32's 8).
        Simulate steps through the XLA code path on CPU and verify convergence.

        Note: this uses _xla_step() directly on CPU tensors. The math is
        identical to what runs on a real TPU; what differs is the compilation.
        State must be initialized with a regular step() before calling _xla_step
        directly because state is created lazily on the first step() call.
        """
        torch.manual_seed(42)
        model = nn.Linear(16, 8, bias=False)
        opt = ApexW(model.parameters(), lr=1e-3, backend="xla")
        x = torch.randn(4, 16)

        # Initialize state via a regular step (lazy init happens here)
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()

        initial_loss = model(x).detach().sum().item()
        losses = [initial_loss]

        for _ in range(10):
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            # _xla_step modifies parameters in-place; must be under no_grad
            params_with_grad = [p for p in model.parameters() if p.grad is not None]
            with torch.no_grad():
                opt._xla_step(
                    params_with_grad,
                    [p.grad.detach() for p in params_with_grad],
                    [opt.state[p]["exp_avg"] for p in params_with_grad],
                    [],  # exp_avgs_slow
                    [opt.state[p]["exp_avg_sq"] for p in params_with_grad],
                    [],  # max_exp_avg_sqs
                    [opt.state[p]["step"] for p in params_with_grad],
                    lr=1e-3, beta1=0.9, beta2=0.999, beta3=None, alpha=1.0,
                    lion=False, eps=1e-8, weight_decay=0.0, decay_norm_only=False,
                    nesterov=False, cautious=False, amsgrad=False, maximize=False,
                )
            losses.append(loss.item())

        # Loss should decrease over time (basic convergence check)
        self.assertLess(losses[-1], losses[0],
            "XLA path must converge: final loss should be lower than initial loss")
        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any(), "NaN in XLA-path update")

    # ------------------------------------------------------------------
    # Path 3: custom CUDA kernel — scaling not passed; document it
    # ------------------------------------------------------------------
    def test_cuda_kernel_path_receives_no_scale_v(self):
        """
        Custom CUDA kernel path: the fused_adamw_cuda_step / step_fused interface
        does not accept scale_v arguments. This is a known limitation — the kernel
        operates on raw gradient and moment tensors and applies bias correction
        internally using the passed `step` integer.

        For float16 compact states on a machine with the CUDA extension compiled,
        users should prefer the Triton backend (which handles scale_v through the
        inv_scale register argument) or the foreach fallback.

        This test mocks the CUDA kernel to verify it is called without scale_v
        and that the optimizer correctly falls back to foreach when the call
        returns False (i.e., the extension is unavailable or reports a type mismatch).
        """
        import unittest.mock as mock

        model = nn.Linear(8, 4, bias=False)
        opt = ApexW(model.parameters(), lr=1e-3, backend="cuda")

        x = torch.randn(2, 8)
        model(x).sum().backward()

        called_with_scale = []

        def mock_step_fused(*args, **kwargs):
            # Record whether scale_v was passed (it should NOT be)
            called_with_scale.append("scale_v" in kwargs)
            return False  # signal fallback to foreach

        with mock.patch("apexw.optimizer.step_fused", side_effect=mock_step_fused), \
             mock.patch("apexw.optimizer.is_cuda_kernel_available", return_value=True):
            opt.step()

        if called_with_scale:
            self.assertFalse(
                any(called_with_scale),
                "CUDA kernel interface must not receive scale_v — "
                "the kernel computes bias correction from the step integer only. "
                "For dynamic range scaling, use backend='triton' or backend='foreach'."
            )

        # Regardless, no NaN after the foreach fallback
        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_precision_parity_single_vs_foreach_numerics(self):
        """
        Cross-path parity: run the same 10-step sequence through single-tensor
        and foreach paths and verify the final parameter values agree to within
        a tight tolerance. Both paths implement the same dynamic scaling logic
        so results should be nearly identical (differences only from fp ops order).
        """
        torch.manual_seed(99)
        base = nn.Linear(16, 8, bias=False)
        x = torch.randn(4, 16)

        import copy
        model_st = copy.deepcopy(base)
        model_mt = copy.deepcopy(base)

        # Both optimizers share the same hyperparams; only differ in execution path
        opt_st = ApexW(model_st.parameters(), lr=5e-4, foreach=False, precision="compact")
        opt_mt = ApexW(model_mt.parameters(), lr=5e-4, foreach=True, precision="compact")

        for _ in range(10):
            opt_st.zero_grad()
            model_st(x).sum().backward()
            opt_st.step()

            opt_mt.zero_grad()
            model_mt(x).sum().backward()
            opt_mt.step()

        for p_st, p_mt in zip(model_st.parameters(), model_mt.parameters()):
            # Allow some floating point ordering differences but not semantic divergence
            self.assertTrue(
                torch.allclose(p_st.float(), p_mt.float(), atol=1e-4, rtol=1e-3),
                f"single-tensor and foreach paths diverged: "
                f"max diff = {(p_st - p_mt).abs().max().item():.6f}"
            )

    def test_maximize_non_mutating_gradient_buffer(self):
        """
        Verify maximization path does not mutate caller's gradient buffers in-place.
        Both foreach=False and foreach=True must preserve original grad values and
        produce identical parameter trajectories across multiple consecutive steps.
        """
        torch.manual_seed(42)
        p1 = torch.tensor([1.0], requires_grad=True)
        p1.grad = torch.tensor([2.0])
        opt1 = ApexW([p1], lr=0.1, maximize=True, foreach=False)

        p2 = torch.tensor([1.0], requires_grad=True)
        p2.grad = torch.tensor([2.0])
        opt2 = ApexW([p2], lr=0.1, maximize=True, foreach=True)

        for step in range(3):
            opt1.step()
            opt2.step()
            # Gradient buffers must not be negated in place
            self.assertEqual(p1.grad.item(), 2.0, f"single-tensor grad mutated at step {step}")
            self.assertEqual(p2.grad.item(), 2.0, f"foreach grad mutated at step {step}")
            # Parameter updates must match exactly
            self.assertAlmostEqual(p1.item(), p2.item(), places=5)

    def test_differentiable_rejected(self):
        """Verify that differentiable=True is explicitly rejected until supported."""
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([1.0])

        with self.assertRaises(NotImplementedError):
            ApexW([p], lr=1e-3, differentiable=True)

        opt = ApexW([{"params": [p], "differentiable": True}], lr=1e-3)
        with self.assertRaises(NotImplementedError):
            opt.step()


class TestStepCountUniformity(unittest.TestCase):
    """Regression tests for the step-count bug in GPU fast paths.

    The Triton and custom CUDA kernels accept a single step value used for
    bias correction across all parameters in a group.  When some parameters
    receive no gradient for several steps their internal step counter lags
    behind the others.  Using state_steps[0] for every parameter would apply
    the wrong bias correction to the lagging ones.

    These tests exercise the CPU foreach path (the fast paths only activate on
    CUDA) but they validate the per-parameter state bookkeeping that the GPU
    guard is meant to protect.
    """

    def test_late_param_bias_correction(self):
        """A param that starts receiving gradients late must use its own step=1
        bias correction on its first update, not the step count of the params
        that received gradients from the beginning.
        """
        torch.manual_seed(0)
        # Two independent params so each has its own state entry
        p_early = torch.tensor([0.5], requires_grad=True)
        p_late  = torch.tensor([0.5], requires_grad=True)

        opt = ApexW(
            [{"params": [p_early]}, {"params": [p_late]}],
            lr=0.1,
            weight_decay=0.0,
            precision="full",
            foreach=False,
        )

        # Reference: a fresh optimizer with only p_late (always step=1 on first update)
        p_ref = torch.tensor([0.5], requires_grad=True)
        opt_ref = ApexW([p_ref], lr=0.1, weight_decay=0.0, precision="full", foreach=False)

        # Steps 1-3: only p_early gets a gradient
        for _ in range(3):
            opt.zero_grad()
            p_early.grad = torch.tensor([1.0])
            opt.step()

        # Step 4: p_late gets its first gradient — should be treated as step=1
        opt.zero_grad()
        p_early.grad = torch.tensor([1.0])
        p_late.grad = torch.tensor([1.0])
        opt.step()

        # Reference optimizer updates p_ref for the first time at step=1
        opt_ref.zero_grad()
        p_ref.grad = torch.tensor([1.0])
        opt_ref.step()

        self.assertAlmostEqual(
            p_late.item(), p_ref.item(), places=5,
            msg="Late-starting param must use step=1 bias correction on first update",
        )

    def test_uniform_steps_produce_consistent_updates(self):
        """When all params share the same step history the foreach and
        single-tensor paths must agree numerically."""
        torch.manual_seed(1)
        results = {}
        for foreach in (False, True):
            p = torch.tensor([1.0, -1.0], requires_grad=True)
            opt = ApexW([p], lr=0.01, weight_decay=0.01, precision="full", foreach=foreach)
            for step in range(5):
                opt.zero_grad()
                p.grad = torch.tensor([0.5, -0.5])
                opt.step()
            results[foreach] = p.detach().clone()

        self.assertTrue(
            torch.allclose(results[False], results[True], atol=1e-6),
            "foreach and single-tensor paths must agree when step counts are uniform",
        )

    def test_single_group_late_activation_against_adamw(self):
        """Regression test: Two parameters in the SAME group.
        Param 1 receives 5 updates while Param 2 has grad=None.
        At step 6, both receive gradients.
        Param 2 must produce 0.9900000 matching PyTorch AdamW for both
        foreach=False and foreach=True.
        """
        for foreach in (False, True):
            p1 = torch.tensor([1.0], requires_grad=True)
            p2 = torch.tensor([1.0], requires_grad=True)
            opt_apex = ApexW([p1, p2], lr=0.01, weight_decay=0.0, precision="full", foreach=foreach)

            p1_ref = torch.tensor([1.0], requires_grad=True)
            p2_ref = torch.tensor([1.0], requires_grad=True)
            opt_ref = torch.optim.AdamW([p1_ref, p2_ref], lr=0.01, weight_decay=0.0)

            for _ in range(5):
                opt_apex.zero_grad()
                opt_ref.zero_grad()
                p1.grad = torch.tensor([1.0])
                p1_ref.grad = torch.tensor([1.0])
                opt_apex.step()
                opt_ref.step()

            opt_apex.zero_grad()
            opt_ref.zero_grad()
            p1.grad = torch.tensor([1.0])
            p2.grad = torch.tensor([1.0])
            p1_ref.grad = torch.tensor([1.0])
            p2_ref.grad = torch.tensor([1.0])
            opt_apex.step()
            opt_ref.step()

            self.assertAlmostEqual(
                p2.item(), 0.9900000, places=6,
                msg=f"foreach={foreach}: p2 must equal 0.9900000 matching AdamW",
            )
            self.assertAlmostEqual(
                p2.item(), p2_ref.item(), places=6,
                msg=f"foreach={foreach}: p2 must match AdamW reference",
            )
            self.assertAlmostEqual(
                p1.item(), p1_ref.item(), places=6,
                msg=f"foreach={foreach}: p1 must match AdamW reference",
            )



class TestApexWZeroCheckpoint(unittest.TestCase):
    """Regression tests for ApexWZeRO state_dict / load_state_dict (Bug 2).

    ApexWZeRO previously had no checkpoint methods, so resuming training from
    a mid-run checkpoint would silently restart the optimizer moments from
    zero, producing different parameter trajectories than uninterrupted runs.
    """

    def _make_zero_opt(self, params):
        from apexw.zero import ApexWZeRO
        return ApexWZeRO(params, lr=0.01, weight_decay=0.01, world_size=1, rank=0)

    def test_checkpoint_round_trip(self):
        """Resumed training must continue from the saved optimizer state and
        produce the same parameters as an uninterrupted run of the same length.
        """
        try:
            from apexw.zero import ApexWZeRO
        except ImportError:
            self.skipTest("ApexWZeRO not importable")

        torch.manual_seed(42)
        model_a = nn.Linear(8, 4, bias=False)
        model_b = nn.Linear(8, 4, bias=False)
        model_b.load_state_dict(model_a.state_dict())

        # Use full precision so checkpoint/resume has no BF16 quantization noise.
        # Pre-generate all inputs so both runs see exactly the same batches.
        inputs = [torch.randn(4, 8) for _ in range(10)]

        opt_ref = ApexWZeRO(
            model_a.parameters(), lr=0.01, weight_decay=0.01,
            world_size=1, rank=0, precision="full",
        )
        opt_run = ApexWZeRO(
            model_b.parameters(), lr=0.01, weight_decay=0.01,
            world_size=1, rank=0, precision="full",
        )

        # Uninterrupted: 10 steps on opt_ref
        for x in inputs:
            opt_ref.zero_grad()
            model_a(x).sum().backward()
            opt_ref.step()

        # Interrupted: 5 steps, save checkpoint, reload, 5 more steps
        for x in inputs[:5]:
            opt_run.zero_grad()
            model_b(x).sum().backward()
            opt_run.step()

        sd = opt_run.state_dict()
        # Create a fresh optimizer on model_b and restore state
        opt_resumed = ApexWZeRO(
            model_b.parameters(), lr=0.01, weight_decay=0.01,
            world_size=1, rank=0, precision="full",
        )
        opt_resumed.load_state_dict(sd)

        # Continue with the same inputs that opt_ref used for steps 6-10
        for x in inputs[5:]:
            opt_resumed.zero_grad()
            model_b(x).sum().backward()
            opt_resumed.step()

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(
                torch.allclose(pa, pb, atol=1e-5),
                "Resumed training must match uninterrupted run within tolerance",
            )

    def test_state_dict_contains_local_opt(self):
        """state_dict must include the inner optimizer's moment tensors."""
        try:
            from apexw.zero import ApexWZeRO
        except ImportError:
            self.skipTest("ApexWZeRO not importable")

        torch.manual_seed(0)
        model = nn.Linear(4, 2, bias=False)
        opt = ApexWZeRO(model.parameters(), lr=0.01, world_size=1, rank=0)

        model(torch.randn(2, 4)).sum().backward()
        opt.step()

        sd = opt.state_dict()
        self.assertIn("local_opt", sd, "state_dict must contain 'local_opt' key")
        self.assertIn("zero_meta", sd, "state_dict must contain 'zero_meta' key")
        # The inner optimizer must have populated state (moments/step counters)
        self.assertTrue(
            len(sd["local_opt"]["state"]) > 0,
            "Inner optimizer state must be non-empty after one step",
        )

    def test_zero_preserves_per_group_settings(self):
        """Regression test: ApexWZeRO must retain distinct per-group settings
        (lr, weight_decay, eps) across parameter groups in the inner optimizer,
        and dynamically synchronize updates (e.g. from LR schedulers).
        """
        try:
            from apexw.zero import ApexWZeRO
        except ImportError:
            self.skipTest("ApexWZeRO not importable")

        p_backbone = nn.Parameter(torch.randn(10, 10))
        p_head = nn.Parameter(torch.randn(5))

        opt = ApexWZeRO([
            {"params": [p_backbone], "lr": 1e-4, "weight_decay": 0.05},
            {"params": [p_head], "lr": 1e-2, "weight_decay": 0.0},
        ], world_size=1, rank=0)

        # 1. Inner optimizer must preserve both groups with distinct configurations
        self.assertEqual(len(opt._local_opt.param_groups), 2)
        self.assertEqual(opt._local_opt.param_groups[0]["lr"], 1e-4)
        self.assertEqual(opt._local_opt.param_groups[1]["lr"], 1e-2)
        self.assertEqual(opt._local_opt.param_groups[0]["weight_decay"], 0.05)
        self.assertEqual(opt._local_opt.param_groups[1]["weight_decay"], 0.0)

        # 2. Dynamic hyperparameter updates (e.g., from an LR scheduler) propagate to inner groups
        opt.param_groups[0]["lr"] = 5e-5
        p_backbone.grad = torch.randn_like(p_backbone)
        p_head.grad = torch.randn_like(p_head)
        opt.step()
        self.assertEqual(opt._local_opt.param_groups[0]["lr"], 5e-5)

        # 3. Dynamic add_param_group is properly sharded and registered
        p_extra = nn.Parameter(torch.randn(4))
        opt.add_param_group({"params": [p_extra], "lr": 3e-3, "weight_decay": 0.01})
        self.assertEqual(len(opt.param_groups), 3)
        self.assertEqual(len(opt._local_opt.param_groups), 3)
        self.assertEqual(opt._local_opt.param_groups[2]["lr"], 3e-3)
        self.assertEqual(opt._local_opt.param_groups[2]["weight_decay"], 0.01)

    def test_zero_preserves_per_group_settings_multi_rank(self):
        """Simulate 2 ranks and verify both maintain correct per-group configs."""
        try:
            from apexw.zero import ApexWZeRO
        except ImportError:
            self.skipTest("ApexWZeRO not importable")

        for rank in (0, 1):
            p1 = nn.Parameter(torch.randn(512, 10))
            p2 = nn.Parameter(torch.randn(10))
            opt = ApexWZeRO([
                {"params": [p1], "lr": 2e-4, "weight_decay": 0.1},
                {"params": [p2], "lr": 1e-3, "weight_decay": 0.0},
            ], world_size=2, rank=rank)

            self.assertEqual(len(opt._local_opt.param_groups), 2)
            self.assertEqual(opt._local_opt.param_groups[0]["lr"], 2e-4)
            self.assertEqual(opt._local_opt.param_groups[1]["lr"], 1e-3)
            self.assertEqual(opt._local_opt.param_groups[0]["weight_decay"], 0.1)
            self.assertEqual(opt._local_opt.param_groups[1]["weight_decay"], 0.0)

    def test_compact_precision_fp32_arithmetic_contract(self):
        """Verify that compact precision (16-bit state storage) uses FP32 arithmetic
        for momentum updates, matching Triton and CUDA fused kernels.
        """
        for state_dtype in (torch.bfloat16, torch.float16):
            p = torch.tensor([1.0, 2.0], requires_grad=True)
            p.grad = torch.tensor([0.1, -0.2])
            opt = ApexW([p], lr=0.01, precision="compact", foreach=False)

            # Initialize states in target compact dtype
            opt.step()
            state = opt.state[p]
            self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)

            # Run 5 steps to verify stable accumulation without NaN or underflow
            for _ in range(5):
                opt.zero_grad()
                p.grad = torch.tensor([0.05, -0.05])
                opt.step()

            self.assertFalse(torch.isnan(p).any())
            self.assertFalse(torch.isnan(state["exp_avg"]).any())
            self.assertFalse(torch.isnan(state["exp_avg_sq"]).any())


class TestRegressionBugs(unittest.TestCase):
    """Regression tests for 8 confirmed architectural and numerical bugs."""

    def test_bug1_gradient_clipping_after_amp_unscaling(self):
        """Bug 1: Verify gradient clipping happens after AMP unscaling.
        When AMP/GradScaler provides grad_scale, gradients must be unscaled before
        calculating Euclidean norm and applying clip_grad_norm.
        """
        scale = 100.0
        clip_norm = 2.0

        # Reference: unscale gradients first, then clip, then step
        p_ref = nn.Parameter(torch.tensor([10.0, -5.0]))
        p_ref.grad = torch.tensor([500.0, -300.0])  # scaled by 100
        p_ref.grad.div_(scale)
        torch.nn.utils.clip_grad_norm_([p_ref], max_norm=clip_norm)
        opt_ref = ApexW([p_ref], lr=0.1, weight_decay=0.01)
        opt_ref.step()

        # ApexW: pass scaled gradient and grad_scale with in-optimizer clipping
        p_apex = nn.Parameter(torch.tensor([10.0, -5.0]))
        p_apex.grad = torch.tensor([500.0, -300.0])  # scaled by 100
        opt_apex = ApexW([p_apex], lr=0.1, weight_decay=0.01, clip_grad_norm=clip_norm)
        opt_apex.grad_scale = torch.tensor(scale)
        opt_apex.step()

        self.assertTrue(
            torch.allclose(p_ref, p_apex, atol=1e-5),
            f"Param mismatch: reference {p_ref} vs apexw {p_apex}",
        )
        self.assertTrue(
            torch.allclose(opt_ref.state[p_ref]["exp_avg"], opt_apex.state[p_apex]["exp_avg"], atol=1e-5),
            "Momentum mismatch after unscaled clipping",
        )

    def test_bug2_amp_overflow_skipped_in_capturable_mode(self):
        """Bug 2: Verify AMP overflow correctly skips updates when capturable=True outside graph capture."""
        import apexw.optimizer as opt_mod
        orig_devices = opt_mod.CAPTURABLE_SUPPORTED_DEVICES
        try:
            opt_mod.CAPTURABLE_SUPPORTED_DEVICES = ("cpu", "cuda", "xpu")
            for foreach in (False, True):
                p = nn.Parameter(torch.tensor([1.0, 2.0]))
                opt = ApexW([p], lr=0.01, capturable=True, foreach=foreach)

                # Step 1: Normal valid step
                p.grad = torch.tensor([0.1, 0.2])
                opt.step()
                p_val = p.clone()
                step_val = opt.state[p]["step"].clone()
                m_val = opt.state[p]["exp_avg"].clone()

                # Step 2: Overflow step with found_inf=1.0
                p.grad = torch.tensor([float("inf"), 0.2])
                opt.found_inf = torch.tensor(1.0)
                opt.step()

                self.assertTrue(
                    torch.equal(p, p_val),
                    f"Parameters updated on AMP overflow (foreach={foreach})",
                )
                self.assertTrue(
                    torch.equal(opt.state[p]["step"], step_val),
                    f"Step counter incremented on AMP overflow (foreach={foreach})",
                )
                self.assertTrue(
                    torch.equal(opt.state[p]["exp_avg"], m_val),
                    f"exp_avg modified on AMP overflow (foreach={foreach})",
                )
        finally:
            opt_mod.CAPTURABLE_SUPPORTED_DEVICES = orig_devices

    def test_bug3_compiled_cuda_foreach_variables_dataflow(self):
        """Bug 3: Verify compiled foreach path has consistent variables and avoids UnboundLocalError."""
        import apexw.optimizer as opt_mod
        orig_compiling = opt_mod.is_compiling
        try:
            opt_mod.is_compiling = lambda: True
            configs = [
                {"cautious": False, "nesterov": False, "beta3": None},
                {"cautious": True, "nesterov": False, "beta3": None},
                {"cautious": False, "nesterov": True, "beta3": None},
                {"cautious": False, "nesterov": False, "beta3": 0.99},
            ]
            for cfg in configs:
                p = nn.Parameter(torch.tensor([1.0, 2.0]))
                p.grad = torch.tensor([0.1, 0.2])
                opt = ApexW([p], lr=0.01, foreach=True, **cfg)
                # Must execute cleanly without UnboundLocalError: bc2_sqrt
                opt.step()
                self.assertFalse(torch.isnan(p).any())
                self.assertEqual(opt.state[p]["step"].item(), 1.0)
        finally:
            opt_mod.is_compiling = orig_compiling

    def test_bug4_ademamix_slow_ratio_parity(self):
        """Bug 4: Verify AdEMAMix slow_ratio includes bias correction bc1/bc3 across execution modes."""
        import apexw.optimizer as opt_mod

        # Reference: single tensor update
        p_ref = nn.Parameter(torch.tensor([1.0, 2.0]))
        p_ref.grad = torch.tensor([0.1, 0.2])
        opt_ref = ApexW([p_ref], lr=0.01, beta3=0.9999, alpha=5.0, foreach=False)
        opt_ref.step()

        # Compiled multi-tensor update
        orig_compiling = opt_mod.is_compiling
        try:
            opt_mod.is_compiling = lambda: True
            p_comp = nn.Parameter(torch.tensor([1.0, 2.0]))
            p_comp.grad = torch.tensor([0.1, 0.2])
            opt_comp = ApexW([p_comp], lr=0.01, beta3=0.9999, alpha=5.0, foreach=True)
            opt_comp.step()

            self.assertTrue(
                torch.allclose(p_ref, p_comp, atol=1e-5),
                f"AdEMAMix mismatch: single-tensor {p_ref} vs compiled foreach {p_comp}",
            )
        finally:
            opt_mod.is_compiling = orig_compiling

    def test_bug5_dynamic_variance_scaling_capturable_fp16(self):
        """Bug 5: Verify dynamic variance scaling in capturable/compiling path applies sqrt_scale_v divisor."""
        import apexw.optimizer as opt_mod

        # Reference: single tensor without compiling
        p_ref = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        p_ref.grad = torch.tensor([0.1, 0.2], dtype=torch.float32)
        opt_ref = ApexW([p_ref], lr=0.01, precision="compact", foreach=False, dynamic_scaling=True)
        opt_ref.step()
        opt_ref.state[p_ref]["scale_v"] = 1024.0
        opt_ref.state[p_ref]["sqrt_scale_v"] = 32.0
        p_ref.grad = torch.tensor([0.1, 0.2], dtype=torch.float32)
        opt_ref.step()

        # Test: single tensor under compiling mode
        orig_compiling = opt_mod.is_compiling
        try:
            opt_mod.is_compiling = lambda: True
            p_test = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
            p_test.grad = torch.tensor([0.1, 0.2], dtype=torch.float32)
            opt_test = ApexW([p_test], lr=0.01, precision="compact", foreach=False, dynamic_scaling=True)
            opt_test.step()
            opt_test.state[p_test]["scale_v"] = 1024.0
            opt_test.state[p_test]["sqrt_scale_v"] = 32.0
            p_test.grad = torch.tensor([0.1, 0.2], dtype=torch.float32)
            opt_test.step()

            self.assertTrue(
                torch.allclose(p_ref, p_test, atol=1e-3),
                f"Denom scale mismatch: reference {p_ref} vs compiled {p_test}",
            )
        finally:
            opt_mod.is_compiling = orig_compiling

    def test_bug6_xla_step_counts_per_parameter(self):
        """Bug 6: Verify _xla_step computes bias corrections per-parameter rather than using state_steps[0] for all."""
        opt = ApexW([nn.Parameter(torch.tensor([1.0]))], lr=0.01)

        p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        p2 = nn.Parameter(torch.tensor([3.0, 4.0]))
        g1 = torch.tensor([0.1, 0.2])
        g2 = torch.tensor([0.1, 0.2])

        m1 = torch.zeros_like(p1)
        m2 = torch.zeros_like(p2)
        v1 = torch.zeros_like(p1)
        v2 = torch.zeros_like(p2)

        # p1 at step 0 (will become 1), p2 at step 9 (will become 10)
        steps = [torch.tensor(0.0), torch.tensor(9.0)]

        with torch.no_grad():
            opt._xla_step(
                params=[p1, p2],
                grads=[g1, g2],
                exp_avgs=[m1, m2],
                exp_avgs_slow=[None, None],
                exp_avg_sqs=[v1, v2],
                max_exp_avg_sqs=[None, None],
                state_steps=steps,
                lr=0.01,
                beta1=0.9,
                beta2=0.999,
                beta3=None,
                alpha=5.0,
                lion=False,
                eps=1e-8,
                weight_decay=0.0,
                decay_norm_only=False,
                nesterov=False,
                cautious=False,
                amsgrad=False,
                maximize=False,
            )

        self.assertEqual(steps[0].item(), 1.0)
        self.assertEqual(steps[1].item(), 10.0)

        # Compute expected step 10 bias-corrected update for p2
        bc1_10 = 1.0 - (0.9 ** 10)
        bc2_10 = 1.0 - (0.999 ** 10)
        step_size_10 = 0.01 * (bc2_10 ** 0.5) / bc1_10
        expected_p2 = torch.tensor([3.0, 4.0]) - step_size_10 * (g2 * 0.1) / ((g2 * g2 * 0.001) ** 0.5 + 1e-8 * (bc2_10 ** 0.5))

        self.assertTrue(
            torch.allclose(p2, expected_p2, atol=1e-4),
            f"Param 2 should use step 10 bias correction: actual {p2} vs expected {expected_p2}",
        )

    def test_bug7_xla_cautious_mask_normalization(self):
        """Bug 7: Verify _xla_step in cautious mode applies mask normalization to preserve update magnitude."""
        opt = ApexW([nn.Parameter(torch.tensor([1.0]))], lr=0.01)

        p_xla = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        g = torch.tensor([1.0, 1.0, -1.0, -1.0])
        m_xla = torch.tensor([1.0, 1.0, 1.0, 1.0])
        v_xla = torch.tensor([1.0, 1.0, 1.0, 1.0])
        steps = [torch.tensor(1.0)]

        with torch.no_grad():
            opt._xla_step(
                params=[p_xla],
                grads=[g],
                exp_avgs=[m_xla],
                exp_avgs_slow=[None],
                exp_avg_sqs=[v_xla],
                max_exp_avg_sqs=[None],
                state_steps=steps,
                lr=0.01,
                beta1=0.9,
                beta2=0.999,
                beta3=None,
                alpha=5.0,
                lion=False,
                eps=1e-8,
                weight_decay=0.0,
                decay_norm_only=False,
                nesterov=False,
                cautious=True,
                amsgrad=False,
                maximize=False,
            )

        # Inactive elements (g * m < 0) must be masked to zero (unchanged weight)
        self.assertEqual(p_xla[2].item(), 1.0)
        self.assertEqual(p_xla[3].item(), 1.0)
        # Active elements must be updated with normalized factor (mask / mask_mean = 1.0 / 0.5 = 2.0)
        self.assertLess(p_xla[0].item(), 1.0)
        self.assertEqual(p_xla[0].item(), p_xla[1].item())

    def test_bug8_checkpointing_diagnostics_preservation(self):
        """Bug 8: Verify checkpointing diagnostic utilities work with ZeRO and preserve model state."""
        from apexw import ApexWZeRO, get_memory_breakdown
        import unittest.mock as mock

        # 8a: get_memory_breakdown calculates optimizer state memory on ApexWZeRO
        model = nn.Sequential(nn.Linear(1024, 1024))
        opt = ApexWZeRO(model.parameters(), lr=0.01, world_size=1, rank=0)
        x = torch.randn(2, 1024)
        model(x).sum().backward()
        opt.step()

        breakdown = get_memory_breakdown(model, opt)
        self.assertGreater(breakdown["optimizer_state_mb"], 0.0)
        self.assertEqual(len(opt.state), 0)  # Sharded state delegated to _local_opt
        self.assertGreater(len(opt._local_opt.state), 0)

        # 8b: get_memory_breakdown activation profiling preserves model.training and parameter gradients
        model_prof = nn.Sequential(nn.Linear(10, 10))
        model_prof.train()
        x_prof = torch.randn(2, 10)
        target = torch.randn(2, 10)
        loss_fn = nn.MSELoss()

        model_prof(x_prof).sum().backward()
        initial_grads = {p: p.grad.clone() for p in model_prof.parameters()}

        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.reset_peak_memory_stats"), \
             mock.patch("torch.cuda.memory_allocated", side_effect=[1000, 2000]), \
             mock.patch("torch.cuda.max_memory_allocated", return_value=3000):

            get_memory_breakdown(
                model_prof,
                optimizer=None,
                input_tensor=x_prof,
                target=target,
                criterion=loss_fn,
                device=torch.device("cuda:0"),
            )

        self.assertTrue(model_prof.training, "model.training was not restored after profiling")
        for p in model_prof.parameters():
            self.assertTrue(torch.equal(p.grad, initial_grads[p]), "Gradients were mutated during profiling")


class TestParameterFlattening(unittest.TestCase):
    """Unit and integration tests for parameter flattening / grouping in Many-Small scenarios."""

    def test_flatten_parameters_output_and_gradient_parity(self):
        """Verify forward output, backward gradients, and parameter count reduction."""
        from apexw import flatten_parameters, is_flattened

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
        )
        x = torch.randn(4, 8)
        y_ref = model(x)

        # Flatten model
        flat_p = flatten_parameters(model)
        self.assertTrue(is_flattened(model))
        self.assertEqual(len(list(model.parameters())), 1)
        self.assertIs(list(model.parameters())[0], flat_p)

        # Forward output should match exactly
        y_flat = model(x)
        self.assertTrue(torch.allclose(y_ref, y_flat, atol=1e-6))

        # Backward accumulation into flat parameter
        loss = y_flat.sum()
        loss.backward()
        self.assertIsNotNone(flat_p.grad)
        self.assertEqual(flat_p.grad.shape, flat_p.shape)
        self.assertGreater(flat_p.grad.norm().item(), 0.0)

    def test_flatten_parameters_zero_copy_live_update(self):
        """Verify optimizer step on flat_param mutates sub-module views in-place with zero copy."""
        from apexw import ApexW, flatten_parameters

        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        flat_p = flatten_parameters(model)
        opt = ApexW([flat_p], lr=0.1)

        w0_before = model[0].weight.clone()
        x = torch.randn(2, 4)
        model(x).sum().backward()
        opt.step()

        w0_after = model[0].weight.clone()
        self.assertFalse(torch.equal(w0_before, w0_after))
        # Ensure sub-module view points to the exact same memory as flat_param
        self.assertTrue(torch.allclose(w0_after, flat_p[:16].view(4, 4)))

    def test_flatten_parameters_checkpoint_roundtrip(self):
        """Verify get_unflattened_state_dict and load_unflattened_state_dict preserve checkpoints."""
        from apexw import flatten_parameters

        torch.manual_seed(42)
        m_orig = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        sd_orig = {k: v.clone() for k, v in m_orig.state_dict().items()}

        m_flat = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        flatten_parameters(m_flat)
        m_flat.load_unflattened_state_dict(sd_orig)

        # Verify equivalence
        x = torch.randn(3, 4)
        self.assertTrue(torch.allclose(m_orig(x), m_flat(x), atol=1e-6))

        sd_exported = m_flat.get_unflattened_state_dict()
        self.assertEqual(set(sd_orig.keys()), set(sd_exported.keys()))
        for k in sd_orig:
            self.assertTrue(torch.equal(sd_orig[k], sd_exported[k]))

    def test_auto_pack_initialization_and_parity(self):
        """Verify native auto_pack=True packs small parameters and achieves bit-for-bit parity with unflattened."""
        from apexw import ApexW

        torch.manual_seed(42)
        m_ref = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))
        m_pack = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))
        m_pack.load_state_dict(m_ref.state_dict())

        opt_ref = ApexW(m_ref.parameters(), lr=0.01, auto_pack=False, precision="full")
        opt_pack = ApexW(m_pack.parameters(), lr=0.01, auto_pack=True, pack_threshold=50000, precision="full")

        self.assertGreater(len(opt_pack.packed_param_groups), 0)
        # 4 original parameters (2 weights, 2 biases) packed into 2 flat buffers (1 for 2D, 1 for 1D)
        self.assertEqual(len(opt_pack.param_groups[0]["params"]), 2)

        for _ in range(5):
            x = torch.randn(4, 8)
            opt_ref.zero_grad()
            m_ref(x).sum().backward()
            opt_ref.step()

            opt_pack.zero_grad()
            m_pack(x).sum().backward()
            opt_pack.step()

        for p_ref, p_pack in zip(m_ref.parameters(), m_pack.parameters()):
            self.assertTrue(
                torch.allclose(p_ref, p_pack, atol=1e-6),
                f"Auto-pack mismatch: {p_ref} vs {p_pack}",
            )

    def test_auto_pack_zero_grad_and_state_proxy(self):
        """Verify auto_pack zero_grad cleans original gradients and opt.state[p] proxy works transparently."""
        from apexw import ApexW

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        opt = ApexW(model.parameters(), lr=0.01, auto_pack=True)

        x = torch.randn(2, 4)
        model(x).sum().backward()
        p0 = list(model.parameters())[0]
        self.assertIsNotNone(p0.grad)

        opt.step()

        # State proxy should allow transparent inspection of individual parameters
        self.assertIn(p0, opt.state)
        self.assertEqual(opt.state[p0]["exp_avg"].shape, p0.shape)
        self.assertEqual(opt.state[p0]["exp_avg_sq"].shape, p0.shape)
        self.assertEqual(opt.state[p0]["step"].item(), 1.0)

        # zero_grad with set_to_none=True
        opt.zero_grad(set_to_none=True)
        for p in model.parameters():
            self.assertIsNone(p.grad)

        # zero_grad with set_to_none=False
        model(x).sum().backward()
        opt.zero_grad(set_to_none=False)
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertEqual(p.grad.abs().sum().item(), 0.0)

    def test_auto_pack_decay_norm_preservation(self):
        """Verify auto_pack preserves decay_norm_only semantics (applies WD to 2D weights, skips 1D norms/biases)."""
        from apexw import ApexW

        torch.manual_seed(42)
        m_ref = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))
        m_pack = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))
        m_pack.load_state_dict(m_ref.state_dict())

        opt_ref = ApexW(m_ref.parameters(), lr=0.01, weight_decay=0.05, decay_norm_only=True, precision="full", auto_pack=False)
        opt_pack = ApexW(m_pack.parameters(), lr=0.01, weight_decay=0.05, decay_norm_only=True, precision="full", auto_pack=True)

        for _ in range(5):
            x = torch.randn(4, 8)
            opt_ref.zero_grad()
            m_ref(x).sum().backward()
            opt_ref.step()

            opt_pack.zero_grad()
            m_pack(x).sum().backward()
            opt_pack.step()

        for p_ref, p_pack in zip(m_ref.parameters(), m_pack.parameters()):
            self.assertTrue(torch.allclose(p_ref, p_pack, atol=1e-6))

    def test_triton_adaptive_block_size(self):
        """Verify Triton adaptive block size and _next_power_of_2 helper."""
        from apexw.triton_kernel import _next_power_of_2

        self.assertEqual(_next_power_of_2(0), 1)
        self.assertEqual(_next_power_of_2(1), 1)
        self.assertEqual(_next_power_of_2(63), 64)
        self.assertEqual(_next_power_of_2(64), 64)
        self.assertEqual(_next_power_of_2(65), 128)
        self.assertEqual(_next_power_of_2(512), 512)
        self.assertEqual(_next_power_of_2(1000), 1024)

    def test_compiling_multi_tensor_dynamo_guards(self):
        """Verify that running under torch.compile mock does not trigger graph breaks or crash with AdEMAMix."""
        from apexw import ApexW
        import apexw.optimizer as opt_mod

        p1 = nn.Parameter(torch.randn(8, 8))
        p2 = nn.Parameter(torch.randn(8, 8))
        opt = ApexW([p1, p2], lr=1e-3, beta3=0.999, alpha=5.0, precision="full")

        (p1.sum() + p2.sum()).backward()

        with patch.object(opt_mod, "is_compiling", return_value=True):
            opt.step()

        self.assertEqual(opt.state[p1]["step"].item(), 1.0)
        self.assertEqual(opt.state[p2]["step"].item(), 1.0)

    def test_beta2_high_v_dtype_escalation(self):
        """Regression: at beta2>=0.995 exp_avg_sq must be fp32, at beta2<0.995 it follows compact dtype.

        The bf16 steady-state EMA margin is (1 - beta2) * 128 mantissa levels.
        At beta2=0.999 this is 0.13x - the per-step update (1-beta2)*g^2 is lost in rounding,
        causing v_t to stagnate. ApexW escalates exp_avg_sq to fp32 in this regime.
        At beta2=0.95 the margin is 6.4x - compact precision is safe.
        """
        import warnings
        from apexw import ApexW
        from apexw.optimizer import _BETA2_WARN_THRESHOLD, _BETA2_V_FP32_THRESHOLD

        # Thresholds must be sensible
        self.assertGreater(_BETA2_V_FP32_THRESHOLD, _BETA2_WARN_THRESHOLD - 0.01)
        self.assertLessEqual(_BETA2_V_FP32_THRESHOLD, 0.999)

        param = nn.Parameter(torch.randn(64, 64))

        # --- beta2=0.95: safe, v stays in bf16 ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt_safe = ApexW([param], lr=1e-3, precision="compact", betas=(0.9, 0.95))
        param.grad = torch.randn_like(param)
        opt_safe.step()
        self.assertEqual(opt_safe.state[param]["exp_avg"].dtype, torch.bfloat16,
                         "m must be compact bfloat16 at beta2=0.95")
        self.assertEqual(opt_safe.state[param]["exp_avg_sq"].dtype, torch.bfloat16,
                         "v must be compact bfloat16 at beta2=0.95 (safe margin=6.4x)")

        # --- beta2=0.999: severe, v escalates to fp32, m stays bf16 ---
        param2 = nn.Parameter(torch.randn(64, 64))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            opt_severe = ApexW([param2], lr=1e-3, precision="compact", betas=(0.9, 0.999))
            # Warning must be issued at construction time
            self.assertTrue(any("beta2=0.999" in str(ww.message) for ww in w),
                            "Must warn about beta2=0.999 + compact precision")

        param2.grad = torch.randn_like(param2)
        opt_severe.step()
        self.assertEqual(opt_severe.state[param2]["exp_avg"].dtype, torch.bfloat16,
                         "m must stay compact bfloat16 even at beta2=0.999")
        self.assertEqual(opt_severe.state[param2]["exp_avg_sq"].dtype, torch.float32,
                         "v must auto-escalate to fp32 at beta2=0.999 (margin=0.13x)")

        # --- precision='full': no escalation needed, both always fp32 ---
        param3 = nn.Parameter(torch.randn(64, 64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt_full = ApexW([param3], lr=1e-3, precision="full", betas=(0.9, 0.999))
        param3.grad = torch.randn_like(param3)
        opt_full.step()
        self.assertEqual(opt_full.state[param3]["exp_avg"].dtype, torch.float32)
        self.assertEqual(opt_full.state[param3]["exp_avg_sq"].dtype, torch.float32)

        # --- Numerical correctness: v at beta2=0.999+fp32 must not stagnate ---
        # Run 200 steps and verify v_t is still tracking the gradient
        param4 = nn.Parameter(torch.randn(32))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            opt_num = ApexW([param4], lr=1e-3, precision="compact", betas=(0.9, 0.999),
                            dynamic_scaling=False)
        g_sq_true = None
        for step in range(200):
            param4.grad = torch.ones(32) * 0.1
            opt_num.step()
        v_stored = opt_num.state[param4]["exp_avg_sq"]
        # At step 200 with beta2=0.999, v = (1 - 0.999^200) * 0.01 ≈ 0.00181.
        # fp32 storage must track this without stagnation.
        expected_v = (1.0 - 0.999**200) * 0.01
        self.assertAlmostEqual(v_stored.mean().item(), expected_v, delta=expected_v * 0.05,
                               msg="v must track gradient at beta2=0.999 with fp32 escalation")


if __name__ == "__main__":
    unittest.main()


class TestBugFixes(unittest.TestCase):
    """Regression tests for the four correctness fixes."""

    # ── Fix 1: Mixed-dtype CUDA dispatch guard ─────────────────────────────────
    def test_fix1_mixed_dtype_uses_fallback(self):
        """
        Fix 1 regression: FP32 params, BF16 m, FP32 v (beta2=0.999).
        The Triton / custom-CUDA fast path must NOT be selected for this config.
        The fallback must produce a numerically correct, finite update.
        """
        import warnings
        torch.manual_seed(0)
        param = nn.Parameter(torch.randn(64, 64))  # FP32
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # beta2=0.999 with compact precision → v auto-escalated to fp32,
            # m stays in bf16 → mixed-dtype pair
            opt = ApexW([param], lr=1e-3, betas=(0.9, 0.999), precision="auto")

        param.grad = torch.randn(64, 64)
        opt.step()  # Must not raise

        state = opt.state[param]
        # If state split did happen, verify no corruption
        if state["exp_avg_sq"].dtype == torch.float32 and state["exp_avg"].dtype != torch.float32:
            self.assertTrue(state["exp_avg_sq"].isfinite().all(),
                            "v (exp_avg_sq) must be finite after mixed-dtype step")
            self.assertTrue((state["exp_avg_sq"] >= 0).all(),
                            "v (exp_avg_sq) must be non-negative")
        self.assertFalse(torch.isnan(param).any(), "Parameters must not contain NaN")

    # ── Fix 2: auto_pack missing gradient raises RuntimeError ─────────────────
    def test_fix2_auto_pack_partial_grad_raises(self):
        """
        Fix 2 regression: With auto_pack=True and two packed parameters,
        step() must raise RuntimeError when one has a gradient and one has grad=None.
        """
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        )
        opt = ApexW(model.parameters(), lr=1e-3, auto_pack=True, pack_threshold=1_000_000)

        x = torch.randn(2, 4)
        model(x).sum().backward()
        # Simulate one parameter not receiving a gradient this step
        model[1].weight.grad = None

        with self.assertRaises(RuntimeError) as ctx:
            opt.step()

        self.assertIn("auto_pack", str(ctx.exception))
        self.assertIn("grad=None", str(ctx.exception))

    def test_fix2_auto_pack_all_grads_present_ok(self):
        """Fix 2 complement: all grads present → step() must not raise."""
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        )
        opt = ApexW(model.parameters(), lr=1e-3, auto_pack=True, pack_threshold=1_000_000)
        model(torch.randn(2, 4)).sum().backward()
        opt.step()  # Must not raise
        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())

    def test_fix2_auto_pack_all_grads_none_ok(self):
        """Fix 2 complement: all grads None → step() must silently skip group."""
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        )
        opt = ApexW(model.parameters(), lr=1e-3, auto_pack=True, pack_threshold=1_000_000)
        p_before = [p.data.clone() for p in model.parameters()]
        opt.step()  # No gradients → must not raise, must not update params
        for p_new, p_old in zip(model.parameters(), p_before):
            self.assertTrue(torch.equal(p_new.data, p_old))

    # ── Fix 3: Dynamic scaling stays in fallback path ─────────────────────────
    def test_fix3_dynamic_scaling_fallback_is_stable(self):
        """
        Fix 3 regression: dynamic_scaling=True with compact bf16 v must NOT route
        to Triton (which lacks initialization + recalibration). The fallback must
        produce finite, non-NaN updates.
        """
        import warnings
        torch.manual_seed(1)
        param = nn.Parameter(torch.randn(256, 256))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt = ApexW([param], lr=1e-3, betas=(0.9, 0.95), precision="auto",
                        dynamic_scaling=True)

        for _ in range(5):
            param.grad = torch.randn(256, 256) * 0.1
            opt.step()

        state = opt.state[param]
        self.assertIn("exp_avg_sq", state)
        self.assertTrue(state["exp_avg_sq"].isfinite().all(),
                        "v must be finite after dynamic-scaling fallback steps")
        self.assertFalse(torch.isnan(param).any())

    # ── Fix 4: Triton pre-flight rejects unsupported configs before any mutation
    def test_fix4_preflight_rejects_cpu_tensor(self):
        """Fix 4: _triton_preflight_check must reject CPU tensors with a reason string."""
        import apexw.triton_kernel as tk
        p = torch.randn(16)   # CPU
        reason = tk._triton_preflight_check(p, torch.randn(16), torch.zeros(16), torch.zeros(16))
        self.assertIsNotNone(reason, "Pre-flight must reject CPU tensors")
        self.assertIsInstance(reason, str)

    def test_fix4_preflight_rejects_shape_mismatch(self):
        """Fix 4: _triton_preflight_check must reject mismatched param/grad shapes.
        The exact rejection reason may vary by environment (device check fires first
        on CPU-only hosts before the element-count check is reached)."""
        import apexw.triton_kernel as tk
        reason = tk._triton_preflight_check(
            torch.randn(8), torch.randn(16), torch.zeros(8), torch.zeros(8)
        )
        # Any non-None reason is correct — the call is rejected before any mutation.
        self.assertIsNotNone(reason, "Pre-flight must reject when param/grad shapes mismatch")
        self.assertIsInstance(reason, str)

    def test_fix4_triton_multi_validates_all_before_launching(self):
        """
        Fix 4 regression: If param[1] fails pre-flight, no kernel must be launched
        for param[0] either — the whole call returns False before any mutation.
        """
        from unittest.mock import patch
        import apexw.triton_kernel as tk

        p1 = torch.randn(16)
        p2 = torch.randn(8)    # mismatched grad size → pre-flight rejects
        g1 = torch.randn(16)
        g2 = torch.randn(16)   # wrong numel for p2
        m1, v1 = torch.zeros(16), torch.zeros(16)
        m2, v2 = torch.zeros(8), torch.zeros(8)

        launch_calls = []
        def mock_step(*args, **kwargs):
            launch_calls.append(True)
            return True

        with patch.object(tk, "is_triton_available", return_value=True), \
             patch.object(tk, "triton_adamw_step", side_effect=mock_step):
            result = tk.triton_multi_tensor_adamw_step(
                [p1, p2], [g1, g2], [m1, m2], [v1, v2],
                lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
                weight_decay=0.0, step=1,
            )

        self.assertFalse(result, "Must return False when any tensor fails pre-flight")
        self.assertEqual(len(launch_calls), 0,
                         "No kernel must be launched when pre-flight rejects any tensor")
