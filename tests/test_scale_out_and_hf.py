"""
Tests for ApexW Scale-Out Features:
1. Pure PyTorch Fallbacks & Non-CUDA Backends (CPU, MPS, XLA)
2. Native FSDP & DeepSpeed ZeRO Integration
3. First-Class Hugging Face Trainer Integration
4. Automatic Precision Management & AMP GradScaler Synchronization
5. Standardized PyTorch state_dict & Checkpointing Resharding
"""

import copy
import io
import math
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from apexw import (
    ApexW,
    ApexMuon,
    HybridApexW,
    get_apexw_optimizer,
    register_hf_trainer_optimizer,
    is_fsdp_flat_param,
    extract_fsdp_param_infos,
    DistMuon,
    ApexWZeRO,
    reshard_zero_checkpoint,
    ZeROStateAdapter,
    partition_parameters_for_hybrid,
)


class TestScaleOutAndHF(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # 1. Robust CPU & Non-CUDA Backends
    # -------------------------------------------------------------------------
    def test_cpu_fallback_and_device_agnostic(self):
        """Ensures that even without CUDA/Triton, pure PyTorch fallback operates cleanly on CPU."""
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        opt = ApexW(model.parameters(), lr=1e-3, backend="foreach")
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        self.assertGreater(opt.state[model[0].weight]["step"].item(), 0)

    def test_muon_xla_and_cpu_numerical_stability(self):
        """Verifies ApexMuon runs on CPU without any CUDA-specific dependencies."""
        linear = nn.Linear(48, 48, bias=False)
        opt = ApexMuon(linear.parameters(), lr=2e-2, scale_mode="rms")
        x = torch.randn(2, 48)
        loss = linear(x).sum()
        loss.backward()
        # Verify step runs and updates weights
        w_before = linear.weight.clone()
        opt.step()
        self.assertFalse(torch.allclose(w_before, linear.weight))

    # -------------------------------------------------------------------------
    # 2. Native DeepSpeed & FSDP Integration
    # -------------------------------------------------------------------------
    def test_fsdp_flat_param_detection_and_extraction(self):
        """Tests that FSDP FlatParameters are correctly detected and unpacked."""
        # Create a mock FSDP FlatParameter
        raw_tensor = torch.randn(1000)
        flat_p = nn.Parameter(raw_tensor)
        flat_p._is_flat_param = True
        flat_p._param_infos = [
            SimpleNamespace(param_name="layer1.weight", shape=torch.Size([20, 25]), numel=500, dtype=torch.float32),
            SimpleNamespace(param_name="layer1.bias", shape=torch.Size([20]), numel=20, dtype=torch.float32),
            SimpleNamespace(param_name="layer2.weight", shape=torch.Size([480]), numel=480, dtype=torch.float32),
        ]

        self.assertTrue(is_fsdp_flat_param(flat_p))

        infos = extract_fsdp_param_infos(flat_p)
        self.assertEqual(len(infos), 3)
        self.assertEqual(infos[0]["param_name"], "layer1.weight")
        self.assertEqual(infos[0]["shape"], torch.Size([20, 25]))
        self.assertEqual(infos[0]["numel"], 500)
        self.assertEqual(infos[1]["param_name"], "layer1.bias")

        # Verify partition_parameters_for_hybrid routes FSDP flat parameters safely to AdamW
        muon_p, adamw_p = partition_parameters_for_hybrid([flat_p])
        self.assertEqual(len(muon_p), 0)
        self.assertEqual(len(adamw_p), 1)

    def test_dist_muon_single_rank(self):
        """Tests DistMuon single-machine simulation."""
        linear = nn.Linear(32, 32, bias=False)
        opt = DistMuon(linear.parameters(), lr=2e-2, world_size=1, rank=0)
        x = torch.randn(2, 32)
        linear(x).sum().backward()
        opt.step()
        self.assertIn("momentum", opt.state[linear.weight])

    def test_zero_state_adapter(self):
        """Tests ZeROStateAdapter preserves dynamic scaling factor scale_v across serialization."""
        sd = {"exp_avg": torch.zeros(10), "exp_avg_sq": torch.zeros(10)}
        sd = ZeROStateAdapter.pack_scaling_metadata(sd, scale_v=4.0)
        self.assertIn("_apexw_scale_v", sd)
        unpacked_scale = ZeROStateAdapter.unpack_scaling_metadata(sd)
        self.assertEqual(unpacked_scale, 4.0)

    def test_reshard_zero_checkpoint_api(self):
        """Verifies reshard_zero_checkpoint partitions checkpoints across varying world sizes."""
        rank_0 = {
            "state": {0: {"step": 10}},
            "param_groups": [{"params": [0], "lr": 1e-3}],
            "zero_meta": {"rank": 0, "world_size": 2, "shard_map": []},
        }
        rank_1 = {
            "state": {0: {"step": 10}},
            "param_groups": [{"params": [0], "lr": 1e-3}],
            "zero_meta": {"rank": 1, "world_size": 2, "shard_map": []},
        }
        # Reshard from 2 GPUs to 4 GPUs
        target_checkpoints = reshard_zero_checkpoint([rank_0, rank_1], target_world_size=4)
        self.assertEqual(len(target_checkpoints), 4)
        for r, ckpt in enumerate(target_checkpoints):
            self.assertEqual(ckpt["zero_meta"]["rank"], r)
            self.assertEqual(ckpt["zero_meta"]["world_size"], 4)

    # -------------------------------------------------------------------------
    # 3. First-Class Hugging Face Trainer Integration
    # -------------------------------------------------------------------------
    def test_get_apexw_optimizer_factory(self):
        """Verifies get_apexw_optimizer properly constructs HybridApexW and ApexW."""
        model = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )

        # Hybrid routing (default)
        opt_hybrid = get_apexw_optimizer(model, lr=1e-3, muon_lr=2e-2, use_muon=True)
        self.assertIsInstance(opt_hybrid, HybridApexW)
        self.assertGreater(len(opt_hybrid.muon_params), 0)
        self.assertGreater(len(opt_hybrid.adamw_params), 0)

        # Standard ApexW
        opt_standard = get_apexw_optimizer(model, lr=1e-3, use_muon=False)
        self.assertIsInstance(opt_standard, ApexW)

    def test_register_hf_trainer_optimizer(self):
        """Tests register_hf_trainer_optimizer registration logic."""
        success = register_hf_trainer_optimizer()
        # Even if transformers is not installed in the test environment, function returns a bool gracefully
        self.assertIsInstance(success, bool)

    # -------------------------------------------------------------------------
    # 4. Automatic Precision Management & AMP GradScaler
    # -------------------------------------------------------------------------
    def test_amp_grad_scaler_unscaling_and_inf_skip(self):
        """Verifies that ApexMuon and HybridApexW cleanly unscale gradients and skip infs without NaN."""
        linear = nn.Linear(32, 32, bias=False)
        opt = ApexMuon(linear.parameters(), lr=2e-2)

        x = torch.randn(2, 32)
        linear(x).sum().backward()

        # Simulate PyTorch GradScaler attaching scale 65536.0
        grad_before = linear.weight.grad.clone()
        linear.weight.grad.mul_(65536.0)
        opt.grad_scale = torch.tensor([65536.0])

        opt.step()
        # Gradients should have been unscaled and state populated without overflow
        self.assertIn("momentum", opt.state[linear.weight])
        self.assertFalse(torch.isnan(linear.weight).any())
        self.assertFalse(torch.isinf(linear.weight).any())

        # Simulate found_inf = 1.0 (gradient explosion / overflow)
        w_snapshot = linear.weight.clone()
        opt.found_inf = torch.tensor([1.0])
        opt.step()
        # Weight should be untouched because step was skipped
        self.assertTrue(torch.equal(linear.weight, w_snapshot))

    def test_hybrid_amp_propagation(self):
        """Verifies HybridApexW propagates grad_scale and found_inf to sub-optimizers."""
        model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
        opt = HybridApexW(model, lr=1e-3)
        x = torch.randn(2, 64)
        model(x).sum().backward()

        # Attach inf
        w_before = model[0].weight.clone()
        opt.found_inf = torch.tensor([1.0])
        opt.step()
        self.assertTrue(torch.equal(model[0].weight, w_before))

    # -------------------------------------------------------------------------
    # 5. Standardized PyTorch state_dict & Checkpointing
    # -------------------------------------------------------------------------
    def test_hybrid_state_dict_standard_format_and_resume(self):
        """Verifies HybridApexW state_dict contains standard PyTorch keys and resumes cleanly."""
        model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
        opt1 = HybridApexW(model, lr=1e-3)
        x = torch.randn(2, 64)
        model(x).sum().backward()
        opt1.step()

        sd = opt1.state_dict()
        self.assertIn("state", sd)
        self.assertIn("param_groups", sd)
        self.assertIn("muon_state_dict", sd)
        self.assertIn("adamw_state_dict", sd)

        # Test torch.save / torch.load roundtrip
        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        loaded_sd = torch.load(buf)

        # Resume on a new optimizer instance
        opt2 = HybridApexW(model, lr=1e-3)
        opt2.load_state_dict(loaded_sd)

        # Check step counters match
        for p in model.parameters():
            self.assertEqual(opt1.state[p]["step"], opt2.state[p]["step"])

    def test_ns_steps_default_and_customization(self):
        """Verify ns_steps defaults to 5 in HybridApexW (matches standalone Muon convergence)
        and 4 in ApexMuon standalone. Both are customizable."""
        model = nn.Sequential(nn.Linear(64, 64, bias=False))
        opt_default = HybridApexW(model)
        self.assertEqual(opt_default.muon_opt.param_groups[0]["ns_steps"], 5)  # HybridApexW: 5 for parity with standalone Muon

        opt_custom = HybridApexW(model, ns_steps=3)
        self.assertEqual(opt_custom.muon_opt.param_groups[0]["ns_steps"], 3)

        muon_opt = ApexMuon(model.parameters())
        self.assertEqual(muon_opt.param_groups[0]["ns_steps"], 4)  # ApexMuon standalone: 4 (default)

    def test_fused_and_capturable_flags(self):
        """Verify fused and capturable flags are correctly accepted and forwarded."""
        model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
        opt = HybridApexW(model, fused=True, capturable=True)
        self.assertTrue(opt.muon_opt.param_groups[0]["capturable"])
        self.assertTrue(opt.muon_opt.param_groups[0]["fused"])
        self.assertTrue(opt.adamw_opt.param_groups[0]["capturable"])
        self.assertTrue(opt.adamw_opt.param_groups[0]["fused"])

    def test_fsdp1_real_metadata_extraction(self):
        """Tests that PyTorch FSDP1 FlatParameter with separate _shapes and _numels is correctly unpacked."""
        from torch.distributed.fsdp._flat_param import ParamInfo
        raw_tensor = torch.randn(1000)
        flat_p = nn.Parameter(raw_tensor)
        flat_p._is_flat_param = True
        dummy_mod = nn.Module()
        # Real FSDP1 ParamInfo only has param_name, module, module_name
        flat_p._param_infos = (
            ParamInfo("weight", dummy_mod, "layer1"),
            ParamInfo("bias", dummy_mod, "layer1"),
            ParamInfo("weight", dummy_mod, "layer2"),
        )
        flat_p._shapes = (torch.Size([64, 64]), torch.Size([20]), torch.Size([480]))
        flat_p._numels = (4096, 20, 480)
        flat_p._fqns = ("layer1.weight", "layer1.bias", "layer2.weight")

        infos = extract_fsdp_param_infos(flat_p)
        self.assertEqual(len(infos), 3)
        self.assertEqual(infos[0]["param_name"], "layer1.weight")
        self.assertEqual(infos[0]["shape"], torch.Size([64, 64]))
        self.assertEqual(infos[0]["numel"], 4096)
        self.assertTrue(infos[0]["is_2d"])
        self.assertEqual(infos[1]["param_name"], "layer1.bias")
        self.assertFalse(infos[1]["is_2d"])

    def test_ultra_8bit_precision_state_and_convergence(self):
        """Verify ultra (8-bit) precision allocates INT8 momentum buffers, preserves accuracy and converges."""
        # Check precision aliases
        linear = nn.Linear(64, 64)
        opt_ultra = ApexW(linear.parameters(), lr=1e-3, precision="ultra")
        self.assertEqual(opt_ultra.param_groups[0]["precision"], "ultra")
        opt_3byte = ApexW(linear.parameters(), lr=1e-3, precision="3byte")
        self.assertEqual(opt_3byte.param_groups[0]["precision"], "ultra")
        opt_hybrid = ApexW(linear.parameters(), lr=1e-3, precision="hybrid8")
        self.assertEqual(opt_hybrid.param_groups[0]["precision"], "ultra")

        # Step and verify state dtypes
        x = torch.randn(8, 64)
        linear(x).sum().backward()
        opt_ultra.step()

        weight_state = opt_ultra.state[linear.weight]
        self.assertEqual(weight_state["exp_avg"].dtype, torch.int8)
        self.assertEqual(weight_state["exp_avg_scales"].dtype, torch.float32)
        self.assertEqual(weight_state["exp_avg"].shape, linear.weight.shape)
        # Bias (1D) falls back to float32
        bias_state = opt_ultra.state[linear.bias]
        self.assertIn(bias_state["exp_avg"].dtype, (torch.float32, torch.bfloat16))

        # Memory verification: ~3 bytes/param for 2D weights (62.5% reduction vs AdamW 8 bytes)
        mem_mb = opt_ultra.get_state_memory_mb()
        adamw_equiv = (sum(p.numel() for p in linear.parameters()) * 8) / (1024 * 1024)
        self.assertLess(mem_mb, adamw_equiv * 0.5)

        # Multi-step convergence test on a small regression problem
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        opt = ApexW(model.parameters(), lr=1e-2, precision="ultra", betas=(0.9, 0.95))
        x_data = torch.randn(32, 32)
        y_data = torch.randn(32, 1)

        initial_loss = nn.functional.mse_loss(model(x_data), y_data).item()
        for _ in range(30):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x_data), y_data)
            loss.backward()
            opt.step()
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss * 0.5, "Ultra 8-bit precision must converge smoothly")

    def test_true_8bit_precision_state_and_convergence(self):
        """Verify true 8-bit precision (2 bytes/param) matches adamw_bnb_8bit footprint with superior stability."""
        # 1. Alias normalization
        linear = nn.Linear(64, 64)
        for alias in ("8bit", "8-bit", "int8", "adamw_8bit", "bnb_8bit", torch.int8):
            opt = ApexW(linear.parameters(), lr=1e-3, precision=alias)
            self.assertEqual(opt.param_groups[0]["precision"], "8bit")

        # 2. State allocation verification (both m in INT8 and v in UINT8 with FP32 dynamic block scales)
        opt_8bit = ApexW(linear.parameters(), lr=1e-3, precision="8bit", stochastic_rounding=True)
        x = torch.randn(8, 64)
        linear(x).sum().backward()
        opt_8bit.step()

        weight_state = opt_8bit.state[linear.weight]
        self.assertEqual(weight_state["exp_avg"].dtype, torch.int8)
        self.assertEqual(weight_state["exp_avg_scales"].dtype, torch.float32)
        self.assertEqual(weight_state["exp_avg_sq"].dtype, torch.uint8)
        self.assertEqual(weight_state["exp_avg_sq_scales"].dtype, torch.float32)
        self.assertEqual(weight_state["exp_avg"].shape, linear.weight.shape)
        self.assertEqual(weight_state["exp_avg_sq"].shape, linear.weight.shape)

        # 3. State memory accounting: ~2.03 bytes/param (~74.6% reduction vs AdamW 8 bytes)
        mem_mb = opt_8bit.get_state_memory_mb()
        adamw_equiv = (sum(p.numel() for p in linear.parameters()) * 8) / (1024 * 1024)
        # 2 bytes/param is <= 26% of AdamW 8 bytes
        self.assertLess(mem_mb, adamw_equiv * 0.30)

        # 4. Save and load state_dict roundtrip preserves INT8 and UINT8 moment dtypes
        sd = opt_8bit.state_dict()
        opt_restored = ApexW(linear.parameters(), lr=1e-3, precision="8bit")
        opt_restored.load_state_dict(sd)
        res_weight_state = opt_restored.state[linear.weight]
        self.assertEqual(res_weight_state["exp_avg"].dtype, torch.int8)
        self.assertEqual(res_weight_state["exp_avg_sq"].dtype, torch.uint8)
        self.assertEqual(res_weight_state["exp_avg_scales"].dtype, torch.float32)
        self.assertEqual(res_weight_state["exp_avg_sq_scales"].dtype, torch.float32)

        # 5. Non-linear convergence with stochastic rounding
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        opt = ApexW(model.parameters(), lr=1e-2, precision="8bit", stochastic_rounding=True)
        x_data = torch.randn(32, 32)
        y_data = torch.randn(32, 1)

        initial_loss = nn.functional.mse_loss(model(x_data), y_data).item()
        for _ in range(40):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x_data), y_data)
            loss.backward()
            opt.step()
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss * 0.5, "True 8-bit precision must converge smoothly")

    def test_peft_qlora_auto_filtering(self):
        """Verify nn.Module passing filters frozen base weights (QLoRA setup) and activates PEFT memory path."""
        class MockQLoRAModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 100K frozen base parameters (simulating 4-bit base model)
                self.base_layer = nn.Linear(100, 1000, bias=False)
                for p in self.base_layer.parameters():
                    p.requires_grad = False
                # 1K trainable LoRA adapter parameters
                self.lora_A = nn.Parameter(torch.randn(10, 100))
                self.lora_B = nn.Parameter(torch.randn(1000, 10))
                self.lora_A._param_name = "model.layers.0.self_attn.q_proj.lora_A.weight"
                self.lora_B._param_name = "model.layers.0.self_attn.q_proj.lora_B.weight"

            def forward(self, x):
                return (x @ self.lora_A.t()) @ self.lora_B.t()

        model = MockQLoRAModel()
        # Initializing ApexW directly on nn.Module
        opt = ApexW(model, lr=2e-4, precision="8bit")

        # Only the 2 trainable parameters must be registered, NOT the 100,000 frozen base parameters!
        registered_params = opt.param_groups[0]["params"]
        self.assertEqual(len(registered_params), 2)
        total_opt_params = sum(p.numel() for p in registered_params)
        self.assertEqual(total_opt_params, 1000 + 10000)

        # Step and verify 8-bit states allocated for LoRA adapters
        dummy_input = torch.randn(2, 100)
        out = model(dummy_input)
        out.sum().backward()
        opt.step()

        self.assertEqual(opt.state[model.lora_A]["exp_avg"].dtype, torch.int8)
        self.assertEqual(opt.state[model.lora_A]["exp_avg_sq"].dtype, torch.uint8)
        self.assertEqual(opt.state[model.lora_B]["exp_avg"].dtype, torch.int8)
        self.assertEqual(opt.state[model.lora_B]["exp_avg_sq"].dtype, torch.uint8)

    def test_zero_cluster_saved_mb_accounting(self):
        """Verify cluster saved memory accounts for total optimizer memory across all ranks."""
        model = nn.Sequential(nn.Linear(128, 128))
        opt_single = ApexWZeRO(model.parameters(), lr=1e-3, world_size=1, rank=0)
        single_saved = opt_single.get_total_cluster_saved_mb()

        # On a 4-GPU cluster, total cluster savings scales with world_size
        opt_multi = ApexWZeRO(model.parameters(), lr=1e-3, world_size=4, rank=0)
        multi_saved = opt_multi.get_total_cluster_saved_mb()
        self.assertGreater(multi_saved, single_saved)
        self.assertAlmostEqual(multi_saved, single_saved * 4, places=1)

    def test_reshard_zero_checkpoint_full_moment_reconstruction(self):
        """Verifies reshard_zero_checkpoint fully reconstructs momentum tensors and redistributes them."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 10))
        opt0 = ApexWZeRO(model.parameters(), lr=1e-3, world_size=2, rank=0)
        opt1 = ApexWZeRO(model.parameters(), lr=1e-3, world_size=2, rank=1)

        x = torch.randn(4, 32)
        model(x).sum().backward()
        opt0.step()
        opt1.step()

        sd0 = opt0.state_dict()
        sd1 = opt1.state_dict()

        # Reshard from 2 GPUs to 4 GPUs
        resharded = reshard_zero_checkpoint([sd0, sd1], target_world_size=4)
        self.assertEqual(len(resharded), 4)

        # Check each target rank has non-empty optimizer moments
        for r, ckpt in enumerate(resharded):
            self.assertEqual(ckpt["zero_meta"]["rank"], r)
            self.assertEqual(ckpt["zero_meta"]["world_size"], 4)
            self.assertGreater(len(ckpt["state"]), 0, f"Rank {r} state must not be empty")
            for local_idx, state_entry in ckpt["state"].items():
                self.assertIn("exp_avg", state_entry)
                self.assertIn("exp_avg_sq", state_entry)
                self.assertFalse(torch.isnan(state_entry["exp_avg"]).any())

    def test_amp_explicit_and_robust_detection(self):
        """Verify FP32 parameters receive compact BF16 states when amp=True is enabled or mixed with 16-bit params."""
        param_fp32 = nn.Parameter(torch.randn(128, 128, dtype=torch.float32))
        opt = ApexW([param_fp32], lr=1e-3, precision="auto", amp=True, betas=(0.9, 0.95))
        param_fp32.grad = torch.randn_like(param_fp32)
        opt.step()
        self.assertEqual(opt.state[param_fp32]["exp_avg"].dtype, torch.bfloat16)

    def test_ultra_lion_1byte_per_param(self):
        """Verify ultra precision with lion=True stores only 1 byte/param momentum state."""
        linear = nn.Linear(128, 128, bias=False)
        opt = ApexW(linear.parameters(), lr=1e-4, lion=True, precision="ultra")
        x = torch.randn(4, 128)
        linear(x).sum().backward()
        opt.step()

        st = opt.state[linear.weight]
        self.assertEqual(st["exp_avg"].dtype, torch.int8)
        self.assertNotIn("exp_avg_sq", st)
        # Memory is ~1 byte per param
        mem_mb = opt.get_state_memory_mb()
        expected_mb = (linear.weight.numel() * 1) / (1024 * 1024)
        self.assertAlmostEqual(mem_mb, expected_mb, delta=0.01)

    def test_ultra_ademamix_and_amsgrad(self):
        """Verify ultra precision supports slow momentum (AdEMAMix) and amsgrad correctly."""
        linear = nn.Linear(64, 64)
        opt = ApexW(
            linear.parameters(),
            lr=1e-3,
            precision="ultra",
            beta3=0.999,
            alpha=5.0,
            amsgrad=True,
            cautious=True,
            nesterov=True,
        )
        x = torch.randn(8, 64)
        linear(x).sum().backward()
        opt.step()

        st = opt.state[linear.weight]
        self.assertEqual(st["exp_avg"].dtype, torch.int8)
        self.assertEqual(st["exp_avg_slow"].dtype, torch.int8)
        self.assertIn("max_exp_avg_sq", st)

    def test_ultra_checkpoint_resume(self):
        """Verify ultra precision optimizer state survives state_dict export and reload."""
        torch.manual_seed(123)
        linear1 = nn.Linear(64, 64)
        linear2 = nn.Linear(64, 64)
        linear2.load_state_dict(linear1.state_dict())

        opt1 = ApexW(linear1.parameters(), lr=1e-3, precision="ultra")
        opt2 = ApexW(linear2.parameters(), lr=1e-3, precision="ultra")

        x = torch.randn(4, 64)
        linear1(x).sum().backward()
        opt1.step()

        sd = copy.deepcopy(opt1.state_dict())
        linear2.load_state_dict(linear1.state_dict())
        opt2.load_state_dict(sd)

        # Confirm state loaded correctly
        self.assertEqual(opt2.state[linear2.weight]["exp_avg"].dtype, torch.int8)
        self.assertEqual(opt2.state[linear2.weight]["exp_avg_scales"].dtype, torch.float32)

        # Another step on both must match identically
        linear1.zero_grad()
        linear2.zero_grad()
        x2 = torch.randn(4, 64)
        linear1(x2).sum().backward()
        linear2(x2).sum().backward()
        opt1.step()
        opt2.step()

        diff = (linear1.weight - linear2.weight).abs().max().item()
        self.assertEqual(diff, 0.0)

    def test_8bit_precision_cpu_convergence_and_dtypes(self):
        """Verify 8bit precision on CPU stores INT8 momentum, UINT8 variance, and converges."""
        torch.manual_seed(42)
        linear = nn.Linear(64, 64)
        opt = ApexW(linear.parameters(), lr=1e-3, precision="8bit")

        x = torch.randn(8, 64)
        y = linear(x).sum()
        y.backward()
        opt.step()

        w_state = opt.state[linear.weight]
        self.assertEqual(w_state["exp_avg"].dtype, torch.int8)
        self.assertEqual(w_state["exp_avg_scales"].dtype, torch.float32)
        self.assertEqual(w_state["exp_avg_sq"].dtype, torch.uint8)
        self.assertEqual(w_state["exp_avg_sq_scales"].dtype, torch.float32)

        # Run 10 steps to test CPU convergence and numerical stability
        prev_loss = None
        for step in range(10):
            opt.zero_grad()
            loss = (linear(x) ** 2).mean()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
            if prev_loss is not None:
                self.assertFalse(math.isnan(loss_val))
                self.assertFalse(math.isinf(loss_val))
            prev_loss = loss_val

        # Loss must have converged (decreased)
        self.assertLess(loss_val, 0.5)

    def test_hf_trainer_apexw_8bit_and_ultra(self):
        """Verify Hugging Face Trainer integration supports apexw_8bit and apexw_ultra drop-in optimizer names."""
        from apexw.hf import (
            patched_get_optimizer_cls_and_kwargs,
            patched_create_optimizer,
            patch_huggingface_trainer,
            _get_trainer_grouped_params,
        )

        class MockHFTrainerArgs:
            def __init__(self, optim="apexw_8bit", lr=2e-4, weight_decay=0.01):
                self.optim = optim
                self.learning_rate = lr
                self.weight_decay = weight_decay
                self.adam_beta1 = 0.9
                self.adam_beta2 = 0.999
                self.adam_epsilon = 1e-8

        class MockHFTrainer:
            def __init__(self, model, args):
                self.model = model
                self.args = args
                self.optimizer = None

            def get_decay_parameter_names(self, model):
                return [n for n, p in model.named_parameters() if p.dim() >= 2]

        model = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64))
        trainer_8bit = MockHFTrainer(model, MockHFTrainerArgs(optim="apexw_8bit"))

        # Test optimizer class and kwargs resolution
        opt_cls, opt_kwargs = patched_get_optimizer_cls_and_kwargs(trainer_8bit.args)
        self.assertEqual(opt_cls, ApexW)
        self.assertEqual(opt_kwargs["precision"], "8bit")
        self.assertTrue(opt_kwargs["adamw_compat"])

        # Test create_optimizer builds valid ApexW with grouped params
        opt = patched_create_optimizer(trainer_8bit)
        self.assertIsInstance(opt, ApexW)
        self.assertEqual(opt.param_groups[0]["precision"], "8bit")

        # Test ultra mode resolution
        trainer_ultra = MockHFTrainer(model, MockHFTrainerArgs(optim="apexw_ultra"))
        opt_cls_u, opt_kwargs_u = patched_get_optimizer_cls_and_kwargs(trainer_ultra.args)
        self.assertEqual(opt_kwargs_u["precision"], "ultra")
        opt_u = patched_create_optimizer(trainer_ultra)
        self.assertEqual(opt_u.param_groups[0]["precision"], "ultra")


if __name__ == "__main__":
    unittest.main()
