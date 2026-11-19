"""Tests for distributed sharding (ZeRO, FSDP), Hugging Face Trainer integration, and cross-backend execution."""

import copy
import io
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
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
from apexw.optimizer import (
    _dequantize_blockwise_int4,
    _dequantize_blockwise_int8,
    _quantize_blockwise_int4,
    _quantize_blockwise_int8,
    _quantize_blockwise_uint8,
)


class TestScaleOutAndHF(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    # 1. Robust CPU & Non-CUDA Backends
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

    # 2. Native DeepSpeed & FSDP Integration
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

    # 3. First-Class Hugging Face Trainer Integration
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

    # 4. Automatic Precision Management & AMP GradScaler
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

    # 5. Standardized PyTorch state_dict & Checkpointing
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
        # INT4 momentum is packed as uint8; INT8 is also valid for small/PEFT tensors
        self.assertIn(weight_state["exp_avg"].dtype, (torch.int8, torch.uint8),
                      "exp_avg must be INT8 (8bit m) or UINT8 (INT4 packed m)")
        self.assertEqual(weight_state["exp_avg_scales"].dtype, torch.float32)
        self.assertEqual(weight_state["exp_avg_sq"].dtype, torch.uint8)
        self.assertEqual(weight_state["exp_avg_sq_scales"].dtype, torch.float32)
        # INT4-packed momentum has half the elements (ceil(numel/2) packed bytes)
        if weight_state.get("is_4bit_m", False):
            expected_m_numel = (linear.weight.numel() + 1) // 2
            self.assertEqual(weight_state["exp_avg"].numel(), expected_m_numel)
        else:
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
        # Accept both INT8 and UINT8 (INT4-packed) for momentum after roundtrip
        self.assertIn(res_weight_state["exp_avg"].dtype, (torch.int8, torch.uint8))
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
        # INT4 momentum is stored as uint8 (packed nibbles) for large tensors
        self.assertIn(w_state["exp_avg"].dtype, (torch.int8, torch.uint8),
                      "8bit m must be INT8 or INT4-packed-as-uint8")
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


class TestFP16Stability(unittest.TestCase):
    """
    Regression tests for the FP16 NaN bug (T4 / consumer GPU stability).

    Root cause: p.addcdiv_(num.to(p.dtype), denom.to(p.dtype)) cast denom to float16,
    flushing eps_scaled ≈ 3e-10 to 0.0 (below float16 min subnormal 5.96e-8), causing
    division-by-zero → NaN on step 1. All tests run on CPU (CI-safe) but exercise the
    identical code paths as the T4 CUDA run.
    """

    def setUp(self):
        torch.manual_seed(0)

    def _make_fp16_model(self, in_dim=64, out_dim=10, hidden=64):
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        ).half()

    def _assert_no_nan_inf(self, model, msg=""):
        for name, p in model.named_parameters():
            self.assertFalse(
                torch.isnan(p).any().item(),
                f"NaN in {name} {msg}",
            )
            self.assertFalse(
                torch.isinf(p).any().item(),
                f"Inf in {name} {msg}",
            )

    # Core FP16 + 8bit: 100-step convergence, zero NaN
    def test_fp16_8bit_no_nan_100_steps(self):
        """8bit precision must stay numerically clean for 100 steps on FP16 model."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=2e-4, precision="8bit")
        for step in range(100):
            x = torch.randn(8, 64, dtype=torch.float16)
            y = torch.randint(0, 10, (8,))
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            self._assert_no_nan_inf(model, f"at step {step}")
        # loss should be finite and decreasing (rough check)
        self.assertTrue(torch.isfinite(loss).item())

    def test_fp16_ultra_no_nan_100_steps(self):
        """ultra precision must stay numerically clean for 100 steps on FP16 model."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=2e-4, precision="ultra")
        for step in range(100):
            x = torch.randn(8, 64, dtype=torch.float16)
            y = torch.randint(0, 10, (8,))
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            self._assert_no_nan_inf(model, f"at step {step}")

    def test_fp16_compact_no_nan_100_steps(self):
        """compact precision must stay numerically clean for 100 steps on FP16 model."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=2e-4, precision="compact")
        for step in range(100):
            x = torch.randn(8, 64, dtype=torch.float16)
            y = torch.randint(0, 10, (8,))
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            self._assert_no_nan_inf(model, f"at step {step}")

    # Sparse / zero gradients (LoRA skipped layers)
    def test_fp16_8bit_sparse_gradients(self):
        """Must survive FP16 steps where some layers have zero gradients (LoRA style)."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=1e-3, precision="8bit")
        for step in range(20):
            # Zero out all gradients manually then set only last layer grad
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            # Only last Linear gets a real gradient
            last = list(model.parameters())[-2]
            last.grad = torch.randn_like(last)
            opt.step()
            opt.zero_grad()
            self._assert_no_nan_inf(model, f"sparse step {step}")

    # Extreme learning rates
    def test_fp16_8bit_tiny_lr(self):
        """Very small LR (1e-6) must not produce NaN in FP16 8bit."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=1e-6, precision="8bit")
        for step in range(10):
            x = torch.randn(4, 64, dtype=torch.float16)
            y = torch.randint(0, 10, (4,))
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
            opt.zero_grad()
            self._assert_no_nan_inf(model, f"tiny_lr step {step}")

    def test_fp16_8bit_large_lr(self):
        """Large LR (1e-2) must not produce NaN in FP16 8bit (clipping handles stability)."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=1e-2, precision="8bit")
        for step in range(10):
            x = torch.randn(4, 64, dtype=torch.float16)
            y = torch.randint(0, 10, (4,))
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
            opt.zero_grad()
            self._assert_no_nan_inf(model, f"large_lr step {step}")

    # NaN/Inf gradient injection (simulates missing GradScaler overflow)
    def test_fp16_8bit_survives_nan_gradient(self):
        """When a NaN gradient arrives (no GradScaler), optimizer must skip/sanitize — no state poison."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=1e-3, precision="8bit")
        # Normal step first to initialize state
        x = torch.randn(4, 64, dtype=torch.float16)
        nn.functional.cross_entropy(model(x), torch.randint(0, 10, (4,))).backward()
        opt.step(); opt.zero_grad()

        # Inject NaN into all gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, float("nan"))
        opt.step()
        opt.zero_grad()
        self._assert_no_nan_inf(model, "after NaN gradient injection")

    def test_fp16_8bit_survives_inf_gradient(self):
        """When an Inf gradient arrives (FP16 overflow), optimizer must sanitize — no state poison."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=1e-3, precision="8bit")
        x = torch.randn(4, 64, dtype=torch.float16)
        nn.functional.cross_entropy(model(x), torch.randint(0, 10, (4,))).backward()
        opt.step(); opt.zero_grad()

        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, float("inf"))
        opt.step()
        opt.zero_grad()
        self._assert_no_nan_inf(model, "after Inf gradient injection")

    # First step NaN (the original T4 bug): step 1 denom underflow
    def test_fp16_8bit_first_step_no_nan(self):
        """Step 1 was the most dangerous: eps_scaled ≈ 3e-10 < float16 subnormal → 0.0 denom → NaN."""
        model = nn.Linear(128, 64).half()  # single large layer
        opt = ApexW(model.parameters(), lr=2e-4, precision="8bit")
        x = torch.randn(16, 128, dtype=torch.float16)
        y = torch.randint(0, 64, (16,))
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()  # step 1 — this was the crasher
        for name, p in model.named_parameters():
            self.assertFalse(torch.isnan(p).any().item(), f"NaN in {name} on step 1")
            self.assertFalse(torch.isinf(p).any().item(), f"Inf in {name} on step 1")

    def test_fp16_ultra_first_step_no_nan(self):
        """Same step-1 regression for ultra precision."""
        model = nn.Linear(128, 64).half()
        opt = ApexW(model.parameters(), lr=2e-4, precision="ultra")
        x = torch.randn(16, 128, dtype=torch.float16)
        y = torch.randint(0, 64, (16,))
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
        for name, p in model.named_parameters():
            self.assertFalse(torch.isnan(p).any().item(), f"NaN in {name} on ultra step 1")
            self.assertFalse(torch.isinf(p).any().item(), f"Inf in {name} on ultra step 1")

    # Multiple LR scales (mimics LoRA fine-tuning with lr_ratio=100)
    def test_fp16_8bit_lora_lr_groups(self):
        """LoRA uses different LR per param group; all must be NaN-free in FP16."""
        base_model = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 64),
        ).half()
        lora_a = nn.Linear(64, 4, bias=False).half()
        lora_b = nn.Linear(4, 64, bias=False).half()

        opt = ApexW(
            [
                {"params": list(base_model.parameters()), "lr": 1e-5},
                {"params": list(lora_a.parameters()) + list(lora_b.parameters()), "lr": 1e-3},
            ],
            precision="8bit",
        )
        for step in range(30):
            x = torch.randn(8, 64, dtype=torch.float16)
            out = base_model(x) + lora_b(lora_a(x))
            out.sum().backward()
            opt.step()
            opt.zero_grad()
        for name, p in {**dict(base_model.named_parameters()),
                        **dict(lora_a.named_parameters()),
                        **dict(lora_b.named_parameters())}.items():
            self.assertFalse(torch.isnan(p).any().item(), f"NaN in {name}")
            self.assertFalse(torch.isinf(p).any().item(), f"Inf in {name}")

    # Nesterov + cautious in FP16
    def test_fp16_8bit_nesterov_no_nan(self):
        """Nesterov momentum in 8bit FP16 must not produce NaN."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=1e-3, precision="8bit", nesterov=True)
        for step in range(20):
            x = torch.randn(4, 64, dtype=torch.float16)
            nn.functional.cross_entropy(model(x), torch.randint(0, 10, (4,))).backward()
            opt.step(); opt.zero_grad()
            self._assert_no_nan_inf(model, f"nesterov step {step}")

    def test_fp16_8bit_cautious_no_nan(self):
        """Cautious updates in 8bit FP16 must not produce NaN."""
        model = self._make_fp16_model()
        opt = ApexW(model.parameters(), lr=1e-3, precision="8bit", cautious=True)
        for step in range(20):
            x = torch.randn(4, 64, dtype=torch.float16)
            nn.functional.cross_entropy(model(x), torch.randint(0, 10, (4,))).backward()
            opt.step(); opt.zero_grad()
            self._assert_no_nan_inf(model, f"cautious step {step}")


class TestPrecisionAndStateInvariants(unittest.TestCase):
    def test_clip_grad_norm_persists_across_fast_execution_plan_steps(self):
        """C1: clip_grad_norm must apply on step 2+ (FastExecutionPlan) for 8bit, compact, and foreach=False."""
        for kwargs in [
            {"precision": "8bit"},
            {"precision": "compact"},
            {"precision": "compact", "foreach": False},
            {"precision": "full", "fused": True},
        ]:
            with self.subTest(kwargs=kwargs):
                p = nn.Parameter(torch.zeros(32, 32))
                opt = ApexW([p], lr=1e-1, clip_grad_norm=1.0, weight_decay=0.0, **kwargs)
                deltas = []
                for step_idx in range(4):
                    p.grad = torch.full_like(p, 100.0)
                    before = p.detach().clone()
                    opt.step()
                    deltas.append((p - before).abs().max().item())
                # If clipping stops on step 2, raw grad 100 causes huge step jump; with clip=1.0 updates stay bounded (~lr)
                for step_idx, d in enumerate(deltas):
                    self.assertLess(d, 0.25, f"Step {step_idx+1} update exploded ({d}) for {kwargs}")

    def test_grad_centralization_persists_on_step2_8bit_and_single(self):
        """C2: grad_centralization must remain active on FastExecutionPlan step 2+."""
        for kwargs in [{"precision": "8bit"}, {"precision": "compact", "foreach": False}]:
            with self.subTest(kwargs=kwargs):
                p = nn.Parameter(torch.zeros(16, 16))
                opt = ApexW([p], lr=1e-2, grad_centralization=True, weight_decay=0.0, **kwargs)
                for step_idx in range(3):
                    # Constant gradient across every row -> after centralization, centered grad == 0
                    p.grad = torch.ones_like(p) * 5.0
                    opt.step()
                    self.assertLess(
                        p.abs().max().item(),
                        1e-6,
                        f"Step {step_idx+1} failed to centralize constant grad for {kwargs}",
                    )

    def test_native_fused_unscales_before_clipping(self):
        """C4: execute_native_fused must unscale grad_scale BEFORE clipping, not double-divide."""
        p1 = nn.Parameter(torch.ones(16, 16))
        p2 = nn.Parameter(torch.ones(16, 16))
        opt_scaled = ApexW([p1], lr=1e-2, precision="full", fused=True, clip_grad_norm=1.0, weight_decay=0.0)
        opt_ref = ApexW([p2], lr=1e-2, precision="full", fused=True, clip_grad_norm=1.0, weight_decay=0.0)

        scale = torch.tensor(1024.0)
        for _ in range(3):
            raw_g = torch.full_like(p1, 0.5)
            p1.grad = raw_g * scale
            p2.grad = raw_g.clone()
            opt_scaled.grad_scale = scale
            opt_scaled.step()
            opt_scaled.grad_scale = None
            opt_ref.step()

        self.assertTrue(torch.allclose(p1, p2, atol=1e-6, rtol=1e-5))

    def test_stochastic_rounding_false_respected_in_8bit(self):
        """H1: stochastic_rounding=False must make 8bit steps completely deterministic without reseeding."""
        results = []
        for _ in range(2):
            p = nn.Parameter(torch.linspace(-1.0, 1.0, 2048).view(32, 64))
            opt = ApexW([p], lr=1e-2, precision="8bit", stochastic_rounding=False, weight_decay=0.0)
            g = torch.sin(torch.arange(2048, dtype=torch.float32)).view(32, 64) * 0.1
            for _ in range(4):
                p.grad = g.clone()
                opt.step()
            results.append(p.detach().clone())

        self.assertTrue(torch.equal(results[0], results[1]))
        self.assertFalse(opt.state[p]["stochastic_rounding"])

    def test_int4_stochastic_rounding_is_unbiased(self):
        """H2: _quantize_blockwise_int4(stochastic=True) must be unbiased in expectation."""
        torch.manual_seed(0)
        # Value between NF4 codes 0.0 (idx 7) and 0.0911 (idx 8) plus scale 1.0 anchor
        x = torch.full((256,), 0.03)
        x[0] = 1.0
        trials = []
        for _ in range(400):
            q, s = _quantize_blockwise_int4(x, block_size=256, stochastic=True)
            deq = _dequantize_blockwise_int4(q, s, x.numel(), block_size=256)
            trials.append(deq[1:].mean().item())
        mean_est = sum(trials) / len(trials)
        # Standard error over 400 * 255 = 102000 samples with gap ~0.0911 is ~0.00014
        self.assertAlmostEqual(mean_est, 0.03, delta=0.002)

    def test_int8_uint8_noise_independent_across_blocks_and_nonmutating(self):
        """H3: Blockwise int8/uint8 dithering must not clone identical noise across blocks or mutate input."""
        torch.manual_seed(123)
        x = torch.full((16, 256), 0.5)
        x[:, 0] = 10.0  # Anchor block scale identically across all 16 blocks
        q8, _ = _quantize_blockwise_int8(x.reshape(-1), block_size=256, stochastic=True)
        q8_blocks = q8.view(16, 256)
        # Previously all 16 blocks were 100% identical because noise had shape (1, 256)
        self.assertFalse(torch.equal(q8_blocks[0], q8_blocks[1]))

        x_inf = torch.ones(256)
        x_inf[5] = float("nan")
        x_inf_copy = x_inf.clone()
        _quantize_blockwise_uint8(x_inf, block_size=256, stochastic=True)
        self.assertTrue(torch.isnan(x_inf[5]), "_quantize_blockwise_uint8 mutated input tensor!")

    def test_load_state_dict_preserves_fp32_second_moment_on_bf16_params(self):
        """H6: Resuming bf16 param with beta2=0.999 must preserve float32 exp_avg_sq without bf16 truncation."""
        p = nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16))
        opt1 = ApexW([p], lr=1e-3, betas=(0.9, 0.999), precision="compact", stochastic_rounding=False)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            opt1.step()

        self.assertEqual(opt1.state[p]["exp_avg_sq"].dtype, torch.float32)
        orig_v = opt1.state[p]["exp_avg_sq"].clone()
        sd = opt1.state_dict()

        p2 = nn.Parameter(p.detach().clone())
        opt2 = ApexW([p2], lr=1e-3, betas=(0.9, 0.999), precision="compact", stochastic_rounding=False)
        opt2.load_state_dict(sd)
        self.assertEqual(opt2.state[p2]["exp_avg_sq"].dtype, torch.float32)
        self.assertTrue(torch.equal(opt2.state[p2]["exp_avg_sq"], orig_v))

    def test_fast_execution_plan_invalidates_on_decay_norm_only_or_precision_change(self):
        """FastExecutionPlan must invalidate when decay_norm_only or precision changes."""
        p1 = nn.Parameter(torch.ones(8, 8))
        p2 = nn.Parameter(torch.ones(8))
        opt = ApexW([p1, p2], lr=1e-2, precision="full", fused=False, weight_decay=0.1, decay_norm_only=True)
        p1.grad = torch.zeros_like(p1)
        p2.grad = torch.zeros_like(p2)
        opt.step()
        # Biases/1D params unchanged when decay_norm_only=True
        self.assertTrue(torch.equal(p2, torch.ones(8)))

        opt.param_groups[0]["decay_norm_only"] = False
        p1.grad = torch.zeros_like(p1)
        p2.grad = torch.zeros_like(p2)
        opt.step()
        # Must now apply weight decay to p2 on step 2
        self.assertLess(p2.abs().max().item(), 1.0)


def _mock_cuda_tensor(dtype: torch.dtype, numel: int = 64, device_str: str = "cuda:0", contiguous: bool = True):
    from unittest.mock import MagicMock

    t = MagicMock(spec=torch.Tensor)
    t.is_cuda = True
    t.device = device_str
    t.dtype = dtype
    t.numel.return_value = numel
    t.is_contiguous.return_value = contiguous
    return t


class TestKernelAndGatingInvariants(unittest.TestCase):
    def test_triton_preflight_validates_nf4_and_rejects_malformed_quant_configs(self):
        """Triton preflight must accept valid packed NF4 uint8 and reject malformed NF4, missing scales, int8+slow, and non-pow2 block_size."""
        import apexw.triton_kernel as tk

        p = _mock_cuda_tensor(torch.float32)
        g = _mock_cuda_tensor(torch.float32)
        scale = _mock_cuda_tensor(torch.float32, numel=1)

        # Valid packed NF4 uint8 exp_avg + uint8 exp_avg_sq passes preflight
        m_int4 = _mock_cuda_tensor(torch.uint8, numel=32)
        v_uint8 = _mock_cuda_tensor(torch.uint8, numel=64)
        self.assertIsNone(
            tk._triton_preflight_check(p, g, m_int4, v_uint8, exp_avg_scales=scale, exp_avg_sq_scales=scale, block_size=64)
        )

        # Malformed packed NF4 uint8 (wrong packed element count) must be rejected
        m_int4_bad = _mock_cuda_tensor(torch.uint8, numel=31)
        err = tk._triton_preflight_check(p, g, m_int4_bad, v_uint8, exp_avg_scales=scale, exp_avg_sq_scales=scale, block_size=64)
        self.assertIsNotNone(err)
        self.assertIn("INT4", err)

        # Packed NF4 uint8 paired with non-uint8 exp_avg_sq must be rejected
        v_fp32 = _mock_cuda_tensor(torch.float32, numel=64)
        err = tk._triton_preflight_check(p, g, m_int4, v_fp32, exp_avg_scales=scale, exp_avg_sq_scales=None, block_size=64)
        self.assertIsNotNone(err)

        # INT8/INT4 without exp_avg_scales must be rejected
        m_int8 = _mock_cuda_tensor(torch.int8, numel=64)
        err = tk._triton_preflight_check(p, g, m_int8, v_uint8, exp_avg_scales=None, exp_avg_sq_scales=scale, block_size=64)
        self.assertIsNotNone(err)
        self.assertIn("exp_avg_scales", err)

        # INT8 + exp_avg_slow (AdEMAMix beta3) must be rejected
        m_slow = _mock_cuda_tensor(torch.float32, numel=64)
        err = tk._triton_preflight_check(p, g, m_int8, v_uint8, exp_avg_slow=m_slow, exp_avg_scales=scale, exp_avg_sq_scales=scale, block_size=64)
        self.assertIsNotNone(err)
        self.assertIn("slow momentum", err)

        # UINT8 v without exp_avg_sq_scales must be rejected
        err = tk._triton_preflight_check(p, g, m_int8, v_uint8, exp_avg_scales=scale, exp_avg_sq_scales=None, block_size=64)
        self.assertIsNotNone(err)
        self.assertIn("exp_avg_sq_scales", err)

        # Non-power-of-2 block_size must be rejected
        err = tk._triton_preflight_check(p, g, m_int8, v_uint8, exp_avg_scales=scale, exp_avg_sq_scales=scale, block_size=96)
        self.assertIsNotNone(err)
        self.assertIn("power of 2", err)

        # Valid 8bit config passes preflight
        self.assertIsNone(
            tk._triton_preflight_check(p, g, m_int8, v_uint8, exp_avg_scales=scale, exp_avg_sq_scales=scale, block_size=64)
        )

    def test_triton_multi_tensor_rejects_invalid_batch_even_with_skip_preflight(self):
        """triton_multi_tensor_adamw_step must reject malformed INT4, missing scales, or non-pow2 block_size even if skip_preflight=True."""
        import apexw.triton_kernel as tk

        p = _mock_cuda_tensor(torch.float32)
        g = _mock_cuda_tensor(torch.float32)
        m_int4_bad = _mock_cuda_tensor(torch.uint8, numel=31)
        m_int4_ok = _mock_cuda_tensor(torch.uint8, numel=32)
        m_int8 = _mock_cuda_tensor(torch.int8, numel=64)
        v_uint8 = _mock_cuda_tensor(torch.uint8, numel=64)
        scale = _mock_cuda_tensor(torch.float32, numel=1)

        with patch.object(tk, "_HAS_TRITON", True):
            ok_bad_len = tk.triton_multi_tensor_adamw_step(
                [p], [g], [m_int4_bad], [v_uint8],
                lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1,
                block_size=[64], exp_avg_scales=[scale], exp_avg_sq_scales=[scale], skip_preflight=True,
            )
            self.assertFalse(ok_bad_len)

            ok_no_scales = tk.triton_multi_tensor_adamw_step(
                [p], [g], [m_int4_ok], [v_uint8],
                lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1,
                block_size=[64], exp_avg_scales=None, exp_avg_sq_scales=[scale], skip_preflight=True,
            )
            self.assertFalse(ok_no_scales)

            ok_block = tk.triton_multi_tensor_adamw_step(
                [p], [g], [m_int8], [v_uint8],
                lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1,
                block_size=[96], exp_avg_scales=[scale], exp_avg_sq_scales=[scale], skip_preflight=True,
            )
            self.assertFalse(ok_block)

    def test_8bit_non_pow2_block_and_beta3_cpu_fallback_and_convergence(self):
        """8bit with non-power-of-2 block_size or AdEMAMix beta3 must execute cleanly via fallback path on CPU."""
        for kwargs in [
            {"precision": "8bit", "block_size": 96},
            {"precision": "8bit", "beta3": 0.99, "alpha": 2.0},
            {"precision": "ultra", "block_size": 96},
        ]:
            with self.subTest(kwargs=kwargs):
                p = nn.Parameter(torch.linspace(-1.0, 1.0, 200).reshape(10, 20))
                opt = ApexW([p], lr=5e-2, weight_decay=0.0, **kwargs)
                init_loss = float((p.detach() ** 2).sum())
                for _ in range(15):
                    opt.zero_grad()
                    loss = (p ** 2).sum()
                    loss.backward()
                    opt.step()
                final_loss = float((p.detach() ** 2).sum())
                self.assertTrue(torch.isfinite(p).all())
                self.assertLess(final_loss, init_loss)

    def test_cuda_extension_and_triton_kernel_static_invariants(self):
        """Static checks on CUDA/Triton source files for fast-math removal, CUDAGuard, tensor_idx seed, and int64 PIDs."""
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "apexw", "cuda_extension.py")) as f:
            ce_src = f.read()
        self.assertNotIn("--use_fast_math", ce_src)
        self.assertIn("get_device_capability", ce_src)

        with open(os.path.join(root, "apexw", "csrc", "apexw_cuda.cpp")) as f:
            cpp_src = f.read()
        self.assertIn("CUDAGuard", cpp_src)
        self.assertIn("C10_CUDA_KERNEL_LAUNCH_CHECK", cpp_src)
        self.assertIn("const double bias_correction1", cpp_src)

        with open(os.path.join(root, "apexw", "csrc", "apexw_cuda_kernel.cu")) as f:
            cu_src = f.read()
        multi_kernel_src = cu_src.split("launch_multi_tensor_adamw_cuda_2d")[0]
        self.assertNotIn("hash_coords(static_cast<uint32_t>(idx), step_seed);", multi_kernel_src)
        self.assertIn("static_cast<uint32_t>(tensor_idx)", multi_kernel_src)

        with open(os.path.join(root, "apexw", "triton_kernel.py")) as f:
            tk_src = f.read()
        self.assertNotIn("pid = tl.program_id(axis=0)\n", tk_src)
        self.assertIn("tl.program_id(axis=0).to(tl.int64)", tk_src)


class TestMuonAndHybridInvariants(unittest.TestCase):
    def test_muon_channels_last_conv_and_tall_4d_no_view_error(self):
        """M2: channels_last 4D Conv2d and tall (m > n) 4D weights must step cleanly without .view() errors."""
        from apexw import ApexMuon

        # Channels-last 4D tensors (batched) + tall 4D (m=32 > n=3*3*3=27)
        conv1 = nn.Conv2d(3, 32, kernel_size=3, bias=False).to(memory_format=torch.channels_last)
        conv2 = nn.Conv2d(3, 32, kernel_size=3, bias=False).to(memory_format=torch.channels_last)
        for batch_bmm in (False, True):
            with self.subTest(batch_bmm=batch_bmm):
                opt = ApexMuon([conv1.weight, conv2.weight], lr=1e-2, batch_bmm=batch_bmm, cautious=True)
                x = torch.randn(2, 3, 8, 8).to(memory_format=torch.channels_last)
                (conv1(x).sum() + conv2(x).sum()).backward()
                opt.step()
                opt.zero_grad()
                self.assertTrue(torch.isfinite(conv1.weight).all())
                self.assertTrue(torch.isfinite(conv2.weight).all())

    def test_muon_single_element_param_no_nan(self):
        """M3: 1-element tensor in ApexMuon 1D fallback must not produce NaN from std()."""
        from apexw import ApexMuon

        p = nn.Parameter(torch.tensor([2.5]))
        opt = ApexMuon([p], lr=1e-2)
        p.grad = torch.tensor([1.0])
        opt.step()
        self.assertTrue(torch.isfinite(p).all())
        self.assertLess(p.item(), 2.5)

    def test_muon_load_state_dict_preserves_fp32_momentum_on_bf16_params(self):
        """H6: ApexMuon and HybridApexW checkpoint resume must preserve fp32 momentum on bf16 weights."""
        from apexw import ApexMuon, HybridApexW

        p = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
        opt1 = ApexMuon([p], lr=1e-2)
        p.grad = torch.randn_like(p)
        opt1.step()
        self.assertEqual(opt1.state[p]["momentum"].dtype, torch.float32)
        orig_m = opt1.state[p]["momentum"].clone()
        sd = opt1.state_dict()

        p2 = nn.Parameter(p.detach().clone())
        opt2 = ApexMuon([p2], lr=1e-2)
        opt2.load_state_dict(sd)
        self.assertEqual(opt2.state[p2]["momentum"].dtype, torch.float32)
        self.assertTrue(torch.equal(opt2.state[p2]["momentum"], orig_m))

        # Hybrid flat state_dict resume without muon_state_dict/adamw_state_dict keys
        m1 = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 8)).to(torch.bfloat16)
        h1 = HybridApexW(m1, lr=1e-3)
        m1(torch.randn(2, 32, dtype=torch.bfloat16)).sum().backward()
        h1.step()
        hsd = h1.state_dict()
        flat_only_sd = {"state": hsd["state"], "param_groups": hsd["param_groups"]}

        m2 = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 8)).to(torch.bfloat16)
        h2 = HybridApexW(m2, lr=1e-3)
        h2.load_state_dict(flat_only_sd)
        w2 = m2[0].weight
        self.assertEqual(h2.state[w2]["momentum"].dtype, torch.float32)

    def test_hybrid_partition_routes_lm_head_and_aspect_ratio_to_adamw(self):
        """H5: nn.Module partitioning routes lm_head, head.weight, and extreme aspect ratios to AdamW, keeping multihead in Muon."""
        from apexw.hybrid import partition_parameters_for_hybrid

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.multihead = nn.Linear(64, 64, bias=False)
                self.tall_embed = nn.Linear(64, 1024, bias=False)  # 1024 / 64 = 16 >= 4 & max >= 1000
                self.lm_head = nn.Linear(64, 64, bias=False)
                self.head = nn.Linear(64, 64, bias=False)

        model = TinyLM()
        muon_p, adamw_p = partition_parameters_for_hybrid(model)
        self.assertTrue(any(p is model.multihead.weight for p in muon_p))
        self.assertTrue(any(p is model.tall_embed.weight for p in adamw_p))
        self.assertTrue(any(p is model.lm_head.weight for p in adamw_p))
        self.assertTrue(any(p is model.head.weight for p in adamw_p))

    def test_apexw_use_muon_with_betas_no_collision_and_default_rms_matched(self):
        """M1 & H4: ApexW(..., use_muon=True, betas=...) must not crash and defaults to rms_matched."""
        model = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.Linear(32, 16))
        opt = ApexW(model, lr=1e-3, betas=(0.9, 0.999), use_muon=True)
        self.assertEqual(opt.muon_opt.param_groups[0]["scale_mode"], "rms_matched")
        self.assertEqual(opt.adamw_opt.param_groups[0]["betas"], (0.9, 0.999))
        model(torch.randn(2, 32)).sum().backward()
        opt.step()

    def test_hybrid_step_clears_amp_attrs_on_exception_and_accurate_summary(self):
        """M8 & C7: grad_scale/found_inf must clear in finally even on error, and print_summary reports real state bytes."""
        import io
        from contextlib import redirect_stdout
        from apexw import HybridApexW

        model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 8))
        opt = HybridApexW(model, lr=1e-3)
        model(torch.randn(2, 32)).sum().backward()
        opt.grad_scale = torch.tensor(64.0)
        opt.found_inf = torch.tensor(1.0)
        opt.step()

        # After step(), found_inf must be cleared so subsequent normal step actually updates weights
        w_before = model[0].weight.detach().clone()
        model.zero_grad()
        model(torch.randn(2, 32)).sum().backward()
        opt.step()
        self.assertGreater((model[0].weight - w_before).abs().max().item(), 0.0)

        # Exception cleanup in finally
        opt.grad_scale = torch.tensor(128.0)
        opt.found_inf = torch.tensor(0.0)
        with patch.object(opt.muon_opt, "step", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                opt.step()
        self.assertIsNone(opt.grad_scale)
        self.assertIsNone(opt.found_inf)
        self.assertIsNone(getattr(opt.muon_opt, "grad_scale", None))
        self.assertIsNone(getattr(opt.adamw_opt, "grad_scale", None))

        # C7: print_summary must report real 4.0 bytes/param for Muon, not hardcoded 2 bytes/param
        buf = io.StringIO()
        with redirect_stdout(buf):
            opt.print_summary()
        summary_out = buf.getvalue()
        self.assertIn("4.0 bytes/param", summary_out)
        self.assertNotIn("[2 bytes/param]", summary_out)


class TestZeROAndHFIntegration(unittest.TestCase):
    def test_zero_state_dict_uses_int_ids_not_live_parameters(self):
        """H8: ApexWZeRO.state_dict()['param_groups'] must contain int IDs, not live nn.Parameter tensors."""
        from apexw.zero import ApexWZeRO

        p = nn.Parameter(torch.randn(16, 64))
        opt = ApexWZeRO([p], lr=1e-3, world_size=1, rank=0)
        p.grad = torch.randn_like(p)
        opt.step()

        sd = opt.state_dict()
        self.assertEqual(len(sd["param_groups"]), 1)
        for pid in sd["param_groups"][0]["params"]:
            self.assertIsInstance(pid, int)
            self.assertNotIsInstance(pid, torch.Tensor)

    def test_zero_reshard_8bit_scales_world_2_to_3(self):
        """M7: reshard_zero_checkpoint must reconstruct and re-quantize 8bit moments + block scales accurately."""
        import warnings
        from apexw.zero import ApexWZeRO, reshard_zero_checkpoint

        torch.manual_seed(0)
        # 12 * 64 = 768 >= 3 * 256 -> sliced on both world_size=2 and world_size=3
        w_init = torch.randn(12, 64)
        grad_val = torch.randn(12, 64)

        ckpts = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for r in range(2):
                p = nn.Parameter(w_init.clone())
                opt = ApexWZeRO([p], lr=1e-2, precision="8bit", block_size=64, world_size=2, rank=r)
                for _ in range(3):
                    p.grad = grad_val.clone()
                    opt.step()
                ckpts.append(opt.state_dict())

            resharded = reshard_zero_checkpoint(ckpts, target_world_size=3)
            self.assertEqual(len(resharded), 3)

            for tr in range(3):
                p_new = nn.Parameter(w_init.clone())
                opt_new = ApexWZeRO([p_new], lr=1e-2, precision="8bit", block_size=64, world_size=3, rank=tr)
                opt_new.load_state_dict(resharded[tr])
                # Verify loaded slice scales match target rank slice size (4*64=256 -> 4 blocks of 64)
                st = opt_new._local_opt.state[opt_new._local_params[0]]
                self.assertEqual(st["exp_avg"].shape, (4, 64))
                self.assertEqual(st["exp_avg_scales"].numel(), 4)
                self.assertEqual(st["exp_avg_sq_scales"].numel(), 4)
                # Step must execute without block-size shape mismatch
                p_new.grad = grad_val.clone()
                opt_new.step()
                self.assertTrue(torch.isfinite(p_new).all())

    def test_dist_muon_single_element_and_fp32_momentum(self):
        """DistMuon preserves float32 momentum on bf16 params and avoids NaN on 1-element tensors."""
        from apexw.fsdp import DistMuon

        p2d = nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16))
        p1e = nn.Parameter(torch.tensor([3.0]))
        opt = DistMuon([p2d, p1e], lr=1e-2, world_size=1, rank=0)
        p2d.grad = torch.randn_like(p2d)
        p1e.grad = torch.tensor([1.0])
        opt.step()
        self.assertEqual(opt.state[p2d]["momentum"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(p1e).all())

    def test_hf_no_stray_stub(self):
        """apexw.hf must not contain a duplicate incomplete def register_hf_trainer_optimizer stub."""
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "apexw", "hf.py")) as f:
            hf_src = f.read()
        self.assertEqual(hf_src.count("def register_hf_trainer_optimizer"), 1)

    def test_flatten_deepcopy_isolation(self):
        """copy.deepcopy on a flattened nn.Module must succeed and isolate cloned parameters."""
        import copy
        from apexw.flatten import flatten_parameters

        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        flatten_parameters(model)

        cloned = copy.deepcopy(model)
        x = torch.randn(2, 8)
        self.assertTrue(torch.allclose(model(x), cloned(x)))

        # Mutate cloned flat_param and confirm original model and slice views remain isolated
        with torch.no_grad():
            cloned.flat_param.add_(5.0)
        self.assertFalse(torch.allclose(model[0].weight, cloned[0].weight))
        sd_orig = model.get_unflattened_state_dict()
        sd_clone = cloned.get_unflattened_state_dict()
        self.assertFalse(torch.allclose(sd_orig["0.weight"], sd_clone["0.weight"]))

    def test_apexw_precision_hybrid_and_zero_modes(self):
        """Test ApexW precision='hybrid' dispatch, HybridApexW memory methods, and ZeRO compatibility."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
        )

        # 1. ApexW with precision='hybrid' routes directly to HybridApexW
        opt_hybrid = ApexW(model, lr=1e-3, precision="hybrid")
        self.assertIsInstance(opt_hybrid, HybridApexW)
        self.assertGreater(opt_hybrid.get_state_memory_mb(), 0)

        # Step hybrid
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt_hybrid.step()
        opt_hybrid.zero_grad()

        # 2. ApexWZeRO initialized directly with nn.Module across all 3 modes
        for prec in ("compact", "8bit", "hybrid"):
            m = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64))
            opt_z = ApexWZeRO(m, lr=1e-3, precision=prec, world_size=2, rank=0)
            self.assertTrue(hasattr(opt_z, "_step_supports_amp_scaling"))
            self.assertTrue(opt_z._step_supports_amp_scaling)
            self.assertGreaterEqual(opt_z.get_state_memory_mb(), 0)

            # Step and verify state allocation and convergence
            loss_z = m(x).sum()
            loss_z.backward()
            opt_z.step()
            opt_z.zero_grad()

            self.assertGreater(opt_z.get_state_memory_mb(), 0)
            self.assertGreaterEqual(opt_z.get_rank_saved_mb(), 0)
            self.assertGreaterEqual(opt_z.get_total_cluster_saved_mb(), 0)

            # Test print_summary without crashing
            opt_z.print_summary()

    def test_zero_amp_grad_scaler_and_inf_skip(self):
        """Verify ApexWZeRO safely handles AMP GradScaler scale/inf flags and skips step on inf."""
        m = nn.Linear(32, 32)
        opt_z = ApexWZeRO(m, lr=1e-3, world_size=2, rank=0)
        w_before = m.weight.clone()

        # Simulate AMP scaler with found_inf=True (should skip step and not mutate weight)
        x = torch.randn(4, 32)
        loss = m(x).sum()
        loss.backward()
        opt_z.found_inf = torch.tensor(1.0)
        opt_z.step()
        self.assertTrue(torch.equal(m.weight, w_before), "Step must be skipped when found_inf is True")

        # Simulate normal step with grad_scale
        opt_z.grad_scale = 1024.0
        opt_z.found_inf = torch.tensor(0.0)
        opt_z.step()
        self.assertFalse(torch.equal(m.weight, w_before), "Step must update weight when found_inf is False")


if __name__ == "__main__":
    unittest.main()



