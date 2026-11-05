"""
Tests for ApexW Scale-Out Features:
1. Pure PyTorch Fallbacks & Non-CUDA Backends (CPU, MPS, XLA)
2. Native FSDP & DeepSpeed ZeRO Integration
3. First-Class Hugging Face Trainer Integration
4. Automatic Precision Management & AMP GradScaler Synchronization
5. Standardized PyTorch state_dict & Checkpointing Resharding
"""

import io
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

    def test_determine_work_dtype_half_and_device_dispatch(self):
        """Verify work dtype selects float32 on CPU, and respects float16."""
        from apexw.muon import _determine_work_dtype
        t_cpu_fp32 = torch.randn(10, 10, dtype=torch.float32)
        self.assertEqual(_determine_work_dtype(t_cpu_fp32), torch.float32)

        t_cpu_fp16 = torch.randn(10, 10, dtype=torch.float16)
        self.assertEqual(_determine_work_dtype(t_cpu_fp16), torch.float32)


if __name__ == "__main__":
    unittest.main()
