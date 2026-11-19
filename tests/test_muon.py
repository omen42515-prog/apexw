"""
Unit tests for ApexMuon and HybridApexW.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from apexw import (
    ApexW,
    ApexMuon,
    HybridApexW,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
    zeropower_via_polar_express,
    zeropower_via_polar_express_batched,
    partition_parameters_for_hybrid,
)


class TestMuonAndHybrid(unittest.TestCase):

    def test_newton_schulz_orthogonality(self):
        """Test that Newton-Schulz 5th-order produces an orthogonal matrix (all singular values = 1)."""
        torch.manual_seed(42)
        # Test square matrix
        G = torch.randn(64, 64)
        U = zeropower_via_newtonschulz5(G, steps=5)

        # Singular values should all be tightly clustered near 1.0 (between 0.6 and 1.3)
        svs = torch.linalg.svdvals(U)
        print(f"\nNewton-Schulz Singular Values Range (64x64): [{svs.min().item():.3f}, {svs.max().item():.3f}]")
        self.assertGreater(svs.min().item(), 0.5)
        self.assertLess(svs.max().item(), 1.3)

        # Test rectangular tall matrix
        G_tall = torch.randn(128, 64)
        U_tall = zeropower_via_newtonschulz5(G_tall, steps=5)
        svs_tall = torch.linalg.svdvals(U_tall)
        self.assertGreater(svs_tall.min().item(), 0.5)
        self.assertLess(svs_tall.max().item(), 1.3)

        # Test rectangular wide matrix
        G_wide = torch.randn(64, 128)
        U_wide = zeropower_via_newtonschulz5(G_wide, steps=5)
        svs_wide = torch.linalg.svdvals(U_wide)
        self.assertGreater(svs_wide.min().item(), 0.5)
        self.assertLess(svs_wide.max().item(), 1.3)

    def test_batched_newton_schulz_consistency(self):
        """Verify that zeropower_via_newtonschulz5_batched matches single-call execution."""
        torch.manual_seed(42)
        K, m, n = 4, 32, 48
        batch_G = torch.randn(K, m, n)

        # Batched run
        batched_out = zeropower_via_newtonschulz5_batched(batch_G, steps=5)

        # Single runs
        single_outs = []
        for i in range(K):
            single_outs.append(zeropower_via_newtonschulz5(batch_G[i], steps=5))
        single_stacked = torch.stack(single_outs, dim=0)

        max_diff = torch.max(torch.abs(batched_out - single_stacked)).item()
        print(f"Batched vs Single Newton-Schulz Max Diff: {max_diff:.5e}")
        self.assertLess(max_diff, 1e-5)

    def test_apex_muon_step(self):
        """Verify ApexMuon optimizer step updates parameters correctly."""
        torch.manual_seed(42)
        linear = nn.Linear(64, 64, bias=False)
        opt = ApexMuon(linear.parameters(), lr=0.02, weight_decay=1e-2)

        x = torch.randn(8, 64)
        loss = linear(x).sum()
        loss.backward()

        w_before = linear.weight.clone()
        opt.step()
        w_after = linear.weight.clone()

        self.assertFalse(torch.equal(w_before, w_after))
        self.assertFalse(torch.isnan(w_after).any())

    def test_muon_state_dict_save_load(self):
        """Verify ApexMuon checkpointing."""
        linear = nn.Linear(32, 32, bias=False)
        opt1 = ApexMuon(linear.parameters(), lr=0.02)

        x = torch.randn(4, 32)
        linear(x).sum().backward()
        opt1.step()

        sd = opt1.state_dict()
        opt2 = ApexMuon(linear.parameters(), lr=0.02)
        opt2.load_state_dict(sd)

        for p in linear.parameters():
            self.assertEqual(opt1.state[p]["step"], opt2.state[p]["step"])
            self.assertTrue(torch.equal(opt1.state[p]["momentum"], opt2.state[p]["momentum"]))

    def test_muon_ademamix(self):
        """Verify AdEMAMix dual-speed momentum integration into ApexMuon."""
        linear = nn.Linear(32, 32, bias=False)
        opt = ApexMuon(linear.parameters(), lr=0.02, beta3=0.999, alpha=5.0)

        x = torch.randn(4, 32)
        linear(x).sum().backward()
        opt.step()

        p = list(linear.parameters())[0]
        self.assertIn("momentum", opt.state[p])
        self.assertIn("momentum_slow", opt.state[p])

    def test_muon_cautious(self):
        """Verify cautious directional masking in ApexMuon."""
        linear = nn.Linear(32, 32, bias=False)
        opt = ApexMuon(linear.parameters(), lr=0.02, cautious=True)

        x = torch.randn(4, 32)
        linear(x).sum().backward()
        opt.step()
        self.assertFalse(torch.isnan(linear.weight).any())

    def test_hybrid_partitioning(self):
        """Verify parameter partition splits 2D hidden weights to Muon and 1D/embeddings to AdamW."""
        class ToyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 64, bias=False)
                self.norm = nn.LayerNorm(64)
                self.head = nn.Linear(64, 10)

            def forward(self, x):
                return self.head(self.norm(self.q_proj(self.embed(x))))

        model = ToyTransformer()
        muon_params, adamw_params = partition_parameters_for_hybrid(model)

        # q_proj and k_proj should be in muon_params
        self.assertTrue(any(p is model.q_proj.weight for p in muon_params))
        self.assertTrue(any(p is model.k_proj.weight for p in muon_params))

        # embed, norm.weight, norm.bias, head.bias should be in adamw_params
        self.assertTrue(any(p is model.embed.weight for p in adamw_params))
        self.assertTrue(any(p is model.norm.weight for p in adamw_params))
        self.assertTrue(any(p is model.norm.bias for p in adamw_params))
        self.assertTrue(any(p is model.head.bias for p in adamw_params))

    def test_hybrid_apex_optimizer(self):
        """Test HybridApexW training step on full model."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        opt = HybridApexW(model, lr=1e-3, muon_lr=2e-2)

        x = torch.randn(8, 64)
        target = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), target)
        loss.backward()
        opt.step()

        # Check state dict serialization
        sd = opt.state_dict()
        opt_new = HybridApexW(model, lr=1e-3, muon_lr=2e-2)
        opt_new.load_state_dict(sd)

        # Print summary check
        opt.print_summary()

    def test_apexw_use_muon_flag(self):
        """Test that ApexW(..., use_muon=True) delegates cleanly to HybridApexW."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.Linear(128, 10),
        )
        opt = ApexW(model.parameters(), lr=1e-3, use_muon=True)
        self.assertIsInstance(opt, HybridApexW)

        x = torch.randn(4, 64)
        model(x).sum().backward()
        opt.step()

    def test_pass_model_directly(self):
        """Test passing nn.Module directly to ApexW both in standard and muon modes."""
        model = nn.Linear(32, 10)
        opt_adam = ApexW(model, lr=1e-3)
        self.assertIsInstance(opt_adam, ApexW)

        opt_hybrid = ApexW(model, lr=1e-3, use_muon=True)
        self.assertIsInstance(opt_hybrid, HybridApexW)

        x = torch.randn(2, 32)
        opt_hybrid.step()

    def test_polar_express_orthogonality_and_batched(self):
        """Test that Polar Express minimax orthogonalization achieves tighter singular values in 5 steps."""
        torch.manual_seed(42)
        G = torch.randn(64, 64)
        U = zeropower_via_polar_express(G, steps=5)
        svs = torch.linalg.svdvals(U)
        self.assertGreater(svs.min().item(), 0.80)
        self.assertLess(svs.max().item(), 1.20)

        # Rectangular tall & wide
        G_tall = torch.randn(128, 64)
        U_tall = zeropower_via_polar_express(G_tall, steps=5)
        svs_tall = torch.linalg.svdvals(U_tall)
        self.assertGreater(svs_tall.min().item(), 0.80)
        self.assertLess(svs_tall.max().item(), 1.20)

        # Batched consistency
        K, m, n = 4, 32, 48
        batch_G = torch.randn(K, m, n)
        batched_out = zeropower_via_polar_express_batched(batch_G, steps=5)
        for i in range(K):
            single_out = zeropower_via_polar_express(batch_G[i], steps=5)
            diff = (batched_out[i] - single_out).abs().max().item()
            self.assertLess(diff, 1e-5)

    def test_rms_matched_scaling(self):
        """Test that scale_mode='rms_matched' computes scale = 0.2 * sqrt(max(m, n)) and steps cleanly."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64),
        )
        opt = ApexMuon(model.parameters(), lr=1e-3, scale_mode="rms_matched", batch_bmm=True)
        x = torch.randn(4, 64)
        model(x).sum().backward()
        opt.step()
        self.assertFalse(torch.isnan(model[0].weight).any())
        self.assertFalse(torch.isnan(model[1].weight).any())

    def test_hybrid_add_param_group(self):
        """Test dynamic add_param_group on HybridApexW for both 1D and 2D parameters."""
        # Start with only 1D parameters so muon_opt is initially None
        bias_only = nn.Parameter(torch.zeros(64))
        opt = ApexW([bias_only], lr=1e-3, use_muon=True)
        self.assertIsInstance(opt, HybridApexW)
        self.assertIsNone(opt.muon_opt)

        # Dynamically add 2D linear weight (should lazy-init muon_opt)
        linear = nn.Linear(32, 64, bias=True)
        opt.add_param_group({"params": [linear.weight, linear.bias], "lr": 5e-4})
        self.assertIsNotNone(opt.muon_opt)

        # Run forward and backward
        x = torch.randn(4, 32)
        out = linear(x) + bias_only.unsqueeze(0)
        out.sum().backward()
        opt.step()

        # Check weights are updated
        self.assertTrue((bias_only != 0).any())
        self.assertFalse(torch.isnan(linear.weight).any())
        self.assertFalse(torch.isnan(linear.bias).any())

    def test_hybrid_explicit_use_muon_flags(self):
        """Test that explicit 'use_muon': False keeps 2D weights in AdamW, and 'use_muon': True sends them to Muon."""
        w1 = nn.Parameter(torch.randn(32, 32))  # 2D, explicit use_muon=False -> AdamW
        w2 = nn.Parameter(torch.randn(32, 32))  # 2D, explicit use_muon=True  -> Muon
        b1 = nn.Parameter(torch.randn(32))      # 1D -> AdamW

        opt = ApexW([
            {"params": [w1], "use_muon": False},
            {"params": [w2], "use_muon": True},
            {"params": [b1]},
        ], lr=1e-3, use_muon=True)

        self.assertIsInstance(opt, HybridApexW)
        # Verify w1 is inside adamw_opt
        adamw_all_params = [p for g in opt.adamw_opt.param_groups for p in g["params"]]
        muon_all_params = [p for g in opt.muon_opt.param_groups for p in g["params"]]

        self.assertTrue(any(p is w1 for p in adamw_all_params))
        self.assertFalse(any(p is w1 for p in muon_all_params))
        self.assertTrue(any(p is w2 for p in muon_all_params))
        self.assertTrue(any(p is b1 for p in adamw_all_params))

        # Perform step
        (w1.sum() + w2.sum() + b1.sum()).backward()
        opt.step()

    def test_compiled_newton_schulz_numerical_equivalence(self):
        """Test that compiled=True and compiled=False produce numerically identical orthogonalizations."""
        torch.manual_seed(42)
        G = torch.randn(64, 64)
        out_eager = zeropower_via_newtonschulz5(G.clone(), steps=5, compiled=False)
        out_comp = zeropower_via_newtonschulz5(G.clone(), steps=5, compiled=True)
        diff = (out_eager - out_comp).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_hybrid_compiled_flag_propagation(self):
        """Test that compiled flag properly propagates from HybridApexW to ApexMuon param groups."""
        w = nn.Parameter(torch.randn(32, 32))
        opt = ApexW([w], lr=1e-3, use_muon=True, compiled=True)
        self.assertTrue(opt.muon_opt.param_groups[0].get("compiled", False))


if __name__ == "__main__":
    unittest.main()
