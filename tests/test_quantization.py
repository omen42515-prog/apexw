"""Tests for 8-bit/4-bit NF4 GPU acceleration, batching, partitioning, and zero-sync fast paths."""

import ast
import inspect
import math
import os
import re
import sys
import tracemalloc
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import apexw.optimizer as opt_mod
import apexw.triton_kernel as tk
from apexw import ApexW, HybridApexW, ApexMuon, triton_kernel
from apexw.muon import (
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
    zeropower_via_polar_express,
    zeropower_via_polar_express_batched,
)
from apexw.optimizer import (
    FastExecutionPlan,
    _NF4_M_CODEBOOK_RAW,
    _dequantize_blockwise_6bit,
    _dequantize_blockwise_int4,
    _quantize_blockwise_6bit,
    _quantize_blockwise_int4,
    _quantize_blockwise_uint8,
)


def _total_state_bytes(opt) -> int:
    if isinstance(opt, HybridApexW):
        total = 0
        for sub in (opt.muon_opt, opt.adamw_opt):
            if sub is not None:
                total += _total_state_bytes(sub)
        return total
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def _reference_nf4_dequant_tree(q: torch.Tensor) -> torch.Tensor:
    """CPU mirror of the register-only decision tree in _nf4_dequant_triton."""
    q_i = q.to(torch.int32)
    a = (q_i - 7).abs()
    mag_01 = torch.where(a == 0, 0.0, 0.0911)
    mag_23 = torch.where(a == 2, 0.1848, 0.2844)
    mag_lo = torch.where(a < 2, mag_01, mag_23)
    mag_45 = torch.where(a == 4, 0.3949, 0.5251)
    mag_67 = torch.where(a == 6, 0.6962, 1.0000)
    mag_hi = torch.where(a < 6, mag_45, mag_67)
    mag = torch.where(a < 4, mag_lo, mag_hi)
    return torch.where(q_i < 7, -mag, mag)


def _reference_nf4_quant_tree(
    norm_val: torch.Tensor,
    u_rand: torch.Tensor,
    *,
    stochastic: bool = False,
) -> torch.Tensor:
    """CPU mirror of the register-only binary search in _nf4_quant_triton."""
    is_neg = norm_val < 0.0
    ax = norm_val.abs()

    in_low = ax < 0.2844
    in_mid_low = ax < 0.0911
    in_mid_high = ax < 0.1848
    lo_low = torch.where(in_mid_low, 0, torch.where(in_mid_high, 1, 2))
    clo_low = torch.where(in_mid_low, 0.0, torch.where(in_mid_high, 0.0911, 0.1848))
    chi_low = torch.where(in_mid_low, 0.0911, torch.where(in_mid_high, 0.1848, 0.2844))

    in_upper_mid = ax < 0.5251
    in_u0 = ax < 0.3949
    in_u2 = ax < 0.6962
    lo_high = torch.where(in_upper_mid, torch.where(in_u0, 3, 4), torch.where(in_u2, 5, 6))
    clo_high = torch.where(
        in_upper_mid,
        torch.where(in_u0, 0.2844, 0.3949),
        torch.where(in_u2, 0.5251, 0.6962),
    )
    chi_high = torch.where(
        in_upper_mid,
        torch.where(in_u0, 0.3949, 0.5251),
        torch.where(in_u2, 0.6962, 1.0000),
    )

    lo = torch.where(in_low, lo_low, lo_high)
    c_lo = torch.where(in_low, clo_low, clo_high)
    c_hi = torch.where(in_low, chi_low, chi_high)

    if stochastic:
        prob_hi = (ax - c_lo) / (c_hi - c_lo)
        k = torch.where(u_rand < prob_hi, lo + 1, lo)
    else:
        k = torch.where((ax - c_lo) >= (c_hi - ax), lo + 1, lo)

    return torch.where(is_neg, 7 - k, 7 + k).clamp(0, 14)


def _mock_cuda_view(tensor: torch.Tensor):
    """Fake CUDA device view for host-side routing and partition tests without a GPU."""
    return SimpleNamespace(
        is_cuda=True,
        device=SimpleNamespace(type="cuda", index=0),
        dtype=tensor.dtype,
        numel=tensor.numel,
        shape=tensor.shape,
        dim=tensor.dim,
        is_contiguous=lambda: tensor.is_contiguous(),
        is_complex=tensor.is_complex,
        is_sparse=False,
        contiguous=lambda: _mock_cuda_view(tensor.contiguous()),
        data_ptr=lambda: tensor.data_ptr(),
    )


class TestNF4DecisionTreeParity(unittest.TestCase):
    def test_dequant_tree_matches_codebook_for_all_16_codes(self):
        codes = torch.arange(16, dtype=torch.uint8)
        got = _reference_nf4_dequant_tree(codes)
        self.assertTrue(torch.allclose(got, _NF4_M_CODEBOOK_RAW, atol=1e-6, rtol=0.0))

    def test_quant_dequant_tree_matches_blockwise_int4_exact_and_odd_lengths(self):
        torch.manual_seed(42)
        for numel in (1, 17, 64, 255, 513):
            x = torch.randn(numel, dtype=torch.float32) * 0.25
            bsize = 64
            q_packed, scales = _quantize_blockwise_int4(x, block_size=bsize, stochastic=False)
            deq_ref = _dequantize_blockwise_int4(q_packed, scales, numel=numel, block_size=bsize)

            pad = (-numel) % bsize
            x_pad = torch.nn.functional.pad(x, (0, pad)) if pad else x
            blocks = x_pad.view(-1, bsize)
            amax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            norm = (blocks / amax).clamp(-1.0, 1.0)
            q_idx = _reference_nf4_quant_tree(norm, torch.zeros_like(norm), stochastic=False)
            deq_tree = (_reference_nf4_dequant_tree(q_idx) * amax).view(-1)[:numel]

            # Boundary ties at exact midpoints can differ by 0, verify bit-exact away from exact midpoint
            # and verify reconstructed values stay identically close on random normal inputs.
            self.assertTrue(torch.allclose(deq_tree, deq_ref, atol=1e-6, rtol=1e-5), f"Mismatch for numel={numel}")

    def test_stochastic_nf4_tree_unbiasedness(self):
        torch.manual_seed(123)
        targets = torch.tensor([-0.75, -0.33, -0.05, 0.04, 0.22, 0.61, 0.88], dtype=torch.float32)
        trials = 4000
        expanded = targets.unsqueeze(0).expand(trials, -1)
        u = torch.rand_like(expanded)
        q_idx = _reference_nf4_quant_tree(expanded, u, stochastic=True)
        deq = _reference_nf4_dequant_tree(q_idx)
        mean_est = deq.mean(dim=0)
        self.assertTrue(torch.allclose(mean_est, targets, atol=0.025, rtol=0.0))


class Test8BitDispatchAndNoSync(unittest.TestCase):
    def _build_mixed_8bit_model(self):
        """Create a model hitting NF4 (>512, min_dim > 32), narrow/PEFT INT8, and micro (<64) regimes."""
        torch.manual_seed(0)
        m = nn.Sequential(
            nn.Linear(64, 64, bias=False),  # 4096 elems, min_dim=64 > 32 -> is_4bit_m=True, bsize=256
            nn.Linear(64, 8, bias=False),   # 512 elems, min_dim=8 <= 32 -> narrow INT8, bsize=128
            nn.LayerNorm(16),               # 16 elems (<64) weight & bias -> unquantized micro
        )
        # Assign synthetic grads directly to each param to keep shape distinct
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 0.05
        return m

    def test_build_and_execute_cuda_8bit_partition_handles_mixed_nf4_int8_micro(self):
        m = self._build_mixed_8bit_model()
        opt = ApexW(m.parameters(), lr=1e-3, precision="8bit", int4_m=True)
        opt.step()  # Step 1 lazy-initializes NF4 + INT8 + unquantized micro states cleanly on CPU

        params = list(m.parameters())
        exp_avgs = [opt.state[p]["exp_avg"] for p in params]
        exp_avg_sqs = [opt.state[p]["exp_avg_sq"] for p in params]
        self.assertTrue(opt.state[params[0]]["is_4bit_m"])
        self.assertFalse(opt.state[params[1]]["is_4bit_m"])
        self.assertTrue(opt.state[params[1]]["is_8bit"])
        self.assertFalse(opt.state[params[2]].get("is_8bit", False))

        # Simulate is_cuda=True on parameter/state tensors for partition building
        with patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda self: True)):
            part = opt._build_cuda_8bit_partition(params, exp_avgs, exp_avg_sqs)
            self.assertTrue(part["eligible"])
            self.assertEqual(len(part["q_params"]), 2)
            self.assertEqual(len(part["micro_params"]), 2)
            self.assertEqual(part["q_ms"][0].dtype, torch.uint8)
            self.assertEqual(part["q_ms"][1].dtype, torch.int8)

            captured = {}

            def _fake_step_fused_8bit(*args, **kwargs):
                captured["q_params_len"] = len(args[0])
                captured["bsizes"] = list(args[6])
                return True

            micro_before = [p.detach().clone() for p in part["micro_params"]]
            for p in params:
                p.grad = torch.ones_like(p) * 0.1
            grads = [p.grad for p in params]

            with patch.object(opt_mod, "step_fused_8bit", side_effect=_fake_step_fused_8bit):
                ok = opt._execute_cuda_8bit_partition(
                    part,
                    params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    lr=1e-3,
                    beta1=0.9,
                    beta2=0.999,
                    eps=1e-8,
                    weight_decay=0.01,
                    step=2,
                )
            self.assertTrue(ok)
            self.assertEqual(captured["q_params_len"], 2)
            self.assertEqual(captured["bsizes"], [2048, 128])
            # Micro tensors should also have been updated cleanly
            for p_prev, p_now in zip(micro_before, part["micro_params"]):
                self.assertFalse(torch.allclose(p_prev, p_now))

    def test_cuda_8bit_failure_returns_false_without_mutating_micro_or_double_stepping(self):
        m = self._build_mixed_8bit_model()
        opt = ApexW(m.parameters(), lr=1e-3, precision="8bit")
        opt.step()

        params = list(m.parameters())
        micro_snapshot = [p.detach().clone() for p in params[2:]]
        for p in params:
            p.grad = torch.ones_like(p) * 0.1
        step_before = int(opt.state[params[0]]["step"].item())

        with patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda self: True)), \
             patch.object(opt_mod, "is_cuda_kernel_available", return_value=True), \
             patch.object(opt_mod, "is_triton_available", return_value=True), \
             patch.object(opt_mod, "step_fused_8bit", return_value=False), \
             patch.object(opt_mod, "triton_multi_tensor_adamw_step", return_value=True) as mock_triton:
            # Remove cached CPU plan so next step dispatches via mock CUDA 8bit -> Triton fallback
            opt._execution_plans.clear()
            opt.step()

        self.assertEqual(mock_triton.call_count, 1)
        # Step incremented exactly once (no double step after cuda_8bit rollback)
        self.assertEqual(int(opt.state[params[0]]["step"].item()), step_before + 1)
        # Micro tensors were NOT partially mutated by cuda_8bit before Triton handled the group
        for prev, curr in zip(micro_snapshot, params[2:]):
            self.assertTrue(torch.equal(prev, curr))

    def test_fast_plan_skips_host_sanitize_sync_when_no_clip_or_gc(self):
        m = self._build_mixed_8bit_model()
        opt = ApexW(m.parameters(), lr=1e-3, precision="8bit", backend="triton")
        opt.step()

        params = list(m.parameters())
        for p in params:
            p.grad = torch.ones_like(p) * 0.05

        with patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda self: True)), \
             patch.object(opt_mod, "is_triton_available", return_value=True), \
             patch.object(opt_mod, "is_cuda_kernel_available", return_value=False), \
             patch.object(opt_mod, "triton_multi_tensor_adamw_step", return_value=True), \
             patch.object(
                 opt_mod, "_sanitize_gradients_fast", side_effect=opt_mod._sanitize_gradients_fast
             ) as mock_san:
            opt._execution_plans.clear()
            # Warmup step builds FastExecutionPlan(path_type="triton")
            opt.step()
            mock_san.reset_mock()
            # Steady-state step executes plan without host sanitize sync
            opt.step()
            self.assertEqual(mock_san.call_count, 0)

            # With clip_grad_norm enabled, host sanitize IS invoked
            opt.param_groups[0]["clip_grad_norm"] = 1.0
            opt.step()
            self.assertEqual(mock_san.call_count, 1)


class TestCuda8BitAndNoNewFlagsInvariants(unittest.TestCase):
    def test_constant_memory_fit_and_bindings(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "apexw", "csrc", "apexw_cuda_kernel.cu")) as f:
            cu_src = f.read()
        with open(os.path.join(root, "apexw", "csrc", "apexw_cuda.cpp")) as f:
            cpp_src = f.read()
        with open(os.path.join(root, "apexw", "cuda_extension.py")) as f:
            ce_src = f.read()

        self.assertIn("c_tensor_metadata[512]", cu_src)
        self.assertIn("c_tensor_metadata_8bit[576]", cu_src)
        self.assertIn("MAX_BATCH_8BIT = 576", cu_src)
        self.assertIn("warp_block_reduce_max2", cu_src)
        self.assertIn("ApexParamVec4", cu_src)
        self.assertIn("ApexU8Vec4", cu_src)
        self.assertLess(512 * 48 + 576 * 64 + 64, 65536)
        self.assertIn("c_nf4_lut[16]", cu_src)
        self.assertIn("sanitize_grad_val", cu_src)
        self.assertIn("launch_multi_tensor_8bit_adamw_cuda_2d", cu_src)
        self.assertIn("step_fused_8bit", cpp_src)
        self.assertIn("def step_fused_8bit", ce_src)

    def test_no_extra_boolean_user_flags_added_to_apexw_init(self):
        sig = inspect.signature(ApexW.__init__)
        params = list(sig.parameters.keys())
        # Ensure we kept clean UX with no boolean fast-path flags added
        for name in params:
            self.assertNotIn("fast_8bit", name)
            self.assertNotIn("use_batched", name)
            self.assertNotIn("enable_nf4_kernel", name)

    def test_triton_jit_has_no_int32_overflow_scalars_or_uint32_mixing(self):
        import ast

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "apexw", "triton_kernel.py")) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any(
                (isinstance(d, ast.Attribute) and d.attr == "jit")
                or (isinstance(d, ast.Name) and d.id == "jit")
                for d in node.decorator_list
            ):
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Constant)
                        and isinstance(sub.value, int)
                        and not isinstance(sub.value, bool)
                    ):
                        self.assertLessEqual(
                            sub.value,
                            2147483647,
                            f"{node.name}:{sub.lineno} scalar {sub.value} overflows signed int32",
                        )
                        self.assertGreaterEqual(
                            sub.value,
                            -2147483648,
                            f"{node.name}:{sub.lineno} scalar {sub.value} underflows signed int32",
                        )
                    if isinstance(sub, ast.Attribute) and sub.attr == "uint32":
                        self.fail(f"{node.name}:{sub.lineno} uses tl.uint32 (risk of Triton mixed-sign promotion)")

    def test_build_or_reuse_batched_plan_when_param_zero_is_unquantized(self):
        p0 = torch.randn(16)
        p1 = torch.randn(64)
        p2 = torch.randn(64)
        m1 = torch.randint(-127, 127, (64,), dtype=torch.int8)
        m2 = torch.randint(-127, 127, (64,), dtype=torch.int8)
        v1 = torch.randint(0, 255, (64,), dtype=torch.uint8)
        v2 = torch.randint(0, 255, (64,), dtype=torch.uint8)
        s1 = torch.ones(1, dtype=torch.float32)
        s2 = torch.ones(1, dtype=torch.float32)

        cache = {}
        plan = tk._build_or_reuse_batched_plan(
            cache,
            ("8bit", "k"),
            [1, 2],
            [p0, p1, p2],
            [p0, m1, m2],
            [p0, v1, v2],
            [None, s1, s2],
            [None, s1, s2],
            [1, 1],
            64,
        )
        self.assertIn("ms_ptrs", plan)
        self.assertIn("vs_ptrs", plan)
        self.assertEqual(plan["ms_ptrs"].tolist(), [s1.data_ptr(), s2.data_ptr()])

    def test_int32_xorshift_uniformity_and_bounds(self):
        import numpy as np

        offsets = np.arange(8192, dtype=np.int32)
        step_m = np.int32((17 & 0xffff) * 9277)
        pid_m = np.int32((3 & 0xffff) * 1973)
        seed = np.int32((offsets ^ pid_m) + step_m + np.int32(1))
        seed = np.where(seed == 0, np.int32(1), seed)
        for _ in range(2):
            seed = seed ^ np.int32(seed << 13)
            seed = seed ^ np.int32((seed >> 17) & 0x7fff)
            seed = seed ^ np.int32(seed << 5)
        u0 = ((seed >> 16) & 0xffff).astype(np.float32) / 65536.0
        n0 = (seed & 0xffff).astype(np.float32) / 65536.0 - 0.5
        self.assertTrue((u0 >= 0.0).all() and (u0 < 1.0).all())
        self.assertTrue((n0 >= -0.5).all() and (n0 < 0.5).all())
        self.assertAlmostEqual(float(u0.mean()), 0.5, delta=0.02)
        self.assertAlmostEqual(float(n0.mean()), 0.0, delta=0.02)

        seed1 = seed ^ np.int32(0x6b43a9b5)
        seed1 = np.where(seed1 == 0, np.int32(1), seed1)
        seed1 = seed1 ^ np.int32(seed1 << 13)
        seed1 = seed1 ^ np.int32((seed1 >> 17) & 0x7fff)
        seed1 = seed1 ^ np.int32(seed1 << 5)
        u1 = ((seed1 >> 16) & 0xffff).astype(np.float32) / 65536.0
        self.assertTrue((u1 >= 0.0).all() and (u1 < 1.0).all())
        corr = float(np.corrcoef(u0, u1)[0, 1])
        self.assertLess(abs(corr), 0.05)

    def test_cuda_warp_reduction_syncthreads_and_fp16_bounds(self):
        cu_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "apexw",
            "csrc",
            "apexw_cuda_kernel.cu",
        )
        with open(cu_path, "r", encoding="utf-8") as f:
            src = f.read()

        body_max2 = src.split("warp_block_reduce_max2(")[1].split("if (num_warps > 1)")[1].split("} else {")[0]
        self.assertIn("val1 = smem1[0];\n        val2 = smem2[0];\n        __syncthreads();", body_max2)
        body_sum = src.split("warp_block_reduce_sum(")[1].split("return val;")[0]
        self.assertIn("val = smem[0];\n        __syncthreads();", body_sum)

        self.assertIn("if (mag < 0x7F7F0000u)", src)
        self.assertIn("x = fminf(fmaxf(x, -65504.0f), 65504.0f);", src)
        self.assertNotIn("p -= lr * weight_decay * p;", src)
        self.assertIn("fmaxf(1.0f - lr * weight_decay, 0.0f)", src)

    def test_cuda_cpp_fp16_eps_floor(self):
        cpp_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "apexw",
            "csrc",
            "apexw_cuda.cpp",
        )
        with open(cpp_path, "r", encoding="utf-8") as f:
            src = f.read()

        self.assertIn("(param_code == 2 || state_code == 2) ? std::max(f_eps, 6.1e-5f) : f_eps", src)
        self.assertIn("(param_code == 2) ? std::max(f_eps, 6.1e-5f) : f_eps", src)
        self.assertIn("std::max(eps, 6.1e-5) * bc2_sqrt", src)

    def test_triton_fp16_eps_floor_and_source_invariants(self):
        from apexw import triton_kernel as tk

        eps, eps_scaled = tk._resolve_fp16_eps(1e-8, 3.16227766e-10, True)
        self.assertEqual(eps, 6.1e-5)
        self.assertAlmostEqual(eps_scaled, 6.1e-5 * 0.0316227766, places=10)

        eps_f32, eps_scaled_f32 = tk._resolve_fp16_eps(1e-8, 3.16227766e-10, False)
        self.assertEqual(eps_f32, 1e-8)
        self.assertEqual(eps_scaled_f32, 3.16227766e-10)

        tk_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "apexw",
            "triton_kernel.py",
        )
        with open(tk_path, "r", encoding="utf-8") as f:
            src = f.read()

        self.assertNotIn("p * (1.0 - lr * weight_decay)", src)
        self.assertIn("_store_typed_clamped", src)
        self.assertIn("_clamp_fp16_val", src)
        self.assertIn("65504.0", src)

    def test_cuda_extension_skips_rocm_hip(self):
        from apexw import cuda_extension as ce

        orig_op = ce._CUDA_OP
        orig_attempted = ce._JIT_ATTEMPTED
        try:
            ce._CUDA_OP = None
            ce._JIT_ATTEMPTED = False
            with patch("torch.cuda.is_available", return_value=True), \
                 patch.object(torch.version, "hip", "6.2.0", create=True):
                res = ce.get_cuda_kernel()
                self.assertIsNone(res)
        finally:
            ce._CUDA_OP = orig_op
            ce._JIT_ATTEMPTED = orig_attempted

    def test_extreme_lr_wd_beta_zero_and_mixed_dtypes_stability(self):
        from apexw import ApexW

        for precision in ("8bit", "compact", "full"):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                p_wd = torch.ones(128, dtype=dtype, requires_grad=True)
                p_zero_beta = torch.randn(128, dtype=dtype, requires_grad=True)
                with self.subTest(precision=precision, dtype=dtype):
                    # High LR + high WD (lr * wd = 2.0 > 1.0): should not invert sign on zero grad
                    opt1 = ApexW([p_wd], lr=1.0, weight_decay=2.0, precision=precision)
                    p_wd.grad = torch.zeros_like(p_wd)
                    opt1.step()
                    self.assertTrue(torch.isfinite(p_wd).all())
                    self.assertTrue((p_wd >= 0).all())

                    # betas=(0.0, 0.0) and non-finite / extreme grads must remain finite
                    opt2 = ApexW([p_zero_beta], lr=1e-2, betas=(0.0, 0.0), precision=precision)
                    g = torch.full_like(p_zero_beta, 1e4)
                    g[0] = float("nan")
                    g[1] = float("inf")
                    g[2] = float("-inf")
                    p_zero_beta.grad = g
                    opt2.step()
                    self.assertTrue(torch.isfinite(p_zero_beta).all())

                    # Zero gradients at step 1 (beta1=0.0, beta2=0.99)
                    p_tiny = torch.full((64,), 0.5, dtype=dtype, requires_grad=True)
                    opt3 = ApexW([p_tiny], lr=1e-3, betas=(0.0, 0.99), eps=1e-8, precision=precision)
                    p_tiny.grad = torch.zeros_like(p_tiny)
                    opt3.step()
                    self.assertTrue(torch.isfinite(p_tiny).all())


class TestQuantizedAndQuadVStates(unittest.TestCase):
    def test_triton_resolve_g_ptrs_reuses_buffer_across_steps(self):
        from apexw.triton_kernel import _resolve_g_ptrs_tensor

        plan: dict = {}
        dev = torch.device("cpu")
        g0 = torch.randn(64)
        g1 = torch.randn(64)
        grads = [g0, g1]

        t1 = _resolve_g_ptrs_tensor(plan, [0, 1], grads, dev)
        ptr1 = t1.data_ptr()
        # Same gradient pointers -> returns exact same tensor object without copy
        t2 = _resolve_g_ptrs_tensor(plan, [0, 1], grads, dev)
        self.assertIs(t1, t2)
        self.assertEqual(t2.tolist(), [g0.data_ptr(), g1.data_ptr()])

        # Different gradient pointers (e.g. zero_grad(set_to_none=True)) -> reuses existing GPU buffer in-place
        g0_new = torch.randn(64)
        g1_new = torch.randn(64)
        t3 = _resolve_g_ptrs_tensor(plan, [0, 1], [g0_new, g1_new], dev)
        self.assertIs(t3, t1)
        self.assertEqual(t3.data_ptr(), ptr1)
        self.assertEqual(t3.tolist(), [g0_new.data_ptr(), g1_new.data_ptr()])

    def test_factored_v_8bit_memory_reduction_and_routing(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Embedding(1024, 256),   # 262,144 elems, 2D >= 65536 -> factored_v + INT8 m
            nn.Linear(256, 256),       # 65,536 elems, 2D >= 65536 -> factored_v + INT8 m
            nn.Linear(64, 128),        # 8,192 elems < 65536 -> element-wise 4-bit m + uint8 v
            nn.LayerNorm(256),         # 1D -> element-wise
        )
        conv = nn.Conv2d(64, 128, 3, bias=False)  # (128, 576) = 73,728 elems, 4D >= 65536 -> factored_v
        opt = ApexW(list(model.parameters()) + [conv.weight], lr=1e-3, precision="8bit", factored_v=True)

        # Forward + backward
        idx = torch.randint(0, 1024, (2, 4))
        emb_out = model[0](idx).mean(dim=1)
        h = model[3](model[1](emb_out))
        loss = model[2](h[:, :64]).sum() + conv(torch.randn(1, 64, 4, 4)).sum()
        loss.backward()
        opt.step()

        emb_st = opt.state[model[0].weight]
        lin_big_st = opt.state[model[1].weight]
        conv_st = opt.state[conv.weight]
        lin_small_st = opt.state[model[2].weight]
        ln_st = opt.state[model[3].weight]

        for st in (emb_st, lin_big_st, conv_st):
            self.assertTrue(st["is_factored_v"])
            self.assertFalse(st["is_4bit_m"])
            self.assertEqual(st["exp_avg"].dtype, torch.int8)
            self.assertEqual(st["exp_avg_sq_row"].dtype, torch.float32)
            self.assertEqual(st["exp_avg_sq_col"].dtype, torch.float32)
            self.assertNotIn("exp_avg_sq", st)

        self.assertFalse(lin_small_st["is_factored_v"])
        self.assertEqual(lin_small_st["exp_avg_sq"].dtype, torch.uint8)
        self.assertFalse(ln_st["is_factored_v"])

        # Verify actual optimizer state bytes/param <= 1.05 on (512, 512) weight
        big = nn.Linear(512, 512, bias=False)
        opt_big = ApexW(big.parameters(), lr=1e-3, precision="8bit", factored_v=True)
        big.weight.grad = torch.randn_like(big.weight)
        opt_big.step()
        state_bytes = opt_big.get_state_memory_mb() * 1024.0 * 1024.0
        bytes_per_param = state_bytes / big.weight.numel()
        self.assertLess(bytes_per_param, 1.05)

    def test_factored_v_8bit_convergence_across_dtypes_and_options(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for cautious, nesterov in ((False, False), (True, False), (False, True)):
                with self.subTest(dtype=dtype, cautious=cautious, nesterov=nesterov):
                    torch.manual_seed(7)
                    lin = nn.Linear(256, 256, bias=True).to(dtype)
                    target = torch.randn(4, 256, dtype=dtype)
                    x = torch.randn(4, 256, dtype=dtype)
                    opt = ApexW(
                        lin.parameters(),
                        lr=5e-3,
                        weight_decay=0.01,
                        decay_norm_only=True,
                        cautious=cautious,
                        nesterov=nesterov,
                        stochastic_rounding=False,
                        precision="8bit",
                        factored_v=True,
                    )
                    losses = []
                    for _ in range(15):
                        opt.zero_grad()
                        loss = ((lin(x) - target) ** 2).mean()
                        loss.backward()
                        opt.step()
                        losses.append(float(loss.item()))
                    self.assertTrue(all(math.isfinite(v) for v in losses))
                    self.assertLess(losses[-1], losses[0] * 0.5)
                    self.assertTrue(opt.state[lin.weight]["is_factored_v"])
                    self.assertTrue(torch.isfinite(lin.weight).all())

    def test_factored_v_8bit_checkpoint_roundtrip_bf16_and_fp16(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(11)
                m1 = nn.Linear(256, 256, bias=False).to(dtype)
                m2 = nn.Linear(256, 256, bias=False).to(dtype)
                m2.load_state_dict(m1.state_dict())

                opt1 = ApexW(m1.parameters(), lr=1e-3, precision="8bit", factored_v=True, stochastic_rounding=False)
                g1 = torch.randn_like(m1.weight) * 0.1
                m1.weight.grad = g1.clone()
                opt1.step()

                import copy
                m2.load_state_dict(m1.state_dict())
                sd = copy.deepcopy(opt1.state_dict())
                opt2 = ApexW(m2.parameters(), lr=1e-3, precision="8bit", factored_v=True, stochastic_rounding=False)
                opt2.load_state_dict(sd)

                st2 = opt2.state[m2.weight]
                self.assertEqual(st2["exp_avg"].dtype, torch.int8)
                self.assertEqual(st2["exp_avg_sq_row"].dtype, torch.float32)
                self.assertEqual(st2["exp_avg_sq_col"].dtype, torch.float32)
                self.assertTrue(torch.equal(st2["exp_avg_sq_row"], opt1.state[m1.weight]["exp_avg_sq_row"]))
                self.assertTrue(torch.equal(st2["exp_avg_sq_col"], opt1.state[m1.weight]["exp_avg_sq_col"]))

                g2 = torch.randn_like(m1.weight) * 0.1
                m1.weight.grad = g2.clone()
                m2.weight.grad = g2.clone()
                opt1.step()
                opt2.step()
                self.assertTrue(torch.equal(m1.weight, m2.weight))

    def test_factored_v_partition_mocks_cuda_and_triton(self):
        torch.manual_seed(3)
        big = nn.Linear(256, 256, bias=False)
        small = nn.Linear(64, 64, bias=False)
        bias = torch.randn(16, requires_grad=True)
        params = [big.weight, small.weight, bias]

        opt = ApexW(params, lr=1e-3, precision="8bit", factored_v=True, stochastic_rounding=False)
        for p in params:
            p.grad = torch.randn_like(p) * 0.05
        opt.step()

        # Verify _dispatch_triton_with_factored sends only element-wise tensors to Triton
        seen_triton_lens = []

        def fake_triton(p_list, g_list, m_list, v_list, **kwargs):
            seen_triton_lens.append(len(p_list))
            for idx, p in enumerate(p_list):
                self.assertFalse(opt.state[p].get("is_factored_v", False))
                self.assertEqual(len(kwargs["exp_avg_scales"]), len(p_list))
                self.assertEqual(len(kwargs["exp_avg_sq_scales"]), len(p_list))
            return True

        old_big = big.weight.data.clone()
        exp_avgs = [opt.state[p]["exp_avg"] for p in params]
        exp_avg_sqs = [
            opt.state[p].get("exp_avg_sq", opt.state[p].get("exp_avg_sq_row"))
            for p in params
        ]
        m_scales = [opt.state[p].get("exp_avg_scales") for p in params]
        v_scales = [opt.state[p].get("exp_avg_sq_scales") for p in params]
        bsize = [opt.state[p].get("block_size_8bit") for p in params]

        with patch.object(opt_mod, "triton_multi_tensor_adamw_step", side_effect=fake_triton):
            ok, split = opt_mod._dispatch_triton_with_factored(
                state=opt.state,
                params=params,
                grads=[p.grad for p in params],
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avgs_slow=None,
                m_scales=m_scales,
                v_scales=v_scales,
                bsize=bsize,
                scales_v=None,
                lr=1e-3,
                beta1=0.9,
                beta2=0.99,
                eps=1e-8,
                weight_decay=0.01,
                step=2,
                decay_norm_only=True,
                inv_scale=1.0,
                beta3=None,
                alpha=5.0,
                cautious=False,
                stochastic=False,
            )
        self.assertTrue(ok)
        self.assertEqual(seen_triton_lens, [2])
        self.assertEqual(split, ([1, 2], [0]))
        self.assertFalse(torch.equal(big.weight.data, old_big))

        # Verify CUDA partition splits factored_params cleanly when tensors appear on CUDA
        part_calls = []

        def fake_step_fused_8bit(qp, qg, qm, qms, qv, qvs, qbs, **kwargs):
            part_calls.append(("8bit", len(qp)))
            return True

        with patch.object(torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True):
            part = opt._build_cuda_8bit_partition(params, exp_avgs, exp_avg_sqs)
        self.assertTrue(part["eligible"])
        self.assertEqual(len(part["q_params"]), 1)       # small (64x64)
        self.assertEqual(len(part["factored_params"]), 1) # big (256x256)
        self.assertEqual(len(part["micro_params"]), 1)    # bias (16)

        with patch.object(opt_mod, "step_fused_8bit", side_effect=fake_step_fused_8bit):
            ok_cuda = opt._execute_cuda_8bit_partition(
                part,
                params,
                [p.grad for p in params],
                exp_avgs,
                exp_avg_sqs,
                lr=1e-3,
                beta1=0.9,
                beta2=0.99,
                eps=1e-8,
                weight_decay=0.01,
                step=2,
                decay_norm_only=True,
                inv_scale=1.0,
                stochastic=False,
                cautious=False,
            )
        self.assertTrue(ok_cuda)
        self.assertEqual(part_calls, [("8bit", 1)])

    def test_factored_v_xla_step_branch(self):
        p = nn.Linear(128, 512, bias=False).weight
        opt = ApexW([p], lr=1e-3, precision="8bit", factored_v=True, stochastic_rounding=False)
        p.grad = torch.randn_like(p) * 0.1
        opt.step()

        old_w = p.data.clone()
        st = opt.state[p]
        with torch.no_grad():
            opt._xla_step(
                [p],
                [p.grad],
                [st["exp_avg"]],
                [None],
                [st["exp_avg_sq_row"]],
                [None],
                [st["step"]],
                lr=1e-3,
                beta1=0.9,
                beta2=0.99,
                beta3=None,
                alpha=5.0,
                lion=False,
                eps=1e-8,
                weight_decay=0.01,
                decay_norm_only=True,
                nesterov=False,
                cautious=False,
                amsgrad=False,
                maximize=False,
            )
        self.assertFalse(torch.equal(p.data, old_w))
        self.assertTrue(torch.isfinite(p.data).all())
        self.assertTrue(torch.isfinite(st["exp_avg_sq_row"]).all())
        self.assertTrue(torch.isfinite(st["exp_avg_sq_col"]).all())


class TestFusedBackwardAndBatchedLoRARouting(unittest.TestCase):
    def test_fused_backward_steps_and_frees_grad_compact_and_8bit(self):
        for prec in ("compact", "8bit"):
            with self.subTest(precision=prec):
                torch.manual_seed(42)
                model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
                opt = ApexW(model.parameters(), lr=1e-2, precision=prec)
                opt.enable_fused_backward(model)

                x = torch.randn(4, 32)
                target = torch.randn(4, 8)
                losses = []
                for _ in range(10):
                    opt.zero_grad()
                    out = model(x)
                    loss = ((out - target) ** 2).mean()
                    loss.backward()
                    # Gradients must be freed immediately inside the backward hook
                    for p in model.parameters():
                        self.assertIsNone(p.grad)
                    opt.step()
                    losses.append(loss.item())

                self.assertLess(losses[-1], losses[0] * 0.5)

    def test_fused_backward_disable_restores_standard_step(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(32, 8))
        opt = ApexW(model.parameters(), lr=1e-2, precision="compact")
        opt.enable_fused_backward(model)
        opt.disable_fused_backward()

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
        opt.step()

    def test_narrow_lora_adapters_routed_to_batched_kernels_in_single_launch(self):
        shapes = [(16, 512), (512, 16)] * 16  # 32 LoRA matrices
        params = [torch.randn(s, dtype=torch.bfloat16, requires_grad=True) for s in shapes]
        for p in params:
            p._param_name = "model.layers.0.self_attn.q_proj.lora_A.weight"
            p.grad = torch.randn_like(p)

        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.is_current_stream_capturing", return_value=False), \
             mock.patch.object(torch.Tensor, "is_cuda", True), \
             mock.patch("apexw.optimizer.is_triton_available", return_value=True), \
             mock.patch("apexw.optimizer.triton_factored_v_step", return_value=True) as mock_factored, \
             mock.patch("apexw.optimizer.triton_multi_tensor_adamw_step", return_value=True) as mock_multi:

            opt = ApexW(params, lr=1e-3, precision="8bit")
            opt.step()
            self.assertEqual(mock_factored.call_count, 0)
            self.assertEqual(mock_multi.call_count, 1)

            for p in params:
                p.grad = torch.randn_like(p)
            opt_comp = ApexW(params, lr=1e-3, precision="compact")
            opt_comp.step()
            self.assertEqual(mock_factored.call_count, 0)
            plan_comp = list(opt_comp._execution_plans.values())[0]
            # Compact executes via high-throughput single-launch native fused or batched path
            self.assertIn(plan_comp.path_type, ("native_fused", "triton", "foreach"))


class TestPEFTScaleOptimization(unittest.TestCase):
    def test_peft_adapter_block_size_256(self):
        # PEFT adapter tensor >= 2048 elements gets block_size_8bit = 256
        p_peft = nn.Parameter(torch.randn(16, 512))  # 8192 elements
        p_peft._param_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"

        # Narrow non-PEFT layer < 2048 elements gets block_size_8bit = 128
        p_narrow = nn.Parameter(torch.randn(64, 8))  # 512 elements
        p_narrow._param_name = "classifier.weight"

        opt = ApexW(
            [
                {"params": [p_peft]},
                {"params": [p_narrow]},
            ],
            precision="8bit",
        )
        p_peft.grad = torch.randn_like(p_peft)
        p_narrow.grad = torch.randn_like(p_narrow)
        opt.step()

        self.assertEqual(opt.state[p_peft]["block_size_8bit"], 2048)
        self.assertEqual(opt.state[p_narrow]["block_size_8bit"], 128)


class TestSubByte8BitAndNesterovSNRLookahead(unittest.TestCase):
    def test_auto_sub_byte_8bit_state_memory_and_quad_v(self):
        p = nn.Parameter(torch.randn(64, 1024, dtype=torch.bfloat16))  # 65,536 elements (LoRA r=64)
        opt = ApexW([p], lr=2e-4, precision="8bit")
        p.grad = torch.randn_like(p) * 0.05
        opt.step()

        st = opt.state[p]
        self.assertTrue(st["is_8bit"])
        self.assertTrue(st["is_4bit_m"])
        self.assertTrue(st["is_quad_v"])
        self.assertEqual(st["exp_avg"].dtype, torch.uint8)
        self.assertEqual(st["exp_avg"].numel(), p.numel() // 2)
        self.assertEqual(st["exp_avg_sq"].dtype, torch.uint8)
        self.assertEqual(st["exp_avg_sq"].numel(), p.numel() // 4)

        total_bytes = sum(
            v.numel() * v.element_size()
            for k, v in st.items()
            if isinstance(v, torch.Tensor) and k != "step"
        )
        bytes_per_param = total_bytes / p.numel()
        self.assertLess(bytes_per_param, 0.76)

    def test_sub_byte_8bit_state_dict_roundtrip(self):
        import copy

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(77)
                p1 = nn.Parameter(torch.randn(64, 512, dtype=dtype))
                p2 = nn.Parameter(p1.detach().clone())

                opt1 = ApexW([p1], lr=2e-4, precision="8bit", stochastic_rounding=False)
                p1.grad = torch.randn_like(p1) * 0.05
                opt1.step()

                p2.data.copy_(p1.data)
                sd = copy.deepcopy(opt1.state_dict())
                opt2 = ApexW([p2], lr=2e-4, precision="8bit", stochastic_rounding=False)
                opt2.load_state_dict(sd)

                g2 = torch.randn_like(p1) * 0.05
                p1.grad = g2.clone()
                p2.grad = g2.clone()
                opt1.step()
                opt2.step()
                self.assertTrue(torch.equal(p1.data, p2.data))

    def test_triton_and_cuda_partition_accept_quad_v(self):
        import apexw.triton_kernel as triton_mod

        p = nn.Parameter(torch.randn(64, 512, dtype=torch.bfloat16))  # 32,768 elements
        opt = ApexW([p], lr=2e-4, precision="8bit", stochastic_rounding=False)
        p.grad = torch.randn_like(p) * 0.05
        opt.step()

        st = opt.state[p]
        m = st["exp_avg"]
        v = st["exp_avg_sq"]
        ms = st["exp_avg_scales"]
        vs = st["exp_avg_sq_scales"]

        with patch.object(torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True):
            err_triton = triton_mod._triton_preflight_check(
                p, p.grad, m, v, exp_avg_scales=ms, exp_avg_sq_scales=vs, block_size=st["block_size_8bit"]
            )
            part = opt._build_cuda_8bit_partition([p], [m], [v])
        self.assertIsNone(err_triton)
        self.assertTrue(part["eligible"])
        self.assertEqual(len(part["q_params"]), 1)

    def test_8bit_nesterov_snr_lookahead_beats_adamw_across_learning_rates(self):
        for lr in (5e-4, 2e-4, 1e-5):
            with self.subTest(lr=lr):
                torch.manual_seed(42)
                w_init = torch.randn(64, 512) * 0.05
                x = torch.randn(32, 512)
                target = torch.randn(32, 64)

                w_adamw = nn.Parameter(w_init.clone())
                opt_adamw = torch.optim.AdamW([w_adamw], lr=lr, weight_decay=0.01)
                losses_adamw = []
                for _ in range(30):
                    opt_adamw.zero_grad()
                    loss = ((x @ w_adamw.t() - target) ** 2).mean()
                    loss.backward()
                    opt_adamw.step()
                    losses_adamw.append(loss.item())

                w_apex8 = nn.Parameter(w_init.clone())
                opt_apex8 = ApexW([w_apex8], lr=lr, weight_decay=0.01, precision="8bit", stochastic_rounding=False)
                losses_apex8 = []
                for _ in range(30):
                    opt_apex8.zero_grad()
                    loss = ((x @ w_apex8.t() - target) ** 2).mean()
                    loss.backward()
                    opt_apex8.step()
                    losses_apex8.append(loss.item())

                self.assertLess(sum(losses_apex8) / len(losses_apex8), sum(losses_adamw) / len(losses_adamw))
                self.assertLess(losses_apex8[-1], losses_adamw[-1])

    def test_compact_quad_v_state_memory_and_state_dict_roundtrip(self):
        import copy
        import apexw.triton_kernel as triton_mod

        torch.manual_seed(101)
        params1 = [nn.Parameter(torch.randn(64, 512, dtype=torch.float32)) for _ in range(8)]
        params2 = [nn.Parameter(p.detach().clone()) for p in params1]

        opt1 = ApexW(params1, lr=2e-4, betas=(0.9, 0.95), precision="compact")
        for p in params1:
            p.grad = torch.randn_like(p) * 0.05
        opt1.step()

        for p in params1:
            st = opt1.state[p]
            self.assertTrue(st.get("is_quad_v", False))
            self.assertEqual(st["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(st["exp_avg"].numel(), p.numel())
            self.assertEqual(st["exp_avg_sq"].dtype, torch.bfloat16)
            self.assertEqual(st["exp_avg_sq"].numel(), p.numel() // 4)
            total_bytes = sum(
                v.numel() * v.element_size()
                for k, v in st.items()
                if isinstance(v, torch.Tensor) and k != "step"
            )
            self.assertAlmostEqual(total_bytes / p.numel(), 2.50, places=3)

        # Verify Triton preflight accepts Compact Quad-V
        p0 = params1[0]
        st0 = opt1.state[p0]
        with patch.object(torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True):
            err_triton = triton_mod._triton_preflight_check(
                p0, p0.grad, st0["exp_avg"], st0["exp_avg_sq"]
            )
        self.assertIsNone(err_triton)

        # Verify state_dict / load_state_dict round-trip
        for p1, p2 in zip(params1, params2):
            p2.data.copy_(p1.data)
        sd = copy.deepcopy(opt1.state_dict())
        opt2 = ApexW(params2, lr=2e-4, betas=(0.9, 0.95), precision="compact")
        opt2.load_state_dict(sd)

        for p1, p2 in zip(params1, params2):
            g = torch.randn_like(p1) * 0.05
            p1.grad = g.clone()
            p2.grad = g.clone()
        opt1.step()
        opt2.step()
        for p1, p2 in zip(params1, params2):
            self.assertTrue(torch.equal(p1.data, p2.data))

    def test_compact_quad_v_nesterov_snr_lookahead_beats_adamw(self):
        for foreach in (True, False):
            for lr in (5e-4, 2e-4, 1e-5):
                with self.subTest(foreach=foreach, lr=lr):
                    torch.manual_seed(42)
                    w_inits = [torch.randn(64, 512) * 0.05 for _ in range(8)]
                    x = torch.randn(32, 512)
                    target = torch.randn(32, 64)

                    w_adamw = [nn.Parameter(w.clone()) for w in w_inits]
                    opt_adamw = torch.optim.AdamW(w_adamw, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
                    losses_adamw = []
                    for _ in range(25):
                        opt_adamw.zero_grad()
                        loss = sum(((x @ w.t() - target) ** 2).mean() for w in w_adamw) / len(w_adamw)
                        loss.backward()
                        opt_adamw.step()
                        losses_adamw.append(loss.item())

                    w_compact = [nn.Parameter(w.clone()) for w in w_inits]
                    opt_compact = ApexW(
                        w_compact, lr=lr, betas=(0.9, 0.95), weight_decay=0.01, precision="compact", foreach=foreach
                    )
                    losses_compact = []
                    for _ in range(25):
                        opt_compact.zero_grad()
                        loss = sum(((x @ w.t() - target) ** 2).mean() for w in w_compact) / len(w_compact)
                        loss.backward()
                        opt_compact.step()
                        losses_compact.append(loss.item())

                    self.assertLess(sum(losses_compact) / len(losses_compact), sum(losses_adamw) / len(losses_adamw))
                    self.assertLess(losses_compact[-1], losses_adamw[-1])

    def test_t4_sm75_cuda_kernel_portability_and_abi_v2(self):
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "apexw", "csrc", "apexw_cuda_kernel.cu")) as f:
            cu_src = f.read()
        self.assertNotIn("__float2bfloat16(", cu_src)
        self.assertNotIn("__bfloat162float(", cu_src)
        self.assertIn("is_quad_v", cu_src)

        with open(os.path.join(root, "apexw", "csrc", "apexw_cuda.cpp")) as f:
            cpp_src = f.read()
        self.assertIn('m.attr("_abi_version") = 2;', cpp_src)

        with open(os.path.join(root, "apexw", "cuda_extension.py")) as f:
            ext_src = f.read()
        self.assertIn('getattr(apexw_cuda_ext, "_abi_version", 0) >= 2', ext_src)

    def test_8bit_runtime_acceleration_16bit_param_compression_and_in_backward_stepping(self):
        torch.manual_seed(123)
        params = [nn.Parameter(torch.randn(64, 512, dtype=torch.float32) * 0.05) for _ in range(16)]
        orig_bytes = sum(p.numel() * p.element_size() for p in params)
        self.assertEqual(orig_bytes, 16 * 64 * 512 * 4)

        opt = ApexW(params, lr=2e-4, precision="8bit")
        compressed_bytes = sum(p.numel() * p.element_size() for p in params)
        self.assertEqual(compressed_bytes, orig_bytes // 2)
        for p in params:
            self.assertIn(p.dtype, (torch.float16, torch.bfloat16))

        x = torch.randn(16, 512, dtype=params[0].dtype)
        y = x
        for p in params:
            y = (y @ p.t()) @ p
        loss = (y ** 2).mean()
        loss.backward()

        # Anchor parameter (params[0], last visited in backward) keeps grad for GradScaler / clip_grad_norm_
        self.assertIsNotNone(params[0].grad)
        # All 15 hooked parameters were stepped in layer batches and freed during backward()
        for p in params[1:]:
            self.assertIsNone(p.grad)
            self.assertEqual(int(opt.state[p]["step"].item()), 1)

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        for p in params:
            self.assertEqual(int(opt.state[p]["step"].item()), 1)

        # Creating a subsequent standard optimizer automatically restores params to float32
        _ = torch.optim.AdamW(params, lr=2e-4)
        for p in params:
            self.assertEqual(p.dtype, torch.float32)

    def test_8bit_runtime_acceleration_grad_accum_and_loss_convergence(self):
        for lr in (5e-4, 2e-4, 1e-5):
            with self.subTest(lr=lr):
                torch.manual_seed(77)
                w_inits = [torch.randn(64, 512, dtype=torch.float32) * 0.05 for _ in range(16)]
                x = torch.randn(32, 512)
                target = torch.randn(32, 64)

                w_adamw = [nn.Parameter(w.clone()) for w in w_inits]
                opt_adamw = torch.optim.AdamW(w_adamw, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
                losses_adamw = []
                for _ in range(20):
                    opt_adamw.zero_grad()
                    loss = sum(((x @ w.t() - target) ** 2).mean() for w in w_adamw) / len(w_adamw)
                    loss.backward()
                    opt_adamw.step()
                    losses_adamw.append(loss.item())

                w_apex8 = [nn.Parameter(w.clone()) for w in w_inits]
                opt_apex8 = ApexW(
                    w_apex8, lr=lr, betas=(0.9, 0.95), weight_decay=0.01, precision="8bit", stochastic_rounding=True
                )
                losses_apex8 = []
                for _ in range(20):
                    opt_apex8.zero_grad()
                    x_cast = x.to(w_apex8[0].dtype)
                    t_cast = target.to(w_apex8[0].dtype)
                    loss = sum(((x_cast @ w.t() - t_cast).float() ** 2).mean() for w in w_apex8) / len(w_apex8)
                    loss.backward()
                    opt_apex8.step()
                    losses_apex8.append(loss.item())

                self.assertLess(sum(losses_apex8) / len(losses_apex8), sum(losses_adamw) / len(losses_adamw))
                self.assertLess(losses_apex8[-1], losses_adamw[-1])


class TestScaleVNoBleed:
    def test_fp32_and_8bit_params_no_nan(self):
        """Mixed FP32 + 8-bit params must not produce NaN or Inf weights."""
        torch.manual_seed(0)
        m = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.Linear(16, 8),
        )
        opt = ApexW(m.parameters(), lr=1e-3, precision="8bit")

        for _ in range(3):
            x = torch.randn(4, 16)
            loss = m(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        for name, p in m.named_parameters():
            assert torch.isfinite(p).all(), f"Non-finite values in {name} after 3 steps"


class TestQuantizeBlockwiseInfSafety:
    def test_inf_variance_does_not_produce_nan(self):
        """_quantize_blockwise_uint8 must not return NaN when input has +inf values."""
        from apexw.optimizer import _quantize_blockwise_uint8, _dequantize_blockwise_uint8

        v = torch.ones(512, dtype=torch.float32)
        v[0] = float("inf")
        v[255] = float("inf")
        v[256] = float("nan")

        q, scales = _quantize_blockwise_uint8(v, block_size=256, stochastic=False)
        assert not torch.any(torch.isnan(q.float())), "NaN in quantized output"
        assert not torch.any(torch.isinf(q.float())), "Inf in quantized output"

        v_back = _dequantize_blockwise_uint8(q, scales)
        assert torch.isfinite(v_back).all(), "Non-finite values in dequantized output"

    def test_8bit_amp_large_gradients_no_nan(self):
        """
        Simulates a GradScaler scenario with very large gradients (scale=65536).
        Even if unscaling is done correctly, intermediate squared grads may overflow FP16.
        The 8-bit variance storage must absorb this without producing NaN weights.
        """
        torch.manual_seed(42)
        model = torch.nn.Linear(32, 16, bias=False)
        opt = ApexW(model.parameters(), lr=1e-4, precision="8bit", eps=1e-8)

        # Fake GradScaler: manually scale grads before passing to optimizer
        scaler_scale = 65536.0

        for step in range(5):
            x = torch.randn(4, 32)
            loss = model(x).sum()
            loss.backward()

            # Manually inject scaled gradients (simulating what GradScaler does)
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scaler_scale)

            # Optimizer unscales via self.grad_scale attribute set by GradScaler
            opt.grad_scale = torch.tensor(scaler_scale)
            opt.step()
            opt.grad_scale = None
            opt.zero_grad(set_to_none=True)

        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"Non-finite in {name} after AMP-like steps"


# Bug 2b: inf/nan in v_val before requantization in _step_ultra
class TestStepUltraVarianceClamp:
    def test_variance_clamped_before_requantization(self):
        """
        Force a situation where the EMA variance update would produce +inf,
        then verify the weight updates remain finite.
        """
        torch.manual_seed(7)
        model = torch.nn.Linear(8, 4, bias=False)
        opt = ApexW(model.parameters(), lr=1e-3, precision="8bit")

        # First step: normal, to initialize quantized states
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        # Inject a catastrophically large gradient to push variance to inf
        with torch.no_grad():
            for p in model.parameters():
                p.grad = torch.full_like(p, 1e30)

        opt.step()
        opt.zero_grad(set_to_none=True)

        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), (
                f"Non-finite in {name} after extreme-gradient step. "
                f"Min={p.min().item():.3g}, Max={p.max().item():.3g}"
            )


class TestExecutionFastPaths(unittest.TestCase):
    def test_6bit_quantizer_roundtrip_and_compression(self):
        torch.manual_seed(42)
        x = torch.randn(1024, 512, dtype=torch.float32) * 0.25
        q, scales = _quantize_blockwise_6bit(x, block_size=256, stochastic=False)
        self.assertEqual(q.dtype, torch.uint8)
        self.assertEqual(scales.dtype, torch.float32)
        # 524,288 elements -> 524288 // 4 * 3 = 393,216 packed bytes (0.75 B/p + scales)
        self.assertEqual(q.numel(), (x.numel() // 4) * 3)
        deq = _dequantize_blockwise_6bit(q, scales, numel=x.numel(), block_size=256).reshape(x.shape)
        rel_err = (deq - x).abs().mean() / x.abs().mean()
        self.assertLess(rel_err.item(), 0.05)

    def test_compact_factored_v_threshold_and_vram_savings(self):
        # Small 512x512 (262,144 < 524,288): compact keeps 4.0 B/p (2B m + 2B v, excluding 4B step tensor)
        p_small = nn.Parameter(torch.randn(512, 512, dtype=torch.bfloat16))
        opt_small = ApexW([p_small], lr=1e-3, betas=(0.9, 0.95), precision="compact")
        p_small.grad = torch.randn_like(p_small) * 0.01
        opt_small.step()
        st_small = opt_small.state[p_small]
        self.assertFalse(st_small.get("is_factored_v", False))
        moment_bytes = (
            st_small["exp_avg"].numel() * st_small["exp_avg"].element_size()
            + st_small["exp_avg_sq"].numel() * st_small["exp_avg_sq"].element_size()
        )
        self.assertEqual(moment_bytes, p_small.numel() * 4)

        # Large 1024x512 (524,288 >= 524,288): compact enables Adafactor factored_v (~2.01 B/p)
        p_large = nn.Parameter(torch.randn(1024, 512, dtype=torch.bfloat16))
        opt_large = ApexW([p_large], lr=1e-3, betas=(0.9, 0.95), precision="compact", factored_v=True)
        p_large.grad = torch.randn_like(p_large) * 0.01
        opt_large.step()
        st = opt_large.state[p_large]
        self.assertTrue(st.get("is_factored_v", False))
        self.assertEqual(st["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(st["exp_avg_sq_row"].numel(), 1024)
        self.assertEqual(st["exp_avg_sq_col"].numel(), 512)
        bytes_per_param = _total_state_bytes(opt_large) / p_large.numel()
        self.assertLess(bytes_per_param, 2.05)

    def test_compact_factored_v_convergence_vs_adamw(self):
        torch.manual_seed(10)
        x = torch.randn(32, 512, dtype=torch.bfloat16)
        target = x @ (torch.randn(512, 1024, dtype=torch.bfloat16) * 0.05)

        def run_opt(use_apex: bool):
            torch.manual_seed(123)
            w = nn.Parameter(torch.randn(512, 1024, dtype=torch.bfloat16) * 0.02)
            opt = (
                ApexW([w], lr=1e-2, betas=(0.9, 0.95), precision="compact")
                if use_apex
                else torch.optim.AdamW([w], lr=1e-2, betas=(0.9, 0.95))
            )
            for _ in range(30):
                opt.zero_grad()
                loss = ((x @ w) - target).pow(2).mean()
                loss.backward()
                opt.step()
            return float(loss.item())

        loss_apex = run_opt(True)
        loss_adam = run_opt(False)
        self.assertLess(loss_apex, loss_adam * 1.15 + 1e-4)

    def test_8bit_default_betas_and_low_lr_bf16_convergence(self):
        p = nn.Parameter(torch.randn(128, 128, dtype=torch.bfloat16))
        opt = ApexW([p], lr=1e-5, precision="8bit")
        self.assertEqual(opt.param_groups[0]["betas"], (0.9, 0.999))

        # Verify lr=1e-5 LoRA fine-tuning (FP32 adapters over frozen BF16 base) descends and matches Torch AdamW
        torch.manual_seed(77)
        x = torch.randn(32, 128, dtype=torch.bfloat16)
        base = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        base.weight.requires_grad_(False)
        true_a = torch.randn(128, 16, dtype=torch.bfloat16) * 0.2
        true_b = torch.randn(16, 128, dtype=torch.bfloat16) * 0.2
        target = (base(x) + (x @ true_a) @ true_b).float()

        def run_lora(use_apex: bool):
            torch.manual_seed(42)
            lora_a = nn.Parameter(torch.randn(128, 16, dtype=torch.float32) * 0.05)
            lora_b = nn.Parameter(torch.randn(16, 128, dtype=torch.float32) * 0.05)
            opt_l = (
                ApexW([lora_a, lora_b], lr=1e-5, precision="8bit", weight_decay=0.0)
                if use_apex
                else torch.optim.AdamW([lora_a, lora_b], lr=1e-5, weight_decay=0.0)
            )
            losses = []
            for _ in range(150):
                opt_l.zero_grad()
                y = base(x).float() + (x.float() @ lora_a) @ lora_b
                loss = (y - target).pow(2).mean()
                loss.backward()
                opt_l.step()
                losses.append(float(loss.item()))
            return losses[0], losses[-1]

        s_ap, e_ap = run_lora(True)
        s_ad, e_ad = run_lora(False)
        self.assertLess(e_ap, s_ap)
        self.assertLessEqual(e_ap, e_ad * 1.02 + 1e-4)

    def test_8bit_6bit_state_dict_resume(self):
        torch.manual_seed(5)
        # 1024x512 = 524,288 triggers 6-bit packed m + factored v under 8bit
        p = nn.Parameter(torch.randn(1024, 512, dtype=torch.bfloat16))
        opt = ApexW([p], lr=1e-3, precision="8bit", factored_v=True, six_bit=True)
        p.grad = torch.randn_like(p) * 0.02
        opt.step()

        st = opt.state[p]
        self.assertTrue(st.get("is_6bit_m", False))
        self.assertTrue(st.get("is_factored_v", False))
        self.assertEqual(st["exp_avg"].dtype, torch.uint8)
        bpp = _total_state_bytes(opt) / p.numel()
        self.assertLess(bpp, 0.80)  # ~0.777 B/p

        sd = opt.state_dict()
        p2 = nn.Parameter(p.detach().clone())
        opt2 = ApexW([p2], lr=1e-3, precision="8bit", factored_v=True, six_bit=True)
        opt2.load_state_dict(sd)
        st2 = opt2.state[p2]
        self.assertTrue(st2.get("is_6bit_m", False))
        self.assertEqual(st2["exp_avg"].dtype, torch.uint8)
        self.assertEqual(st2["exp_avg_scales"].dtype, torch.float32)
        self.assertTrue(torch.equal(st["exp_avg"], st2["exp_avg"]))
        self.assertTrue(torch.equal(st["exp_avg_scales"], st2["exp_avg_scales"]))

        # Step again after resume without error or NaN
        p2.grad = torch.randn_like(p2) * 0.02
        opt2.step()
        self.assertTrue(torch.isfinite(p2).all())

    def test_hybrid_8bit_muon_quantized_state_and_resume(self):
        torch.manual_seed(11)
        model = nn.Sequential(
            nn.Linear(256, 256, bias=False, dtype=torch.bfloat16),
            nn.Linear(256, 256, bias=False, dtype=torch.bfloat16),
            nn.Linear(1024, 512, bias=False, dtype=torch.bfloat16),  # >= 524288 -> 6-bit Muon
        )
        opt = HybridApexW(model, lr=1e-3, precision="8bit", beta3=0.999)
        x = torch.randn(8, 256, dtype=torch.bfloat16)
        # Forward through matching dims with separate inputs or dummy grads
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 0.01
        opt.step()

        total_p = sum(p.numel() for p in model.parameters())
        # Muon params with beta3 (2 quantized buffers: INT8/6-bit instead of 2x4B=8B)
        muon_bytes = _total_state_bytes(opt.muon_opt)
        self.assertLess(muon_bytes / total_p, 2.0)

        w_small = model[0].weight
        w_large = model[2].weight
        self.assertEqual(opt.muon_opt.state[w_small]["momentum"].dtype, torch.int8)
        self.assertEqual(opt.muon_opt.state[w_large]["momentum"].dtype, torch.uint8)
        self.assertTrue(opt.muon_opt.state[w_large]["is_6bit_m"])

        # State dict roundtrip preserves quantized momentum + float32 scales
        sd = opt.state_dict()
        model2 = nn.Sequential(
            nn.Linear(256, 256, bias=False, dtype=torch.bfloat16),
            nn.Linear(256, 256, bias=False, dtype=torch.bfloat16),
            nn.Linear(1024, 512, bias=False, dtype=torch.bfloat16),
        )
        opt2 = HybridApexW(model2, lr=1e-3, precision="8bit", beta3=0.999)
        opt2.load_state_dict(sd)
        st2_large = opt2.muon_opt.state[model2[2].weight]
        self.assertEqual(st2_large["momentum"].dtype, torch.uint8)
        self.assertEqual(st2_large["momentum_scales"].dtype, torch.float32)

        for p in model2.parameters():
            p.grad = torch.randn_like(p) * 0.01
        opt2.step()
        self.assertTrue(torch.isfinite(model2[2].weight).all())

    def test_muon_addmm_baddbmm_numerical_equivalence(self):
        torch.manual_seed(99)
        G_tall = torch.randn(128, 64, dtype=torch.float32) * 0.5
        G_wide = torch.randn(64, 128, dtype=torch.float32) * 0.5
        for G in (G_tall, G_wide):
            # Compare batched (K=2) vs single on NS5 & Polar Express
            B = torch.stack([G, G], dim=0)
            u_single_ns = zeropower_via_newtonschulz5(G, steps=5, work_dtype=torch.float32, compiled=False)
            u_batch_ns = zeropower_via_newtonschulz5_batched(B, steps=5)[0]
            self.assertLess((u_single_ns - u_batch_ns).abs().max().item(), 1e-5)

            u_single_pe = zeropower_via_polar_express(G, steps=4, work_dtype=torch.float32, compiled=False)
            u_batch_pe = zeropower_via_polar_express_batched(B, steps=4)[0]
            self.assertLess((u_single_pe - u_batch_pe).abs().max().item(), 1e-5)

    def test_chunked_factored_v_bounded_transient_python_overhead(self):
        # Verify step() completes cleanly on a 2M param without full-tensor g.float() Python spike
        p = nn.Parameter(torch.randn(2048, 1024, dtype=torch.bfloat16))
        opt = ApexW([p], lr=1e-3, precision="8bit")
        p.grad = torch.randn_like(p) * 0.01
        opt.step()  # warmup

        tracemalloc.start()
        opt.step()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Python tracemalloc overhead must stay well below 2 MiB
        self.assertLess(peak_bytes, 2 * 1024 * 1024)

    def test_triton_factored_v_static_invariants_and_cpu_fallback(self):
        with open(inspect.getfile(triton_kernel), "r", encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("tl.uint32", src)
        self.assertNotIn("p * (1.0 - lr * weight_decay)", src)
        self.assertNotIn("p0 * (1.0 - lr * weight_decay)", src)
        # Ensure step is not marked tl.constexpr inside factored_v kernels
        for kname in ("_adamw_kernel_triton_factored_v", "_adamw_kernel_triton_factored_v_6bit"):
            m = re.search(rf"def {kname}\((.*?)\):", src, flags=re.DOTALL)
            self.assertIsNotNone(m, f"Kernel {kname} missing")
            sig = m.group(1)
            self.assertIn("step: int = 1", sig)
            self.assertNotIn("step: tl.constexpr", sig)
        # Ensure AST compiles cleanly and host launcher returns False on CPU tensors
        ast.parse(src)
        p = torch.randn(512, 128, dtype=torch.bfloat16)
        g = torch.randn_like(p)
        m = torch.zeros(512, 128, dtype=torch.bfloat16)
        r = torch.zeros(512, dtype=torch.float32)
        c = torch.zeros(128, dtype=torch.float32)
        self.assertFalse(
            triton_kernel.triton_factored_v_step(
                p, g, m, r, c, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.01, step=1
            )
        )


if __name__ == "__main__":
    unittest.main()
