import unittest

import torch

from day6.layers.prefill_attention import prefill_attention, prefill_attention_reference


class PrefillAttentionTests(unittest.TestCase):
    def make_inputs(self, device="cpu", dtype=torch.float32, prefix=False):
        torch.manual_seed(2026)
        q_lengths = [3, 5, 2]
        k_lengths = [5, 6, 4] if prefix else q_lengths
        q = torch.randn(sum(q_lengths), 4, 8, device=device, dtype=dtype)
        k = torch.randn(sum(k_lengths), 2, 8, device=device, dtype=dtype)
        v = torch.randn_like(k)
        cu_q = torch.tensor([0, 3, 8, 10], device=device, dtype=torch.int32)
        cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
        return q, k, v, cu_q, cu_k

    def test_packed_gqa_matches_reference(self):
        args = self.make_inputs()
        torch.testing.assert_close(prefill_attention(*args), prefill_attention_reference(*args), rtol=1e-5, atol=1e-6)

    def test_prefix_uses_bottom_right_causal_alignment(self):
        args = self.make_inputs(prefix=True)
        torch.testing.assert_close(prefill_attention(*args), prefill_attention_reference(*args), rtol=1e-5, atol=1e-6)

    def test_sequences_do_not_attend_across_boundaries(self):
        q, k, v, cu_q, cu_k = self.make_inputs()
        baseline = prefill_attention(q, k, v, cu_q, cu_k)
        k[3:] += 1000
        v[3:] += 1000
        changed = prefill_attention(q, k, v, cu_q, cu_k)
        torch.testing.assert_close(changed[:3], baseline[:3])
        self.assertFalse(torch.allclose(changed[3:], baseline[3:]))

    def test_invalid_gqa_and_lengths_are_rejected(self):
        q, k, v, cu_q, cu_k = self.make_inputs()
        with self.assertRaises(ValueError):
            prefill_attention(q[:, :3], k, v, cu_q, cu_k)
        with self.assertRaises(ValueError):
            prefill_attention(q, k, v, torch.tensor([0, 4, 8, 10]), cu_k)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_half_matches_reference(self):
        args = self.make_inputs("cuda", torch.float16, prefix=True)
        torch.testing.assert_close(prefill_attention(*args), prefill_attention_reference(*args), rtol=3e-3, atol=3e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_triton_flash_kernel_matches_reference(self):
        torch.manual_seed(7)
        q = torch.randn(12, 4, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(12, 2, 32, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        cu = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
        actual = prefill_attention(q, k, v, cu, cu, backend="triton")
        expected = prefill_attention_reference(q, k, v, cu, cu)
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    unittest.main()
