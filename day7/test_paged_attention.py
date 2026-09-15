import unittest
import torch
from day7.layers.paged_attention import paged_attention_decode, paged_attention_reference


class PagedAttentionTests(unittest.TestCase):
    def make_inputs(self, device="cpu", dtype=torch.float32, dim=12):
        torch.manual_seed(2026)
        query = torch.randn(3, 4, dim, device=device, dtype=dtype)
        kc = torch.randn(9, 4, 2, dim, device=device, dtype=dtype)
        vc = torch.randn_like(kc)
        tables = torch.tensor([[7, 1, 5], [2, 8, -1], [6, 0, 4]], device=device, dtype=torch.long)
        lens = torch.tensor([10, 6, 9], device=device, dtype=torch.int32)
        return query, kc, vc, tables, lens

    def test_torch_path_matches_reference(self):
        args = self.make_inputs()
        torch.testing.assert_close(paged_attention_decode(*args, backend="torch"), paged_attention_reference(*args))

    def test_random_table_differs_from_sequential_table(self):
        q, kc, vc, tables, lens = self.make_inputs()
        actual = paged_attention_reference(q, kc, vc, tables, lens)
        sequential = torch.tensor([[0,1,2],[3,4,-1],[5,6,7]], dtype=torch.long)
        self.assertFalse(torch.allclose(actual, paged_attention_reference(q, kc, vc, sequential, lens)))

    def test_invalid_used_block_and_zero_length_are_rejected(self):
        q, kc, vc, tables, lens = self.make_inputs()
        bad = tables.clone(); bad[0, 1] = -1
        with self.assertRaises(ValueError): paged_attention_decode(q, kc, vc, bad, lens)
        with self.assertRaises(ValueError): paged_attention_decode(q, kc, vc, tables, torch.tensor([10,0,9]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_triton_gqa_matches_reference(self):
        args = self.make_inputs("cuda", torch.float16, 32)
        actual = paged_attention_decode(*args, backend="triton")
        expected = paged_attention_reference(*args)
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


if __name__ == "__main__": unittest.main()
