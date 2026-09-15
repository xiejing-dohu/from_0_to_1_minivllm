import unittest

import torch

from day5.layers.kv_cache import store_kv_cache, store_kv_cache_torch


class KVCacheWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.num_blocks = 5
        self.block_size = 4
        self.num_heads = 2
        self.head_dim = 6

    def make_tensors(self, device: str = "cpu"):
        key = torch.randn(6, self.num_heads, self.head_dim, device=device)
        value = torch.randn_like(key)
        k_cache = torch.full(
            (self.num_blocks, self.block_size, self.num_heads, self.head_dim),
            -99.0,
            device=device,
        )
        v_cache = torch.full_like(k_cache, -77.0)
        slots = torch.tensor([17, 2, -1, 11, 4, 19], device=device, dtype=torch.long)
        return key, value, k_cache, v_cache, slots

    def assert_written_correctly(self, key, value, k_cache, v_cache, slots) -> None:
        flat_k = k_cache.flatten(0, 1)
        flat_v = v_cache.flatten(0, 1)
        written = set()
        for token, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            written.add(slot)
            torch.testing.assert_close(flat_k[slot], key[token])
            torch.testing.assert_close(flat_v[slot], value[token])
        for slot in set(range(self.num_blocks * self.block_size)) - written:
            torch.testing.assert_close(flat_k[slot], torch.full_like(flat_k[slot], -99.0))
            torch.testing.assert_close(flat_v[slot], torch.full_like(flat_v[slot], -77.0))

    def test_random_physical_slots_and_unwritten_positions(self) -> None:
        tensors = self.make_tensors()
        store_kv_cache_torch(*tensors, block_size=self.block_size)
        self.assert_written_correctly(*tensors)

    def test_slot_maps_to_block_and_offset(self) -> None:
        key = torch.tensor([[[3.0, 4.0]]])
        value = torch.tensor([[[5.0, 6.0]]])
        k_cache = torch.zeros(3, 4, 1, 2)
        v_cache = torch.zeros_like(k_cache)
        slots = torch.tensor([9], dtype=torch.long)
        store_kv_cache_torch(key, value, k_cache, v_cache, slots)
        torch.testing.assert_close(k_cache[2, 1], key[0])
        torch.testing.assert_close(v_cache[2, 1], value[0])

    def test_minus_one_skips_write(self) -> None:
        key = torch.randn(2, 1, 3)
        value = torch.randn_like(key)
        k_cache = torch.ones(2, 2, 1, 3)
        v_cache = torch.ones_like(k_cache)
        slots = torch.tensor([-1, -1], dtype=torch.long)
        store_kv_cache(key, value, k_cache, v_cache, slots)
        torch.testing.assert_close(k_cache, torch.ones_like(k_cache))
        torch.testing.assert_close(v_cache, torch.ones_like(v_cache))

    def test_invalid_mappings_are_rejected(self) -> None:
        key, value, k_cache, v_cache, _ = self.make_tensors()
        for slots in (
            torch.tensor([0, 1, 2, 3, 4, 20], dtype=torch.long),
            torch.tensor([0, 1, 2, 3, 4, -2], dtype=torch.long),
            torch.tensor([0, 1, 2, 3, 4, 4], dtype=torch.long),
        ):
            with self.assertRaises(ValueError):
                store_kv_cache(key, value, k_cache, v_cache, slots)
        with self.assertRaises(TypeError):
            store_kv_cache(key, value, k_cache, v_cache, torch.arange(6, dtype=torch.int32))

    def test_shape_and_block_size_are_checked(self) -> None:
        key, value, k_cache, v_cache, slots = self.make_tensors()
        with self.assertRaises(ValueError):
            store_kv_cache(key[:, :1], value, k_cache, v_cache, slots)
        with self.assertRaises(ValueError):
            store_kv_cache(key, value, k_cache, v_cache, slots, block_size=8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_triton_matches_reference_on_cuda(self) -> None:
        tensors = self.make_tensors("cuda")
        key, value, k_cache, v_cache, slots = tensors
        ref_k, ref_v = k_cache.clone(), v_cache.clone()
        store_kv_cache_torch(key, value, ref_k, ref_v, slots)
        store_kv_cache(key, value, k_cache, v_cache, slots, backend="triton")
        torch.cuda.synchronize()
        torch.testing.assert_close(k_cache, ref_k)
        torch.testing.assert_close(v_cache, ref_v)


if __name__ == "__main__":
    unittest.main()
