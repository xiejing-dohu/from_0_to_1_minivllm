import unittest
from day10.engine import BlockManager,Sequence


class BlockManagerTests(unittest.TestCase):
    def test_sequence_boundaries_and_input_copy(self):
        ids=[1,2,3,4]; s=Sequence(ids,4); ids.append(5)
        self.assertEqual(s.num_blocks,1); self.assertEqual(s.last_block_num_tokens,4); self.assertEqual(s.token_ids,[1,2,3,4])
        e=Sequence([],4); self.assertEqual((e.num_blocks,e.last_block_num_tokens),(0,0))

    def test_allocate_append_crosses_boundary(self):
        m=BlockManager(4,4); s=Sequence([1,2,3,4],4); m.allocate(s); first=s.block_table[0]
        s.append_token(5); self.assertTrue(m.can_append(s)); m.append(s)
        self.assertEqual(len(s.block_table),2); self.assertNotEqual(first,s.block_table[1]); m.assert_invariants([s])

    def test_cross_sequence_prefix_reuse_and_refcounts(self):
        m=BlockManager(5,4); a=Sequence([1,2,3,4,9],4); m.allocate(a)
        b=Sequence([1,2,3,4,8],4); m.allocate(b)
        self.assertEqual(a.block_table[0],b.block_table[0]); self.assertEqual(b.num_cached_tokens,4)
        shared=a.block_table[0]; self.assertEqual(m.blocks[shared].ref_count,2)
        m.deallocate(a); self.assertEqual(m.blocks[shared].ref_count,1); m.deallocate(b); self.assertEqual(m.blocks[shared].ref_count,0)

    def test_cached_free_block_can_be_reclaimed(self):
        m=BlockManager(2,4); a=Sequence([1,2,3,4],4); m.allocate(a); block=a.block_table[0]; m.deallocate(a)
        b=Sequence([1,2,3,4,5],4); m.allocate(b); self.assertEqual(b.block_table[0],block); self.assertEqual(b.num_cached_tokens,4)

    def test_same_tokens_after_different_prefix_do_not_alias(self):
        m=BlockManager(6,2); a=Sequence([1,2,7,8],2); b=Sequence([3,4,7,8],2); m.allocate(a); m.allocate(b)
        self.assertNotEqual(a.block_table[1],b.block_table[1])

    def test_hash_collision_still_compares_tokens(self):
        m=BlockManager(4,2); m.compute_hash=lambda tokens,prefix:b"collision"
        a=Sequence([1,2],2); b=Sequence([8,9],2); m.allocate(a); m.allocate(b)
        self.assertNotEqual(a.block_table,b.block_table)
        c=Sequence([8,9,10],2); m.allocate(c); self.assertEqual(c.block_table[0],b.block_table[0]); self.assertEqual(c.num_cached_tokens,2)

    def test_insufficient_capacity_and_deallocation(self):
        m=BlockManager(2,4); s=Sequence(list(range(9)),4); self.assertFalse(m.can_allocate(s))
        with self.assertRaises(RuntimeError): m.allocate(s)
        a=Sequence([1,2,3],4); m.allocate(a); m.deallocate(a); self.assertEqual(len(m.free),2); m.assert_invariants()

    def test_free_cached_hit_is_counted_in_capacity(self):
        m=BlockManager(1,2); seed=Sequence([1,2],2); m.allocate(seed); m.deallocate(seed)
        request=Sequence([1,2,3],2)
        self.assertFalse(m.can_allocate(request))
        with self.assertRaises(RuntimeError): m.allocate(request)


if __name__=="__main__": unittest.main()
