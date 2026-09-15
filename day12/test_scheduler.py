import unittest
from day10.engine import SamplingParams,Sequence,SequenceStatus
from day12.scheduler import Scheduler


class SchedulerTests(unittest.TestCase):
    def test_prefill_obeys_token_and_sequence_budgets(self):
        s=Scheduler(2,6,8,4,99); a=Sequence([1,2,3,4],4); b=Sequence([5,6,7],4); s.add_sequence(a); s.add_sequence(b)
        batch,prefill=s.schedule(); self.assertTrue(prefill); self.assertEqual(batch,[a]); self.assertEqual(list(s.waiting),[b])
    def test_oversized_prompt_fails_immediately(self):
        s=Scheduler(2,8,2,4,99)
        with self.assertRaisesRegex(ValueError,"never fit"): s.add_sequence(Sequence(list(range(9)),4))
        with self.assertRaisesRegex(ValueError,"no room"): s.add_sequence(Sequence([1,2,3],4,SamplingParams(max_model_length=3)))
    def test_decode_then_stop_at_max_tokens_releases_blocks(self):
        s=Scheduler(2,8,4,4,99); q=Sequence([1,2],4,SamplingParams(max_tokens=1)); s.add_sequence(q); batch,_=s.schedule(); s.postprocess(batch,[7])
        self.assertTrue(q.is_finished); self.assertTrue(s.is_finished); self.assertEqual(len(s.block_manager.free),4)
    def test_eos_and_ignore_eos(self):
        s=Scheduler(2,8,4,4,9); a=Sequence([1],4); b=Sequence([2],4,SamplingParams(ignore_eos=True,max_tokens=3)); s.add_sequence(a); s.add_sequence(b); batch,_=s.schedule(); s.postprocess(batch,[9,9])
        self.assertTrue(a.is_finished); self.assertEqual(b.status,SequenceStatus.RUNNING)
    def test_preemption_frees_blocks_without_leak(self):
        s=Scheduler(2,8,2,4,99); a=Sequence([1,2,3,4],4); b=Sequence([5,6,7,8],4); s.add_sequence(a); s.add_sequence(b); batch,_=s.schedule()
        s.postprocess(batch,[10,11])
        decode,prefill=s.schedule(); self.assertFalse(prefill); self.assertEqual(len(decode),1); self.assertEqual(len(s.waiting),1); s.block_manager.assert_invariants(list(s.running))


if __name__=="__main__": unittest.main()
