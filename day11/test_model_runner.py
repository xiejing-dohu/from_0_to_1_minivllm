import unittest
import struct
import torch
import multiprocessing as mp
from day10.engine import BlockManager,Sequence
from day11.model_runner import CudaGraphRunner,ModelRunner
from day11.kv_capacity import available_kv_blocks,kv_block_bytes
from day11.shared_memory_rpc import SharedMemoryMailbox,increment_int64_tensor_worker


class RunnerTests(unittest.TestCase):
    def allocated(self,tokens,blocks,manager=None):
        s=Sequence(tokens,4); s.block_table=list(blocks); return s
    def test_prefill_tensor_contract_with_cached_prefix(self):
        a=self.allocated([1,2,3,4,5,6],[7,2]); a.num_cached_tokens=4
        b=self.allocated([8,9,10],[5])
        x=ModelRunner(4).prepare_prefill([a,b])
        self.assertEqual(x.input_ids.tolist(),[5,6,8,9,10]); self.assertEqual(x.positions.tolist(),[4,5,0,1,2])
        self.assertEqual(x.slot_mapping.tolist(),[8,9,20,21,22]); self.assertEqual(x.cu_seqlens_q.tolist(),[0,2,5]); self.assertEqual(x.cu_seqlens_k.tolist(),[0,6,9])
        self.assertEqual(x.block_tables.tolist(),[[7,2],[5,-1]])
    def test_decode_contract_and_physical_slot(self):
        a=self.allocated([1,2,3,4,5],[7,2]); b=self.allocated([8,9,10],[5])
        x=ModelRunner(4).prepare_decode([a,b])
        self.assertEqual(x.input_ids.tolist(),[5,10]); self.assertEqual(x.positions.tolist(),[4,2]); self.assertEqual(x.slot_mapping.tolist(),[8,22]); self.assertEqual(x.context_lens.tolist(),[5,3])
    def test_invalid_metadata_is_rejected(self):
        with self.assertRaises(ValueError): ModelRunner(4).prepare_decode([self.allocated([1,2,3,4,5],[2])])
    def test_kv_capacity_counts_layers_k_and_v(self):
        shape=dict(num_layers=28,block_size=16,num_kv_heads=8,head_dim=128,dtype_bytes=2)
        self.assertEqual(kv_block_bytes(**shape),2*28*16*8*128*2)
        self.assertEqual(available_kv_blocks(1_000_000,100_000,0.9,50_000,**shape),750_000//kv_block_bytes(**shape))
    def test_two_process_shared_memory_roundtrip(self):
        ctx=mp.get_context("spawn"); request,response=ctx.Event(),ctx.Event(); box=SharedMemoryMailbox(4096)
        process=ctx.Process(target=increment_int64_tensor_worker,args=(box.name,box.capacity,request,response)); process.start()
        try:
            source=torch.tensor([[40,41]],dtype=torch.int64)
            box.write({"dtype":"int64","shape":list(source.shape),"data":source.numpy().tobytes()}); request.set(); self.assertTrue(response.wait(30)); message=box.read()
            values=struct.unpack("<qq",message["data"]); torch.testing.assert_close(torch.tensor(values).reshape(message["shape"]),source+1)
            process.join(30); self.assertEqual(process.exitcode,0)
        finally:
            if process.is_alive(): process.terminate(); process.join()
            box.close(); box.unlink()
    @unittest.skipUnless(torch.cuda.is_available(),"CUDA is unavailable")
    def test_cuda_graph_matches_eager_for_smaller_batch(self):
        def fn(ids,slots,lens,tables): return ids.float()+slots.float()+lens.float()+tables[:,0].float()
        graph=CudaGraphRunner(fn,4,3); a=self.allocated([1,2,3,4,5],[7,2]); b=self.allocated([8,9,10],[5])
        batch=ModelRunner(4,"cuda").prepare_decode([a,b]); actual=graph.replay(batch); expected=fn(batch.input_ids,batch.slot_mapping,batch.context_lens,batch.block_tables)
        torch.testing.assert_close(actual,expected)


if __name__=="__main__": unittest.main()
