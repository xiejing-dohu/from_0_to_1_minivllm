from __future__ import annotations
from dataclasses import dataclass
import torch
from day10.engine.sequence import Sequence


@dataclass
class PreparedBatch:
    input_ids:torch.Tensor; positions:torch.Tensor; slot_mapping:torch.Tensor
    cu_seqlens_q:torch.Tensor|None=None; cu_seqlens_k:torch.Tensor|None=None
    context_lens:torch.Tensor|None=None; block_tables:torch.Tensor|None=None
    max_seqlen_q:int=0; max_seqlen_k:int=0


class ModelRunner:
    def __init__(self,block_size:int,device:torch.device|str="cpu"):
        self.block_size=block_size; self.device=torch.device(device)
    def _tensor(self,data,dtype): return torch.tensor(data,dtype=dtype,device=self.device)
    def _tables(self,seqs):
        width=max((len(s.block_table) for s in seqs),default=0)
        return self._tensor([s.block_table+[-1]*(width-len(s.block_table)) for s in seqs],torch.long) if width else None
    def _slot(self,seq,position):
        physical=seq.block_table[position//self.block_size]
        return physical*self.block_size+position%self.block_size
    def prepare_prefill(self,seqs:list[Sequence])->PreparedBatch:
        if not seqs: raise ValueError("Prefill batch cannot be empty")
        ids=[]; positions=[]; slots=[]; qlens=[]; klens=[]
        for seq in seqs:
            if not seq.block_table or seq.num_cached_tokens>seq.num_tokens or seq.num_cached_tokens%self.block_size:
                raise ValueError("sequence has invalid allocation/cache metadata")
            start=seq.num_cached_tokens; new=seq.token_ids[start:]
            if not new: raise ValueError("Prefill sequence must contain at least one uncached token")
            ids.extend(new); positions.extend(range(start,seq.num_tokens)); slots.extend(self._slot(seq,p) for p in range(start,seq.num_tokens))
            qlens.append(len(new)); klens.append(seq.num_tokens)
        cuq=[0]; cuk=[0]
        for q,k in zip(qlens,klens): cuq.append(cuq[-1]+q); cuk.append(cuk[-1]+k)
        return PreparedBatch(self._tensor(ids,torch.long),self._tensor(positions,torch.long),self._tensor(slots,torch.long),self._tensor(cuq,torch.int32),self._tensor(cuk,torch.int32),block_tables=self._tables(seqs),max_seqlen_q=max(qlens),max_seqlen_k=max(klens))
    def prepare_decode(self,seqs:list[Sequence])->PreparedBatch:
        if not seqs or any(not s.token_ids or len(s.block_table)!=s.num_blocks for s in seqs): raise ValueError("Decode sequences must be non-empty and fully allocated")
        ids=[s.last_token for s in seqs]; positions=[s.num_tokens-1 for s in seqs]; slots=[self._slot(s,s.num_tokens-1) for s in seqs]
        return PreparedBatch(self._tensor(ids,torch.long),self._tensor(positions,torch.long),self._tensor(slots,torch.long),context_lens=self._tensor([s.num_tokens for s in seqs],torch.int32),block_tables=self._tables(seqs))


class CudaGraphRunner:
    """Capture a fixed-shape decode callable and safely replay smaller batches."""
    def __init__(self,fn,max_batch:int,max_blocks:int,device="cuda"):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA Graph requires CUDA")
        self.fn=fn; self.max_batch=max_batch; self.device=torch.device(device)
        self.ids=torch.zeros(max_batch,dtype=torch.long,device=device); self.slots=torch.full((max_batch,),-1,dtype=torch.long,device=device)
        self.lens=torch.zeros(max_batch,dtype=torch.int32,device=device); self.tables=torch.full((max_batch,max_blocks),-1,dtype=torch.long,device=device)
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): self.output=fn(self.ids,self.slots,self.lens,self.tables)
        torch.cuda.current_stream().wait_stream(stream); self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self.output=fn(self.ids,self.slots,self.lens,self.tables)
    def replay(self,batch:PreparedBatch):
        size=batch.input_ids.numel()
        if size>self.max_batch or batch.context_lens is None or batch.block_tables is None or batch.block_tables.shape[1]>self.tables.shape[1]: raise ValueError("batch does not fit captured graph")
        self.ids.zero_(); self.slots.fill_(-1); self.lens.zero_(); self.tables.fill_(-1)
        self.ids[:size].copy_(batch.input_ids); self.slots[:size].copy_(batch.slot_mapping); self.lens[:size].copy_(batch.context_lens); self.tables[:size,:batch.block_tables.shape[1]].copy_(batch.block_tables)
        self.graph.replay(); return self.output[:size].clone()
