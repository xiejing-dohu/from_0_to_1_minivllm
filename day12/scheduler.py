from collections import deque
from day10.engine import BlockManager,Sequence,SequenceStatus


class Scheduler:
    def __init__(self,max_num_sequences:int,max_num_batched_tokens:int,max_cached_blocks:int,block_size:int,eos_token_id:int):
        if min(max_num_sequences,max_num_batched_tokens,max_cached_blocks,block_size)<=0: raise ValueError("scheduler limits must be positive")
        self.max_num_sequences=max_num_sequences; self.max_num_batched_tokens=max_num_batched_tokens; self.eos_token_id=eos_token_id
        self.block_manager=BlockManager(max_cached_blocks,block_size); self.waiting=deque(); self.running=deque()
    @property
    def is_finished(self): return not self.waiting and not self.running
    def add_sequence(self,seq:Sequence):
        if seq.num_blocks>len(self.block_manager.blocks): raise ValueError("prompt can never fit in KV cache")
        if seq.num_tokens>self.max_num_batched_tokens: raise ValueError("prompt exceeds max_num_batched_tokens")
        if seq.num_tokens>=seq.max_model_length: raise ValueError("prompt leaves no room for a completion token")
        if seq.status is not SequenceStatus.WAITING: raise ValueError("only WAITING sequences can be added")
        self.waiting.append(seq)
    def _new_prefill_tokens(self,seq):
        plan,_=self.block_manager._plan(seq)
        cached=sum(len(tokens) for block,_,_,tokens in plan if block is not None)
        return seq.num_tokens-cached
    def schedule(self):
        selected=[]; tokens=0
        while self.waiting and len(self.running)<self.max_num_sequences and len(selected)<self.max_num_sequences:
            seq=self.waiting[0]; needed=self._new_prefill_tokens(seq)
            if tokens+needed>self.max_num_batched_tokens or not self.block_manager.can_allocate(seq): break
            self.waiting.popleft(); self.block_manager.allocate(seq); seq.status=SequenceStatus.RUNNING; self.running.append(seq); selected.append(seq); tokens+=needed
        if selected: return selected,True

        for seq in list(self.running):
            if seq.status is not SequenceStatus.RUNNING:
                continue
            if len(selected)>=self.max_num_sequences or tokens>=self.max_num_batched_tokens: break
            if not self.block_manager.can_append(seq):
                victims=[candidate for candidate in reversed(self.running) if candidate is not seq and candidate not in selected]
                if victims: self.preempt(victims[0])
                else: self.preempt(seq); continue
            if seq.status is SequenceStatus.RUNNING:
                self.block_manager.append(seq); selected.append(seq); tokens+=1
        if not selected and (self.waiting or self.running):
            raise RuntimeError("scheduler made no progress; check budgets or leaked KV blocks")
        return selected,False
    def preempt(self,seq):
        self.running.remove(seq); self.block_manager.deallocate(seq); seq.status=SequenceStatus.WAITING; self.waiting.appendleft(seq)
    def postprocess(self,seqs,token_ids):
        if len(seqs)!=len(token_ids): raise ValueError("one sampled token is required per sequence")
        for seq,token in zip(seqs,token_ids):
            seq.append_token(token)
            stop=(not seq.ignore_eos and token==self.eos_token_id) or seq.num_completion_tokens>=seq.max_tokens or seq.num_tokens>=seq.max_model_length
            if stop:
                seq.status=SequenceStatus.FINISHED
                if seq in self.running: self.running.remove(seq)
                self.block_manager.deallocate(seq)
