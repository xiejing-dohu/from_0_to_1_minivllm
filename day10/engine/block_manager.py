from __future__ import annotations
from collections import defaultdict,deque
from dataclasses import dataclass,field
import hashlib
from .sequence import Sequence


@dataclass
class Block:
    block_id:int; ref_count:int=0; hash_value:bytes|None=None; prefix_hash:bytes|None=None; token_ids:tuple[int,...]=field(default_factory=tuple)


class BlockManager:
    def __init__(self,num_blocks:int,block_size:int):
        if num_blocks<=0 or block_size<=0: raise ValueError("sizes must be positive")
        self.block_size=block_size; self.blocks=[Block(i) for i in range(num_blocks)]
        self.free=deque(range(num_blocks)); self.free_set=set(range(num_blocks)); self.used=set(); self.hash_index=defaultdict(set)
    def compute_hash(self,tokens,prefix):
        h=hashlib.blake2b(digest_size=16); h.update(prefix or b"ROOT")
        for token in tokens: h.update(int(token).to_bytes(8,"little",signed=True))
        return h.digest()
    def _unindex(self,b):
        if b.hash_value is not None:
            ids=self.hash_index[b.hash_value]; ids.discard(b.block_id)
            if not ids: del self.hash_index[b.hash_value]
    def _evict_and_claim(self):
        if not self.free: raise RuntimeError("KV cache has no free physical block")
        block_id=self.free.popleft(); self.free_set.remove(block_id); b=self.blocks[block_id]
        self._unindex(b); b.hash_value=None; b.prefix_hash=None; b.token_ids=(); b.ref_count=1; self.used.add(block_id); return b
    def _find(self,h,tokens,prefix):
        for block_id in self.hash_index.get(h,()):
            b=self.blocks[block_id]
            if b.token_ids==tuple(tokens) and b.prefix_hash==prefix: return b
        return None
    def _claim_cached(self,b):
        if b.ref_count==0:
            self.free.remove(b.block_id); self.free_set.remove(b.block_id); self.used.add(b.block_id)
        b.ref_count+=1; return b
    def _plan(self,seq):
        prefix=None; hits=[]; required_free=0; claimed_cached=set()
        for i in range(seq.num_blocks):
            tokens=seq.block(i)
            # Keep the final prompt block as a query-producing miss. If the
            # whole prompt were reported cached, ModelRunner would have no
            # hidden state from which to compute the first sampled token.
            if len(tokens)==self.block_size and i < seq.num_blocks-1:
                h=self.compute_hash(tokens,prefix); b=self._find(h,tokens,prefix)
            else:
                h=self.compute_hash(tokens,prefix) if len(tokens)==self.block_size else None
                b=None
            hits.append((b,h,prefix,tuple(tokens)))
            if b is None:
                required_free+=1
            elif b.ref_count==0 and b.block_id not in claimed_cached:
                required_free+=1; claimed_cached.add(b.block_id)
            prefix=h
        return hits,required_free
    def can_allocate(self,seq): return self._plan(seq)[1]<=len(self.free)
    def allocate(self,seq):
        if seq.block_table: raise ValueError("sequence is already allocated")
        plan,required_free=self._plan(seq)
        if required_free>len(self.free): raise RuntimeError("insufficient KV cache blocks")
        seq.num_cached_tokens=0
        for cached,h,prefix,tokens in plan:
            if cached is not None:
                b=self._claim_cached(cached); seq.num_cached_tokens+=self.block_size
            else:
                b=self._evict_and_claim(); b.token_ids=tokens; b.prefix_hash=prefix
                if h is not None: b.hash_value=h; self.hash_index[h].add(b.block_id)
            seq.block_table.append(b.block_id)
        self.assert_invariants([seq])
    def can_append(self,seq):
        needs_block=seq.num_tokens>0 and (seq.num_tokens-1)%self.block_size==0 and len(seq.block_table)<seq.num_blocks
        return not needs_block or bool(self.free)
    def append(self,seq):
        if not seq.token_ids or not seq.block_table: raise ValueError("append_token must follow allocation")
        needs_block=len(seq.block_table)<seq.num_blocks
        if needs_block:
            if len(seq.block_table)+1!=seq.num_blocks: raise ValueError("sequence/cache metadata diverged")
            seq.block_table.append(self._evict_and_claim().block_id)
        b=self.blocks[seq.block_table[-1]]; tokens=tuple(seq.block(seq.num_blocks-1)); self._unindex(b)
        b.token_ids=tokens; b.hash_value=None
        prefix=self.blocks[seq.block_table[-2]].hash_value if len(seq.block_table)>1 else None; b.prefix_hash=prefix
        if len(tokens)==self.block_size:
            b.hash_value=self.compute_hash(tokens,prefix); self.hash_index[b.hash_value].add(b.block_id)
    def deallocate(self,seq):
        for block_id in seq.block_table:
            b=self.blocks[block_id]
            if b.ref_count<=0: raise RuntimeError("block reference underflow")
            b.ref_count-=1
            if b.ref_count==0:
                self.used.remove(block_id); self.free.append(block_id); self.free_set.add(block_id)
                if b.hash_value is None: b.token_ids=(); b.prefix_hash=None
        seq.block_table=[]; seq.num_cached_tokens=0; self.assert_invariants()
    def assert_invariants(self,sequences=()):
        if len(self.free)!=len(self.free_set): raise AssertionError("duplicate free block")
        referenced={b.block_id for b in self.blocks if b.ref_count>0}
        if referenced!=self.used or self.free_set & self.used or len(self.free_set)+len(self.used)!=len(self.blocks): raise AssertionError("free/used partition is invalid")
        for seq in sequences:
            if any(i in self.free_set for i in seq.block_table): raise AssertionError("sequence references a free block")
