from __future__ import annotations
import torch
import torch.nn.functional as F
from day4.layers.sampler import Sampler
from day5.layers.kv_cache import store_kv_cache
from day6.layers.prefill_attention import prefill_attention
from day7.layers.paged_attention import paged_attention_decode
from day10.engine import SamplingParams,Sequence
from day11.model_runner import ModelRunner,PreparedBatch
from day12.scheduler import Scheduler


class PagedQwen3Runner:
    def __init__(self,model,num_blocks:int,block_size:int,device="cpu",greedy=False):
        self.model=model.to(device).eval(); self.device=torch.device(device); self.block_size=block_size; self.greedy=greedy
        c=model.config; dtype=next(model.parameters()).dtype
        self.k_cache=torch.zeros(c.num_hidden_layers,num_blocks,block_size,c.num_key_value_heads,c.head_dim,device=device,dtype=dtype)
        self.v_cache=torch.zeros_like(self.k_cache); self.preparer=ModelRunner(block_size,device); self.sampler=Sampler()
    def _history(self,cache,tables,lengths):
        parts=[]
        for row,length in zip(tables,lengths.tolist()):
            pos=torch.arange(length,device=self.device); physical=row[pos//self.block_size]; parts.append(cache[physical,pos%self.block_size])
        return torch.cat(parts)
    def _attention(self,attention,x,batch,layer,is_prefill):
        q,k,v=attention.qkv_proj(x).split((attention.q_size,attention.kv_size,attention.kv_size),-1)
        c=attention.c; q=q.view(-1,c.num_attention_heads,c.head_dim); k=k.view(-1,c.num_key_value_heads,c.head_dim); v=v.view_as(k)
        q,k=attention.rotary_emb(batch.positions,attention.q_norm(q),attention.k_norm(k))
        store_kv_cache(k,v,self.k_cache[layer],self.v_cache[layer],batch.slot_mapping)
        if is_prefill:
            lengths=batch.cu_seqlens_k[1:]-batch.cu_seqlens_k[:-1]
            full_k=self._history(self.k_cache[layer],batch.block_tables,lengths); full_v=self._history(self.v_cache[layer],batch.block_tables,lengths)
            out=prefill_attention(q,full_k,full_v,batch.cu_seqlens_q,batch.cu_seqlens_k)
        else:
            out=paged_attention_decode(q,self.k_cache[layer],self.v_cache[layer],batch.block_tables,batch.context_lens)
        return attention.o_proj(out.reshape(x.shape[0],-1))
    @torch.inference_mode()
    def forward(self,batch:PreparedBatch,is_prefill:bool):
        x=self.model.model.embed_tokens(batch.input_ids)
        for i,layer in enumerate(self.model.model.layers):
            x=x+self._attention(layer.self_attn,layer.input_layernorm(x),batch,i,is_prefill)
            x=x+layer.mlp(layer.post_attention_layernorm(x))
        x=self.model.model.norm(x)
        if is_prefill: x=x.index_select(0,batch.cu_seqlens_q[1:].long()-1)
        return F.linear(x,self.model.lm_head.weight).float()
    def run(self,seqs,is_prefill):
        batch=self.preparer.prepare_prefill(seqs) if is_prefill else self.preparer.prepare_decode(seqs)
        logits=self.forward(batch,is_prefill)
        if self.greedy: return logits.argmax(-1).tolist()
        temperatures=torch.tensor([s.temperature for s in seqs],device=self.device)
        return self.sampler(logits,temperatures).tolist()


class LLMEngine:
    def __init__(self,runner,max_num_sequences,max_num_batched_tokens,max_cached_blocks,block_size,eos_token_id,tokenizer=None):
        self.runner=runner; self.tokenizer=tokenizer
        self.scheduler=Scheduler(max_num_sequences,max_num_batched_tokens,max_cached_blocks,block_size,eos_token_id)
    def add_request(self,prompt,sampling_params=None):
        ids=self.tokenizer.encode(prompt) if isinstance(prompt,str) else list(prompt)
        if not ids: raise ValueError("prompt cannot be empty")
        seq=Sequence(ids,self.scheduler.block_manager.block_size,sampling_params); self.scheduler.add_sequence(seq); return seq
    def step(self):
        seqs,is_prefill=self.scheduler.schedule()
        if not seqs: return []
        tokens=self.runner.run(seqs,is_prefill); self.scheduler.postprocess(seqs,tokens); return list(zip(seqs,tokens))
    def generate_token_ids(self,prompts,sampling_params=None):
        if isinstance(sampling_params,(SamplingParams,type(None))): params=[sampling_params]*len(prompts)
        else: params=list(sampling_params)
        if len(params)!=len(prompts): raise ValueError("sampling params must match prompts")
        seqs=[self.add_request(prompt,param) for prompt,param in zip(prompts,params)]
        while not self.scheduler.is_finished: self.step()
        return [seq.token_ids[seq.num_prompt_tokens:] for seq in seqs]
    def generate(self,prompts,sampling_params=None):
        ids=self.generate_token_ids(prompts,sampling_params)
        return [self.tokenizer.decode(tokens) for tokens in ids] if self.tokenizer else ids
