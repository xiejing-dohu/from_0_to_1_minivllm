from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from day8.layers.rotary_embedding import QKRMSNorm, RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__(); self.weight=nn.Parameter(torch.ones(size)); self.eps=eps
    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)*self.weight.float()).to(x.dtype)


@dataclass
class Qwen3Config:
    vocab_size:int; hidden_size:int; intermediate_size:int; num_hidden_layers:int
    num_attention_heads:int; num_key_value_heads:int; head_dim:int
    max_position_embeddings:int=4096; rope_theta:float=1_000_000.0
    rms_norm_eps:float=1e-6; attention_bias:bool=False; tie_word_embeddings:bool=True

    @classmethod
    def from_hf(cls,c):
        values={name:getattr(c,name) for name in cls.__dataclass_fields__ if name!="rope_theta"}
        rope=getattr(c,"rope_parameters",None) or {}
        values["rope_theta"]=rope.get("rope_theta",rope.get("base",1_000_000.0))
        return cls(**values)


class Qwen3Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        q=c.num_attention_heads*c.head_dim; kv=c.num_key_value_heads*c.head_dim
        self.q_size,self.kv_size=q,kv
        self.qkv_proj=nn.Linear(c.hidden_size,q+2*kv,bias=c.attention_bias)
        self.q_norm=QKRMSNorm(c.head_dim,c.rms_norm_eps); self.k_norm=QKRMSNorm(c.head_dim,c.rms_norm_eps)
        self.rotary_emb=RotaryEmbedding(c.head_dim,max_position=c.max_position_embeddings,base=c.rope_theta)
        self.o_proj=nn.Linear(q,c.hidden_size,bias=False)
    def forward(self,x,positions):
        b,s,_=x.shape; q,k,v=self.qkv_proj(x).split((self.q_size,self.kv_size,self.kv_size),-1)
        q=q.view(b,s,self.c.num_attention_heads,self.c.head_dim); k=k.view(b,s,self.c.num_key_value_heads,self.c.head_dim); v=v.view(b,s,self.c.num_key_value_heads,self.c.head_dim)
        q,k=self.rotary_emb(positions,self.q_norm(q),self.k_norm(k))
        repeat=self.c.num_attention_heads//self.c.num_key_value_heads
        k=k.repeat_interleave(repeat,dim=2); v=v.repeat_interleave(repeat,dim=2)
        out=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True,dropout_p=0.0,scale=1/math.sqrt(self.c.head_dim))
        return self.o_proj(out.transpose(1,2).reshape(b,s,-1))


class Qwen3MLP(nn.Module):
    def __init__(self,c):
        super().__init__(); self.intermediate_size=c.intermediate_size
        self.gate_up_proj=nn.Linear(c.hidden_size,2*c.intermediate_size,bias=False)
        self.down_proj=nn.Linear(c.intermediate_size,c.hidden_size,bias=False)
    def forward(self,x):
        gate,up=self.gate_up_proj(x).chunk(2,-1)
        return self.down_proj(F.silu(gate)*up)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self,c):
        super().__init__(); self.input_layernorm=RMSNorm(c.hidden_size,c.rms_norm_eps); self.self_attn=Qwen3Attention(c); self.post_attention_layernorm=RMSNorm(c.hidden_size,c.rms_norm_eps); self.mlp=Qwen3MLP(c)
    def forward(self,x,positions):
        x=x+self.self_attn(self.input_layernorm(x),positions)
        return x+self.mlp(self.post_attention_layernorm(x))


class Qwen3Model(nn.Module):
    def __init__(self,c):
        super().__init__(); self.embed_tokens=nn.Embedding(c.vocab_size,c.hidden_size); self.layers=nn.ModuleList(Qwen3DecoderLayer(c) for _ in range(c.num_hidden_layers)); self.norm=RMSNorm(c.hidden_size,c.rms_norm_eps)
    def forward(self,input_ids):
        x=self.embed_tokens(input_ids); positions=torch.arange(input_ids.shape[1],device=input_ids.device).expand(input_ids.shape[0],-1)
        for layer in self.layers: x=layer(x,positions)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c; self.model=Qwen3Model(c); self.lm_head=nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        if c.tie_word_embeddings: self.lm_head.weight=self.model.embed_tokens.weight
    def forward(self,input_ids): return self.lm_head(self.model(input_ids)).float()


def load_hf_state_dict(model:Qwen3ForCausalLM,state:dict[str,torch.Tensor])->set[str]:
    """Map separate HF projections into the runtime's merged parameters."""
    used=set()
    def copy(param,name):
        if name not in state or param.shape!=state[name].shape: raise ValueError(f"missing or incompatible checkpoint tensor: {name}")
        param.copy_(state[name]); used.add(name)
    with torch.no_grad():
        copy(model.model.embed_tokens.weight,"model.embed_tokens.weight")
        for i,layer in enumerate(model.model.layers):
            p=f"model.layers.{i}."
            copy(layer.input_layernorm.weight,p+"input_layernorm.weight"); copy(layer.post_attention_layernorm.weight,p+"post_attention_layernorm.weight")
            parts=[]
            for name in ("q_proj.weight","k_proj.weight","v_proj.weight"):
                key=p+"self_attn."+name; parts.append(state[key]); used.add(key)
            layer.self_attn.qkv_proj.weight.copy_(torch.cat(parts,0))
            if layer.self_attn.qkv_proj.bias is not None:
                biases=[]
                for name in ("q_proj.bias","k_proj.bias","v_proj.bias"):
                    key=p+"self_attn."+name; biases.append(state[key]); used.add(key)
                layer.self_attn.qkv_proj.bias.copy_(torch.cat(biases,0))
            copy(layer.self_attn.q_norm.weight,p+"self_attn.q_norm.weight"); copy(layer.self_attn.k_norm.weight,p+"self_attn.k_norm.weight"); copy(layer.self_attn.o_proj.weight,p+"self_attn.o_proj.weight")
            gates=[]
            for name in ("gate_proj.weight","up_proj.weight"):
                key=p+"mlp."+name; gates.append(state[key]); used.add(key)
            layer.mlp.gate_up_proj.weight.copy_(torch.cat(gates,0)); copy(layer.mlp.down_proj.weight,p+"mlp.down_proj.weight")
        copy(model.model.norm.weight,"model.norm.weight")
        if "lm_head.weight" in state:
            copy(model.lm_head.weight,"lm_head.weight")
    ignored={k for k in state if "rotary_emb.inv_freq" in k}
    unexpected=set(state)-used-ignored
    if unexpected: raise ValueError(f"unmapped checkpoint keys: {sorted(unexpected)}")
    return ignored
