import math
import torch
import torch.nn.functional as F
from torch import nn
from day3.layers.linear import MergedColumnParallelLinear,QKVColumnParallelLinear,RowParallelLinear
from day4.layers.embedding_head import ParallelLMHead,VocabParallelEmbedding
from day8.layers.rotary_embedding import QKRMSNorm,RotaryEmbedding
from day9.modeling_qwen3 import RMSNorm


class TPAttention(nn.Module):
    def __init__(self,c):
        super().__init__(); import torch.distributed as dist
        self.c=c; self.size=dist.get_world_size(); self.qh=c.num_attention_heads//self.size; self.kvh=c.num_key_value_heads//self.size
        self.qkv=QKVColumnParallelLinear(c.hidden_size,c.head_dim,c.num_attention_heads,c.num_key_value_heads,bias=c.attention_bias)
        self.qnorm=QKRMSNorm(c.head_dim,c.rms_norm_eps); self.knorm=QKRMSNorm(c.head_dim,c.rms_norm_eps)
        self.rope=RotaryEmbedding(c.head_dim,max_position=c.max_position_embeddings,base=c.rope_theta)
        self.out=RowParallelLinear(c.num_attention_heads*c.head_dim,c.hidden_size,bias=False)
    def forward(self,x,pos):
        qsize=self.qh*self.c.head_dim; kvsize=self.kvh*self.c.head_dim
        q,k,v=self.qkv(x).split((qsize,kvsize,kvsize),-1); b,s,_=x.shape
        q=q.view(b,s,self.qh,self.c.head_dim); k=k.view(b,s,self.kvh,self.c.head_dim); v=v.view(b,s,self.kvh,self.c.head_dim)
        q,k=self.rope(pos,self.qnorm(q),self.knorm(k)); repeat=self.qh//self.kvh
        k=k.repeat_interleave(repeat,2); v=v.repeat_interleave(repeat,2)
        y=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True,scale=1/math.sqrt(self.c.head_dim))
        return self.out(y.transpose(1,2).reshape(b,s,-1))


class TPMLP(nn.Module):
    def __init__(self,c):
        super().__init__(); self.gateup=MergedColumnParallelLinear(c.hidden_size,[c.intermediate_size,c.intermediate_size],bias=False); self.down=RowParallelLinear(c.intermediate_size,c.hidden_size,bias=False)
    def forward(self,x): gate,up=self.gateup(x).chunk(2,-1); return self.down(F.silu(gate)*up)


class TPDecoder(nn.Module):
    def __init__(self,c): super().__init__(); self.inorm=RMSNorm(c.hidden_size,c.rms_norm_eps); self.attn=TPAttention(c); self.pnorm=RMSNorm(c.hidden_size,c.rms_norm_eps); self.mlp=TPMLP(c)
    def forward(self,x,pos): x=x+self.attn(self.inorm(x),pos); return x+self.mlp(self.pnorm(x))


class TPQwen3(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c; self.embed=VocabParallelEmbedding(c.vocab_size,c.hidden_size); self.layers=nn.ModuleList(TPDecoder(c) for _ in range(c.num_hidden_layers)); self.norm=RMSNorm(c.hidden_size,c.rms_norm_eps); self.head=ParallelLMHead(c.vocab_size,c.hidden_size); self.head.tie_weights(self.embed)
    def forward(self,ids):
        x=self.embed(ids); pos=torch.arange(ids.shape[1],device=ids.device).expand(ids.shape[0],-1)
        for layer in self.layers:x=layer(x,pos)
        return self.head(self.norm(x))


def load_tp_hf(model,state):
    with torch.no_grad():
        model.embed.weight_loader(model.embed.weight,state["model.embed_tokens.weight"])
        for i,layer in enumerate(model.layers):
            p=f"model.layers.{i}."; layer.inorm.weight.copy_(state[p+"input_layernorm.weight"]); layer.pnorm.weight.copy_(state[p+"post_attention_layernorm.weight"])
            for name,wid in (("q_proj.weight","q"),("k_proj.weight","k"),("v_proj.weight","v")): layer.attn.qkv.weight_loader(layer.attn.qkv.weight,state[p+"self_attn."+name],wid)
            layer.attn.qnorm.weight.copy_(state[p+"self_attn.q_norm.weight"]); layer.attn.knorm.weight.copy_(state[p+"self_attn.k_norm.weight"])
            layer.attn.out.weight_loader(layer.attn.out.weight,state[p+"self_attn.o_proj.weight"])
            layer.mlp.gateup.weight_loader(layer.mlp.gateup.weight,state[p+"mlp.gate_proj.weight"],0); layer.mlp.gateup.weight_loader(layer.mlp.gateup.weight,state[p+"mlp.up_proj.weight"],1)
            layer.mlp.down.weight_loader(layer.mlp.down.weight,state[p+"mlp.down_proj.weight"])
        model.norm.weight.copy_(state["model.norm.weight"])
