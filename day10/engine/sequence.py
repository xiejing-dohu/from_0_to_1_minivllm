from dataclasses import dataclass
from enum import Enum,auto
from itertools import count


class SequenceStatus(Enum): WAITING=auto(); RUNNING=auto(); FINISHED=auto()


@dataclass(frozen=True)
class SamplingParams:
    temperature:float=1.0; max_tokens:int=32; ignore_eos:bool=False; max_model_length:int=4096
    def __post_init__(self):
        if self.temperature<=0 or self.max_tokens<=0 or self.max_model_length<=0: raise ValueError("invalid sampling parameters")


class Sequence:
    _ids=count()
    def __init__(self,token_ids:list[int],block_size:int,sampling_params:SamplingParams|None=None):
        if block_size<=0: raise ValueError("block_size must be positive")
        self.seq_id=next(self._ids); self.status=SequenceStatus.WAITING; self.block_size=block_size
        self.token_ids=list(token_ids); self.num_prompt_tokens=len(token_ids); self.num_cached_tokens=0; self.block_table=[]
        self.sampling_params=sampling_params or SamplingParams()
    def __len__(self): return len(self.token_ids)
    def __getitem__(self,index): return self.token_ids[index]
    @property
    def last_token(self): return self.token_ids[-1] if self.token_ids else None
    @property
    def num_tokens(self): return len(self.token_ids)
    @property
    def num_completion_tokens(self): return self.num_tokens-self.num_prompt_tokens
    @property
    def num_blocks(self): return (self.num_tokens+self.block_size-1)//self.block_size
    @property
    def num_cached_blocks(self): return self.num_cached_tokens//self.block_size
    @property
    def last_block_num_tokens(self): return 0 if not self.num_tokens else (self.num_tokens-1)%self.block_size+1
    @property
    def is_finished(self): return self.status is SequenceStatus.FINISHED
    @property
    def temperature(self): return self.sampling_params.temperature
    @property
    def max_tokens(self): return self.sampling_params.max_tokens
    @property
    def ignore_eos(self): return self.sampling_params.ignore_eos
    @property
    def max_model_length(self): return self.sampling_params.max_model_length
    def block(self,index):
        if not 0<=index<self.num_blocks: raise IndexError("block index out of range")
        start=index*self.block_size; return self.token_ids[start:start+self.block_size]
    def append_token(self,token_id:int): self.token_ids.append(int(token_id))
