def kv_block_bytes(num_layers:int,block_size:int,num_kv_heads:int,head_dim:int,dtype_bytes:int)->int:
    values=(num_layers,block_size,num_kv_heads,head_dim,dtype_bytes)
    if any(v<=0 for v in values): raise ValueError("KV dimensions must be positive")
    return 2*num_layers*block_size*num_kv_heads*head_dim*dtype_bytes


def available_kv_blocks(total_bytes:int,current_peak_bytes:int,utilization:float,safety_bytes:int,**shape)->int:
    if not 0<utilization<=1 or min(total_bytes,current_peak_bytes,safety_bytes)<0: raise ValueError("invalid memory budget")
    available=int(total_bytes*utilization)-current_peak_bytes-safety_bytes
    return max(0,available//kv_block_bytes(**shape))
