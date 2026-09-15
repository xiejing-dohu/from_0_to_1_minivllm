import pickle,struct
from multiprocessing import shared_memory


class SharedMemoryMailbox:
    HEADER=8
    def __init__(self,capacity:int,name:str|None=None,create:bool=True):
        if capacity<=self.HEADER: raise ValueError("capacity is too small")
        self.capacity=capacity; self.shm=shared_memory.SharedMemory(name=name,create=create,size=capacity if create else 0)
    @property
    def name(self): return self.shm.name
    def write(self,value):
        payload=pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload)>self.capacity-self.HEADER: raise ValueError("payload exceeds shared memory capacity")
        self.shm.buf[self.HEADER:self.HEADER+len(payload)]=payload
        self.shm.buf[:self.HEADER]=struct.pack("<Q",len(payload))
    def read(self):
        length=struct.unpack("<Q",self.shm.buf[:self.HEADER])[0]
        if length>self.capacity-self.HEADER: raise RuntimeError("corrupt shared-memory message length")
        return pickle.loads(bytes(self.shm.buf[self.HEADER:self.HEADER+length]))
    def close(self): self.shm.close()
    def unlink(self): self.shm.unlink()


def increment_int64_tensor_worker(name,capacity,request_ready,response_ready):
    """Minimal worker kept free of PyTorch/CUDA imports for reliable spawning."""
    box=SharedMemoryMailbox(capacity,name=name,create=False)
    try:
        if not request_ready.wait(10): return
        message=box.read()
        if message["dtype"]!="int64": raise ValueError("worker expects int64 tensor bytes")
        count=1
        for dimension in message["shape"]: count*=dimension
        values=struct.unpack("<"+"q"*count,message["data"])
        message["data"]=struct.pack("<"+"q"*count,*(value+1 for value in values))
        box.write(message); response_ready.set()
    finally: box.close()
