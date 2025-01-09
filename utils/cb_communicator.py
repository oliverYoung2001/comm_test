import torch
import torch.distributed as dist
from .distributed.device_communicators.pynccl import PyNcclCommunicator
from .distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from functools import partial


class Cb_Communicator():
    def __init__(self, PROC_INFO: dict, args, group_dict: dict):
        self.ops = []
        self.comm_module = args.comm_module
        self.group_dict = group_dict    # key -> gloo_group
        self.ncclcomm_dict = {} # key -> PyNcclCommunicator
        self.reqs = []
        self.streams = []
        self.device = torch.cuda.current_device()
        if self.comm_module == 'raw-nccl':
            self.streams = [
                torch.cuda.Stream(device=self.device),  # Send Stream
                torch.cuda.Stream(device=self.device),  # Recv Stream
            ]
        self.rank = PROC_INFO['rank']
        self.world_size = PROC_INFO['world_size']
        try:
            self.nccl = NCCLLibrary(None)
        except Exception:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            raise Exception("NCCL library not found")
    
    def send(self, t: torch.Tensor, peer: int, stream = None):
        if self.comm_module == 'torch-distributed':
            self.ops.append(partial(dist.isend, t, peer))
        elif self.comm_module == 'raw-nccl':
            # create PyNcclCommunicator for ranks
            key = (self.rank, peer)
            if key not in self.ncclcomm_dict.keys():
                group_key = tuple(sorted([self.rank, peer]))
                self.ncclcomm_dict[key] = PyNcclCommunicator(self.group_dict[group_key], device=self.device)
            # record ops
            ncclcomm = self.ncclcomm_dict[key]
            if stream is None:
                stream = self.streams[0]
            self.ops.append(partial(ncclcomm.send, t, self.rank < peer, stream))
        
    def recv(self, t: torch.Tensor, peer: int, stream = None):
        if self.comm_module == 'torch-distributed':
            self.ops.append(partial(dist.irecv, t, peer))
        elif self.comm_module == 'raw-nccl':
            # create PyNcclCommunicator for ranks
            key = (peer, self.rank)
            if key not in self.ncclcomm_dict.keys():
                group_key = tuple(sorted([self.rank, peer]))
                self.ncclcomm_dict[key] = PyNcclCommunicator(self.group_dict[group_key], device=self.device)
            # record ops
            ncclcomm = self.ncclcomm_dict[key]
            if stream is None:
                stream = self.streams[1]
            self.ops.append(partial(ncclcomm.recv, t, self.rank < peer, stream))
    
    def all_reduce(self, t: torch.Tensor, stream = None):
        if self.comm_module == 'torch-distributed':
            self.ops.append(partial(dist.all_reduce, t, async_op=True))
        elif self.comm_module == 'raw-nccl':
            assert self.world_size == 2
            key = (0, 1)
            if key not in self.ncclcomm_dict.keys():
                group_key = key
                self.ncclcomm_dict[key] = PyNcclCommunicator(self.group_dict[group_key], device=self.device)
            # record ops
            ncclcomm = self.ncclcomm_dict[key]
            if stream is None:
                stream = self.streams[0]
            
            self.ops.append(partial(ncclcomm.all_reduce, t, stream=stream))
    
    def execute_comm_ops(self, barrier=False, light_barrier=False, nccl_group=False):
        if len(self.ops) > 0:
            # if nccl_group:  # Useless !!!
            #     self.nccl.ncclGroupStart()
            # works = dist.batch_isend_irecv(ops)
            works = [op() for op in self.ops]
            # if nccl_group:
            #     self.nccl.ncclGroupEnd()
            for work in works:
                if work:
                    work.wait()
        if barrier:
            torch.cuda.synchronize()                            # 每一轮是否barrier对性能没有影响
            dist.barrier()
        elif light_barrier:
            torch.cuda.synchronize() 
    
    def wait_ops(self):
        if self.comm_module == 'torch-distributed':
            for req in self.reqs:
                req.wait()
            self.reqs = []
        elif self.comm_module == 'raw-nccl':
            for s in self.streams:
                s.synchronize()
        
    def clear_ops(self):
        self.ops = []