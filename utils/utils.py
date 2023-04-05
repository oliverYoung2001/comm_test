import torch
import torch.distributed as dist

def calc_comm_vol_a2a(t: torch.Tensor, world_size): # -> GB
    return t.nelement() * t.element_size() * (world_size - 1) / world_size / pow(1024, 3)

def calc_comm_vol_ag(t: torch.Tensor, world_size):  # -> GB
    return t.nelement() * t.element_size() * (world_size - 1) / pow(1024, 3)

def execute_comm_ops(ops, barrier=False, light_barrier=False):
    if len(ops) > 0:
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
    if barrier:
        torch.cuda.synchronize()                            # 每一轮是否barrier对性能没有影响
        dist.barrier()
    elif light_barrier:
        torch.cuda.synchronize() 
