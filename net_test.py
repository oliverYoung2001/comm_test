import torch
from utils.argparser import get_args
from utils.parallel_context import gpc
# from ..utils.parallel_context import gpc
import torch.distributed as dist
import time
from functools import reduce 

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 10375

BATCH = 1

SIZES = [
    [BATCH, 1024, 1024, 64],
    [BATCH, 1024, 1024, 128],
    [BATCH, 1024, 1024, 256],
    [BATCH, 1024, 1024, 512],
    [BATCH, 1024, 1024, 1024],
    # [BATCH, 1024, 1024, 2048],
    # [BATCH, 1024, 1024, 4096],
    # [1, 1024, 1024, 8192],
]

def net_test(rank, world_size, args):
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank, group_name='net_test')
    gpc.set_world_size(world_size=world_size)
    gpc.set_local_rank(local_rank=rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    TIMES = 20
    
    for SIZE in SIZES:
        if rank == 0:
            print(f'SIZE: {SIZE}')
        # p2p; 0 -> 1, 1 -> 0
        if rank == 0:
            a = torch.randn(SIZE, dtype=torch.float32).cuda()
        if rank == 1:
            a = torch.empty(SIZE, dtype=torch.float32).cuda()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        if rank == 0:
            for _ in range(TIMES):
                dist.send(a, 1)
        if rank == 1:
            for _ in range(TIMES):
                dist.recv(a, 0)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.time()
        if rank == 0:
            t_d = t1 - t0
            print(f'time: {t_d}')
            calc = reduce((lambda x,y: x*y), SIZE) * 4 * TIMES / (1024 * 1024 * 1024) # GB
            print(f'BD: {calc / t_d} GB/s')
        
        

def main():
    args = get_args()
    torch.multiprocessing.spawn(net_test, nprocs=args.gpus, args=(args.gpus, args))

if __name__ == '__main__':
    main()