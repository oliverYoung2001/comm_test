import torch
from utils.argparser import get_args
from utils.parallel_context import gpc
# from ..utils.parallel_context import gpc
import torch.distributed as dist
import time
from functools import reduce 
import sys

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 10376

BATCH = 1

SIZES = [
    # [BATCH, 1024, 1024, 64],
    # [BATCH, 1024, 1024, 128],
    # [BATCH, 1024, 1024, 256],
    # [BATCH, 1024, 1024, 512],
    # [BATCH, 1024, 1024, 1024],
    # [BATCH, 1024, 1024, 2048],
    [BATCH, 1024, 1024, 4096],
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
        # Bidirectional p2p
        a = torch.randn(SIZE, dtype=torch.float32).cuda()
        b = torch.empty(SIZE, dtype=torch.float32).cuda()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        for _ in range(TIMES):
            ops = []
            if rank == 0:
                # print(f'0 in send !!!')
                # sys.stdout.flush()
                ops.append(dist.P2POp(dist.isend, a, 1))
                # works.append(dist.isend(a, 1))
                # print(f'0 send done !!!')
                # sys.stdout.flush()
                ops.append(dist.P2POp(dist.irecv, b, 1))
                # works.append(dist.irecv(b, 1))
                # print(f'0 recv done !!!')
                # sys.stdout.flush()
            if rank == 1:
                # print(f'1 in recv')
                # sys.stdout.flush()
                ops.append(dist.P2POp(dist.irecv, b, 0))
                # works.append(dist.irecv(b, 0))
                # print(f'1 recv done !!!')
                # sys.stdout.flush()
                ops.append(dist.P2POp(dist.isend, a, 0))
                # works.append(dist.isend(a, 0))
                # print(f'1 send done !!!')
                # sys.stdout.flush()
            works = dist.batch_isend_irecv(ops)
            for work in works:
                work.wait()
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.time()
        if rank == 0:
            t_d = t1 - t0
            print(f'time: {t_d}')
            calc = reduce((lambda x,y: x*y), SIZE) * 4 * TIMES / (1024 * 1024 * 1024) # GB
            print(f'BD: {calc / t_d} GB/s')
        
        del a, b
        torch.cuda.empty_cache()
        
        

def main():
    args = get_args()
    torch.multiprocessing.spawn(net_test, nprocs=args.gpus, args=(args.gpus, args))

if __name__ == '__main__':
    main()