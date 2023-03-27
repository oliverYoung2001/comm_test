import torch
import torch.distributed as dist
import time
from utils.argparser import get_args
import multiprocessing as mp
import random
from utils.parallel_context import gpc
from utils.comm import _all_to_all
from functools import reduce
import copy

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 10377

SIZES = [
    [1, 1024, 1024, 32],
    [1, 1024, 1024, 64],
    [1, 1024, 1024, 128],
    [1, 1024, 1024, 256],
    [1, 1024, 1024, 512],
    [1, 1024, 1024, 1024],
    [1, 1024, 1024, 2048],
    # [1, 1024, 1024, 4096],
    # [1, 1024, 1024, 8192],
]
TIMES = 10

def comm_test(rank, world_size, SIZE, in_dim, out_dim, args):
    # print(f'dist.is_available(): {dist.is_available()}')
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank, group_name='all2all_test')
    gpc.set_world_size(world_size=world_size)
    gpc.set_local_rank(local_rank=rank)
    gpc.set_rank(rank=rank)
    # set cuda device
    if torch.cuda.is_available():
        # if local rank is not given, calculate automatically
        torch.cuda.set_device(rank)
    
    # SIZE = (1, 1024, 1024, 128)
    # SIZE = (1, 2048, 2048, 1024)
    # in_dim = 2
    # out_dim = 1
    if rank == 0:
        print(f'SIZE: {SIZE}, in/out_dim: {in_dim}/{out_dim}')
    SSIZE = copy.deepcopy(SIZE)
    SSIZE[out_dim] //= world_size   # single-node SIZE
    for method in ['SC0', 'SC1', 'SC2', 'SC3', 'A2A', 'AGD']:
    # for method in ['AGD']:
    # for method in ['A2A']:
    # for method in ['P2P']:
    # for method in ['RIN']:
    # for method in ['SC0']:
    # for method in ['SC1']:
    # for method in ['SC2']:
    # for method in ['8SC0', '8SC1']:
    # for method in ['8SC2', '8SC3']:
        # dist.ProcessGroup
        # a = torch.randn((1, 2048 // world_size, 2048, 1024), dtype=torch.float32)
        a = torch.randn(tuple(SSIZE), dtype=torch.float32)
        # total_num = 1 * 4 * 4
        # a = torch.arange(total_num // world_size).view(1, 4 // world_size, 4) + (total_num // world_size * rank)
        a = a.cuda()
        avg_bd = 0
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        for _ in range(TIMES):
            # print(f'local_rank: {local_rank}, a.is_contiguous(): {a.is_contiguous()}')
            # a = a.contiguous()
            # torch.dist._all
            b, bandwidth = _all_to_all(a, in_dim, out_dim, method=method)
            avg_bd += bandwidth
            # a = a.transpose(out_dim, in_dim)
            # a += random.randint(0, 1000)
            # print(a)
            # torch.cuda.empty_cache()
            torch.cuda.synchronize()    # avoid pre-allocating memory
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.time()
        if rank == 0:
            t_d = t1 - t0
            calc = reduce((lambda x,y: x*y), SIZE) * 4 * TIMES * ((world_size - 1) / (pow(world_size, 2))) \
                    / (pow(1024, 3)) # GB
            avg_bd /= TIMES
            print(f'{method}: {t_d}, REAL_BD: {avg_bd} GB/s, TOTAL_BD: {calc / t_d} GB/s, comm_vol: {calc} GB')
    
    # result_q.put(rank)
    dist.destroy_process_group()


def main():
    args = get_args()
    print(f'CONFIG:')
    print(f'WORLD_SIZE: {args.gpus}')
    print()
    times = 1
    for _ in range(times):
        manager = mp.Manager()
        result_q = manager.Queue()
        print(f'round {_}:')
        for SIZE in SIZES:
            torch.multiprocessing.spawn(comm_test, nprocs=args.gpus, args=(args.gpus, SIZE, 2, 1, args))
        print()
    # while not result_q.empty():
    #     out = result_q.get()
    #     print(out)
    
    
if __name__ == '__main__':
    main()