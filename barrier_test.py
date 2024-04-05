import torch
from utils.argparser import get_args
from utils.parallel_context import gpc
# from ..utils.parallel_context import gpc
import torch.distributed as dist
import time
from functools import reduce 
import pandas as pd
from pandas import DataFrame
import os
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from utils.common import execute_comm_ops

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 10375

BATCH = 1

SIZES = [
    [BATCH, 1024, 1024, 64],
    # [BATCH, 1024, 1024, 128],
    # [BATCH, 1024, 1024, 256],
    # [BATCH, 1024, 1024, 512],
    [BATCH, 1024, 1024, 1024],
    # [BATCH, 1024, 1024, 2048],
    # [BATCH, 1024, 1024, 4096],
    # [1, 1024, 1024, 8192],
]

GPU_NUM = 8

def net_test(rank, world_size, args):
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank, group_name='net_test')
    gpc.set_world_size(world_size=world_size)
    gpc.set_local_rank(local_rank=rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    WARM_UP = 100
    TIMES = 2000
    # GPUIDs = [int(i) for i in args.gpuids.split(',')]
    
    for _ in range(WARM_UP):
        torch.cuda.synchronize()
        dist.barrier()
    
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for _ in range(TIMES // 10):
        pass
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
        # torch.cuda.synchronize()
        # dist.barrier()
    t1 = time.time()
    
    if rank == 0:
        t_d = (t1 - t0) / TIMES
        print(f'GPUs: {args.gpus}, time {t_d * pow(1000, 2)} us')

def main():
    args = get_args()
    torch.multiprocessing.spawn(net_test, nprocs=args.gpus, args=(args.gpus, args))

if __name__ == '__main__':
    main()