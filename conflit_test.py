import torch
from utils.argparser import get_args
from utils.parallel_context import gpc
# from ..utils.parallel_context import gpc
import torch.distributed as dist
import time
from functools import reduce 
import sys
import pandas as pd
from pandas import DataFrame
import os
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from utils.utils import execute_comm_ops

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 10376

BATCH = 1

SIZES = [
    # [BATCH, 1024],              # 1K * 4B
    # [BATCH, 1024, 1024],        # 1M * 4B
    # [BATCH, 1024, 1024, 64],
    [BATCH, 1024, 1024, 128],
    # [BATCH, 1024, 1024, 256],
    # [BATCH, 1024, 1024, 512],
    # [BATCH, 1024, 1024, 1024],
    # [BATCH, 1024, 1024, 2048],
    # [BATCH, 1024, 1024, 4096],
    # [1, 1024, 1024, 8192],
]

GPU_NUM = 8
WARM_UP = 10
TIMES = 20  

def evaluate_GPU_pairs(GPU_pairs, rank, args):
    if max([max(pair[0], pair[1]) for pair in GPU_pairs]) >= args.gpus:
        return
    if rank == 0:
        print(f'GPU_pairs: {GPU_pairs}')
        sys.stdout.flush()
    
    for i, SIZE in enumerate(SIZES):
        a = torch.randn(SIZE, dtype=torch.float32).cuda()
        b = torch.randn(SIZE, dtype=torch.float32).cuda()
        
        ops = []
        for pair in GPU_pairs:
            if rank == pair[0]:
                ops.append(dist.P2POp(dist.isend, a, pair[1]))        # 是否用相同的buffer对性能没有影响
            if rank == pair[1]:
                ops.append(dist.P2POp(dist.irecv, b, pair[0]))
        
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(WARM_UP):
            execute_comm_ops(ops, barrier=True)
            
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        for _ in range(TIMES):
            # light-barrier 稍快一些(0.1GB/s)
            # barrier(=light-barrier+cpu-barrier) 与light-barrier相同
            # cpu-barrier本身没有意义
            execute_comm_ops(ops, barrier=True, light_barrier=True)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.time()
        
        # if True:
        if rank == 0:
            t_d = t1 - t0
            # calc = len(GPU_pairs) * reduce((lambda x,y: x*y), SIZE) * 4 * TIMES / pow(1024, 3) # GB
            calc = reduce((lambda x,y: x*y), SIZE) * 4 * TIMES / pow(1024, 3) # GB
            BD = calc / t_d
            print(f'rank {rank}, SIZE {SIZE}, REAL_BD {BD} GB/s, time {t_d} s, comm_vol {calc} GB')
            sys.stdout.flush()
            if i + 1 == len(SIZES):
                file_path = args.excel_file
                if not os.path.exists(file_path):
                    df = DataFrame(
                        np.zeros((GPU_NUM, GPU_NUM)),
                        # index=[i for i in range(GPU_NUM)],
                        columns=[i for i in range(GPU_NUM)],
                    )
                    df.to_excel(file_path, sheet_name='Sheet1')
                df = pd.read_excel(file_path, index_col=0)
                df[GPU_pairs[0][1]][GPU_pairs[0][0]] = BD
                DataFrame(df).to_excel(file_path, sheet_name='Sheet1')
        
        del a, b
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.barrier()
            
            
def net_test(rank, world_size, args):
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank, group_name='net_test')
    gpc.set_world_size(world_size=world_size)
    gpc.set_local_rank(local_rank=rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
      
    # GPUIDs = [int(i) for i in args.gpuids.split(',')]
    
    GPU_XXX = []
    for src in range(world_size):
        for dst in range(world_size):
            if src != dst:
                GPU_XXX.append([(src, dst), (dst, src)])
    
    GPU_pairs = [
        (1, 3),
        (2, 3),
        # (2, 3),
    ]
    # evaluate_GPU_pairs(GPU_pairs, rank, args)
    for GPU_pairs in GPU_XXX:
        evaluate_GPU_pairs(GPU_pairs, rank, args)
        
        

def main():
    args = get_args()
    torch.multiprocessing.spawn(net_test, nprocs=args.gpus, args=(args.gpus, args))

if __name__ == '__main__':
    main()