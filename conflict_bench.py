# from utils.argparser import get_args
# from utils.parallel_context import gpc
# # from ..utils.parallel_context import gpc
# import torch.distributed as dist
# import time
import functools as F
# from functools import reduce 
# import sys
# import pandas as pd
# from pandas import DataFrame
# import os
# import numpy as np
# pd.options.mode.chained_assignment = None  # default='warn'
# from utils.utils import execute_comm_ops


import warnings
warnings.filterwarnings("ignore")   # [NOTE]: disable all warning
import time
import os
import socket
import math
import argparse
import json
import numpy
from utils.comm_impl import *
from utils.common import *
import torch
import torch.distributed as dist
# BATCH = 1

MSG_SIZES = [
    # BYTE_MULTPLE_UP,
    # BYTE_MULTPLE_UP * 4,
    # BYTE_MULTPLE_UP * 16,
    # BYTE_MULTPLE_UP * 64,
    # BYTE_MULTPLE_UP * 256,
    # pow(BYTE_MULTPLE_UP, 2),
    # pow(BYTE_MULTPLE_UP, 2) * 4,
    pow(BYTE_MULTPLE_UP, 2) * 16,
    pow(BYTE_MULTPLE_UP, 2) * 64,
    pow(BYTE_MULTPLE_UP, 2) * 256,
    pow(BYTE_MULTPLE_UP, 3),         # 1GB
    # pow(BYTE_MULTPLE_UP, 3) * 4,     # 4GB
    # pow(BYTE_MULTPLE_UP, 3) * 16,    # 16GB
]

# MSG_SIZES = [
#     # BYTE_MULTPLE_UP,
#     # BYTE_MULTPLE_UP * 4,
#     # BYTE_MULTPLE_UP * 16,
#     # BYTE_MULTPLE_UP * 64,
#     # BYTE_MULTPLE_UP * 256,
#     pow(BYTE_MULTPLE_UP, 2),            # 1MB
#     pow(BYTE_MULTPLE_UP, 2) * 2,
#     pow(BYTE_MULTPLE_UP, 2) * 4,
#     pow(BYTE_MULTPLE_UP, 2) * 8,
#     pow(BYTE_MULTPLE_UP, 2) * 16,
#     pow(BYTE_MULTPLE_UP, 2) * 32,
#     pow(BYTE_MULTPLE_UP, 2) * 64,
#     pow(BYTE_MULTPLE_UP, 2) * 128,
#     pow(BYTE_MULTPLE_UP, 2) * 256,
#     pow(BYTE_MULTPLE_UP, 2) * 512,
#     pow(BYTE_MULTPLE_UP, 3),         # 1GB
#     pow(BYTE_MULTPLE_UP, 3) * 2,
#     pow(BYTE_MULTPLE_UP, 3) * 4,     # 4GB
#     # pow(BYTE_MULTPLE_UP, 3) * 16,    # 16GB
# ]

# GPU_NUM = 8
WARM_UP = 5
TIMES = 10
# TIMES = 100

def parse_args():
    parser = argparse.ArgumentParser(description='Conflict Benchmark Arguments',
                                    allow_abbrev=True)
    parser.add_argument('--backend', type=str, default='NCCL',
                       help='Communication backend.')
    parser.add_argument('--config', type=str, default=None,
                       help='Json config file of conflict patterns.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    global PROC_INFO
    PROC_INFO = get_proc_info()
    MASTER_ADDR = os.getenv('MASTER_ADDR', None)
    MASTER_PORT = os.getenv('MASTER_PORT', None)
    init_cluster(PROC_INFO, MASTER_ADDR, MASTER_PORT, backend=args.backend)
    
    torch.cuda.synchronize()
    dist.barrier(group=None, async_op=False)
    print_rank_0(f'[INFO]: Cluster init done !!!')
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    with open(args.config, 'r') as f:
        patterns = json.load(f)
    row_a = torch.empty(max(MSG_SIZES), dtype=torch.int8).cuda()
    row_b = torch.empty(max(MSG_SIZES), dtype=torch.int8).cuda()

    for pattern in patterns:
        if max(max(pattern)) >= world_size:
            continue
        print_rank_0(f'{pattern}')
        ranks = [rank for pair in pattern for rank in pair]
        # Use minimized ProcessGroup, deduplicate and sort
        ranks = sorted(list(set(ranks)))
        # print_rank_0(f'ranks: {ranks}')
        pg = torch.distributed.new_group(ranks)
        for i, SIZE in enumerate(MSG_SIZES):
            a = row_a[: SIZE]
            b = row_b[: SIZE]
            
            ops = []
            for pair in pattern:
                if rank == pair[0]:
                    ops.append(dist.P2POp(dist.isend, a, pair[1], group=pg))        # 是否用相同的buffer对性能没有影响
                if rank == pair[1]:
                    ops.append(dist.P2POp(dist.irecv, b, pair[0], group=pg))
            
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
                execute_comm_ops(ops, barrier=False, light_barrier=False)
            torch.cuda.synchronize()
            dist.barrier()
            t1 = time.time()
        
            t_d = t1 - t0
            # calc = len(GPU_pairs) * reduce((lambda x,y: x*y), SIZE) * 4 * TIMES / pow(1024, 3) # GB
            calc = SIZE * len(pattern) * TIMES # B
            BD = calc / t_d
            print_rank_0(f'SIZE {convert_size(SIZE)}, REAL_BD {convert_throughput(BD)}/s, time {round(t_d, 4)} s, comm_vol {convert_throughput(calc)}')
            # print(f'rank:{rank}, SIZE {convert_size(SIZE)}, REAL_BD {convert_throughput(BD)}/s, time {round(t_d, 4)} s, comm_vol {convert_throughput(calc)}', flush=True)
            # if i + 1 == len(SIZES):
            #     file_path = args.excel_file
            #     if not os.path.exists(file_path):
            #         df = DataFrame(
            #             np.zeros((GPU_NUM, GPU_NUM)),
            #             # index=[i for i in range(GPU_NUM)],
            #             columns=[i for i in range(GPU_NUM)],
            #         )
            #         df.to_excel(file_path, sheet_name='Sheet1')
            #     df = pd.read_excel(file_path, index_col=0)
            #     df[GPU_pairs[0][1]][GPU_pairs[0][0]] = BD
            #     DataFrame(df).to_excel(file_path, sheet_name='Sheet1')
        
        del a, b
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.barrier()
    
if __name__ == '__main__':
    main()