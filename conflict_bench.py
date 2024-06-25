# from utils.argparser import get_args
# from utils.parallel_context import gpc
# # from ..utils.parallel_context import gpc
# import torch.distributed as dist
# import time
import functools as F
from functools import partial
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
    parser.add_argument('--profiler-with-tensorboard', action='store_true', default=False, help='whether to profile with tensorboard')
    parser.add_argument('--tb-dir', default=None, type=str, help='where to storage tensorboard files')

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
        # 1 Use minimized ProcessGroup, deduplicate and sort
        ranks = [rank for pair in pattern for rank in pair]
        ranks = sorted(list(set(ranks)))
        pgs = [torch.distributed.new_group(ranks)]
        
        # # 2 Use different ProcessGroup for each pair
        # pgs = []
        # for pair in pattern:
        #     pg = torch.distributed.new_group(pair)
        #     pgs.append(pg)
        
        for i, SIZE in enumerate(MSG_SIZES):
            a = row_a[: SIZE]
            b = row_b[: SIZE]
            
            ops = []
            for i, pair in enumerate(pattern):
                pg = pgs[min(i, len(pgs) - 1)]
                if rank == pair[0]:
                    ops.append(dist.P2POp(dist.isend, a, pair[1], group=pg))
                    # ops.append(partial(dist.isend, a, pair[1], group=pg))
                if rank == pair[1]:
                    ops.append(dist.P2POp(dist.irecv, b, pair[0], group=pg))
                    # ops.append(partial(dist.irecv, b, pair[0], group=pg))
            
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(WARM_UP):
                execute_comm_ops(ops, barrier=True)
            
            if args.profiler_with_tensorboard:
                args.tb_profiled = True
                is_runned = True
                BARRIER_FREQ = 4
                WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
                TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
                TRACE_NAME = f'{os.environ["TRACE_NAME"]}_w{world_size}_r{rank}'
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        dir_name=f'{args.tb_dir}', 
                        worker_name=TRACE_NAME,
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    torch.cuda.synchronize()
                    dist.barrier()
                    t0 = time.time()
                    for _ in range(TOTAL_TURNS):
                        execute_comm_ops(ops, barrier=False, light_barrier=False)
                        if (_ + 1) % BARRIER_FREQ == 0:
                            torch.cuda.synchronize()
                            dist.barrier()
                        prof.step()
                    t1 = time.time()
            else:
                torch.cuda.synchronize()
                dist.barrier()
                t0 = time.time()
                for _ in range(TIMES):
                    execute_comm_ops(ops, barrier=False, light_barrier=False)
                torch.cuda.synchronize()
                dist.barrier()
                t1 = time.time()
        
            t_d = t1 - t0
            # calc = len(GPU_pairs) * reduce((lambda x,y: x*y), SIZE) * 4 * TIMES / pow(1024, 3) # GB
            calc = SIZE * len(pattern) * TIMES # B
            BD = calc / t_d
            print_rank_0(f'SIZE {convert_size(SIZE)}, REAL_BD {convert_throughput(BD)}/s, time {round(t_d, 4)} s, comm_vol {convert_throughput(calc)}')

        del a, b
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.barrier()
    
if __name__ == '__main__':
    main()