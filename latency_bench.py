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

import torch.distributed
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
from utils.cb_communicator import Cb_Communicator
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

MSG_SIZES = [
    # BYTE_MULTPLE_UP,
    # BYTE_MULTPLE_UP * 4,
    BYTE_MULTPLE_UP * 8,        # 8KB, S=32 * bs=1 * Nh=1 * D=128 * 2B = 8KB (intra-machine start point)
    BYTE_MULTPLE_UP * 16,
    BYTE_MULTPLE_UP * 32,
    BYTE_MULTPLE_UP * 64,       # 64KB, S=256 * bs=1 * Nh=1 * D=128 * 2B = 64KB (inter-machine start point)
    BYTE_MULTPLE_UP * 128,
    BYTE_MULTPLE_UP * 256,
    BYTE_MULTPLE_UP * 512,
    pow(BYTE_MULTPLE_UP, 2),            # 1MB
    pow(BYTE_MULTPLE_UP, 2) * 2,
    pow(BYTE_MULTPLE_UP, 2) * 4,
    pow(BYTE_MULTPLE_UP, 2) * 8,
    pow(BYTE_MULTPLE_UP, 2) * 16,
    pow(BYTE_MULTPLE_UP, 2) * 32,
    pow(BYTE_MULTPLE_UP, 2) * 64,
    pow(BYTE_MULTPLE_UP, 2) * 128,
    pow(BYTE_MULTPLE_UP, 2) * 256,
    pow(BYTE_MULTPLE_UP, 2) * 512,
    pow(BYTE_MULTPLE_UP, 3),         # 1GB
    # pow(BYTE_MULTPLE_UP, 3) * 2,
    # pow(BYTE_MULTPLE_UP, 3) * 4,     # 4GB
    # pow(BYTE_MULTPLE_UP, 3) * 16,    # 16GB
]

MSG_SIZES = [
    1,
    # 2,
    # 4,
    # 8,
    # 16,
    32,
    # 64,
    # 128,
    # 256,
    # 512,
    # BYTE_MULTPLE_UP,
    # BYTE_MULTPLE_UP * 4,
    # BYTE_MULTPLE_UP * 16,
    # BYTE_MULTPLE_UP * 64,
    # BYTE_MULTPLE_UP * 256,
    # pow(BYTE_MULTPLE_UP, 2),           
    # pow(BYTE_MULTPLE_UP, 2) * 4,   
    # pow(BYTE_MULTPLE_UP, 2) * 16,
    pow(BYTE_MULTPLE_UP, 2) * 64,        
]

# GPU_NUM = 8
WARM_UP = 5
TIMES = 10
TIMES = 100
# TIMES = 1000

def parse_args():
    parser = argparse.ArgumentParser(description='Conflict Benchmark Arguments',
                                    allow_abbrev=True)
    parser.add_argument('--backend', type=str, default='NCCL',
                       help='Communication backend.')
    parser.add_argument('--config', type=str, default=None,
                       help='Json config file of conflict patterns.')
    parser.add_argument('--profiler-with-tensorboard', action='store_true', default=False, help='whether to profile with tensorboard')
    parser.add_argument('--tb-dir', default=None, type=str, help='where to storage tensorboard files')
    parser.add_argument('--comm-module', default='torch-distributed', choices=['torch-distributed', 'raw-nccl'], type=str, help='which module to use for communication')

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
    
        # print(f'MSG_SIZES: {MSG_SIZES}')
    row_a = torch.empty(max(MSG_SIZES), dtype=torch.int8).cuda()
    row_b = torch.empty(max(MSG_SIZES), dtype=torch.int8).cuda()
    
    assert PROC_INFO['world_size'] == 2
    # Create gloo groups for each pair
    group_dict = {}
    for i in range(PROC_INFO['world_size']):
        for j in range(i + 1, PROC_INFO['world_size']):
            key = tuple(sorted([i, j]))
            if key not in group_dict.keys():
                group_dict[key] = torch.distributed.new_group(key, backend='gloo')
    # Create communicator
    communicator = Cb_Communicator(PROC_INFO, args, group_dict)

    _t = torch.empty((10 * 1024, 10 * 1024), dtype=torch.bfloat16, device=torch.cuda.current_device())
    comm_stream = torch.cuda.current_stream()
    
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    
    for i, SIZE in enumerate(MSG_SIZES):
        a = row_a[: SIZE]
        b = row_b[: SIZE]
        communicator.clear_ops()
        
        # For Sync:
        communicator.all_reduce(_t, comm_stream)    # placeholder_op
        communicator.execute_comm_ops(barrier=False, light_barrier=False)
        communicator.clear_ops()
        if rank == 1:
            torch.matmul(_t, _t)
        
        # ops1:
        # communicator.all_reduce(a, comm_stream)
        # ops2:
        if rank == 0:
            communicator.send(a, 1, comm_stream)
        if rank == 1:
            communicator.recv(a, 0, comm_stream)

        for _ in range(WARM_UP):
            communicator.execute_comm_ops(barrier=False, light_barrier=False)
        
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
                    # execute_comm_ops(ops, barrier=False, light_barrier=False)
                    communicator.execute_comm_ops(barrier=False, light_barrier=False)
                    if (_ + 1) % BARRIER_FREQ == 0:
                        torch.cuda.synchronize()
                        dist.barrier()
                    prof.step()
                t1 = time.time()
        else:
            event_start.record()
            for _ in range(TIMES):
                communicator.execute_comm_ops(barrier=False, light_barrier=False)
            event_end.record()
            torch.cuda.synchronize()
            t_d = event_start.elapsed_time(event_end) / 1000 # s
            dist.barrier()
    
        print_rank_0(f'SIZE {convert_size(SIZE)}, time {t_d:.3e} s, time/iter {t_d/TIMES:.3e} s')


        del a, b
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.barrier()
    
if __name__ == '__main__':
    main()