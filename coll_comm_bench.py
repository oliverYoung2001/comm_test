import torch
from utils.comm_impl import *
from utils.common import *
import torch.distributed as dist
import time
import os
import socket
import math
import argparse
import json

# BYTE_MULTPLE_UP = 1024
# BYTE_MULTPLE_DOWN = 1000

PROC_INFO: dict

COLL_COMMs = ['b', 'r', 'g', 's', 'ag', 'rs', 'ar', 'a2a']
COLL_COMMs = ['ag', 'rs', 'ar']
# COLL_COMMs = ['a2a']
abbr2full = {
    'b': 'broadcast',
    'r': 'reduce',
    'g': 'gather',
    's': 'scatter',
    'ag': 'allgather',
    'rs': 'reducescatter',
    'ar': 'allreduce',
    'a2a': 'alltoall',
}

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
    # 2 * 6688350208 // 16,   # 7b for 16 GPU
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


WARMUP = 2
TIMES = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Conflict Benchmark Arguments',
                                    allow_abbrev=True)
    parser.add_argument('--backend', type=str, default='NCCL',
                       help='Communication backend.')
    parser.add_argument('--output', type=str, default=None,
                       help='Prof results of json format.')
    args = parser.parse_args()
    return args


def make_args(coll_comm, msg_size, src, dst, group):
    world_size = torch.distributed.get_world_size()
    args = {}
    # args['async_op'] = False
    if coll_comm not in ['ar', 'a2a']:
        # tensor = torch.randint(0, 100, size=(msg_size, ), dtype=torch.int8).cuda()
        tensor = torch.empty((msg_size, ), dtype=torch.int8).cuda()
        # tensor_list = [torch.zeros(size=(msg_size, ), dtype=torch.int8).cuda() for _ in range(PROC_INFO['world_size'])]
        tensor_list = [torch.empty(size=(msg_size, ), dtype=torch.int8).cuda() for _ in range(PROC_INFO['world_size'])]
    if coll_comm == 'b':
        args = {
            'tensor': tensor,
            'src': src,
            'group': group,
            'async_op': False,
        }
    elif coll_comm == 'r':
        args = {
            'tensor': tensor, 
            'dst': dst, 
            'group': group,
            'async_op': False,
        }
    elif coll_comm == 'g':
        # tensor, gather_list, dst, group, async_op
        args = {
            'tensor': tensor, 
            'gather_list': tensor_list if PROC_INFO['rank'] == dst else None,
            'dst': dst, 
            'group': group,
            'async_op': False,
        }
    elif coll_comm == 's':
        # tensor, scatter_list, src, group, async_op
        args = {
            'tensor': tensor, 
            'scatter_list': tensor_list if PROC_INFO['rank'] == src else None,
            'src': src, 
            'group': group,
            'async_op': False,
        }
    elif coll_comm == 'ag':
        # tensor_list, tensor, group, async_op
        args = {
            'tensor_list': tensor_list,
            'tensor': tensor, 
            'group': group,
            'async_op': False,
        }
        pass
    elif coll_comm == 'rs':
        # tensor, tensor_list, group, async_op
        args = {
            'tensor': tensor, 
            'tensor_list': tensor_list,
            'group': group,
            'async_op': False,
        }
        pass
    elif coll_comm == 'ar':
        # tensor, group, async_op
        tensor = torch.empty((msg_size * world_size, ), dtype=torch.int8).cuda()
        args = {
            'tensor': tensor, 
            'group': group,
            'async_op': False,
        }
        pass
    elif coll_comm == 'a2a':
        # output_list, input_list, group, async_op
        tensor_list = [torch.empty(size=(msg_size // world_size, ), dtype=torch.int8).cuda() for _ in range(PROC_INFO['world_size'])]
        # input_list = [torch.zeros(size=(msg_size, ), dtype=torch.int8).cuda() for _ in range(PROC_INFO['world_size'])]
        input_list = [torch.empty(size=(msg_size // world_size, ), dtype=torch.int8).cuda() for _ in range(PROC_INFO['world_size'])]
        args = {
            'output_list': tensor_list,
            'input_list': input_list,
            'group': group,
            'async_op': False,
        }
        pass
    
    return args


def calc_bw_log(comm_op, size, duration):
    n = PROC_INFO['world_size']
    tput = 0
    busbw = 0
    if comm_op == "alltoall":
        # size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "allgather" or comm_op == "reducescatter":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "allreduce":
        size *= n
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "send" or comm_op == "recv" or comm_op == "isend" or comm_op == "irecv" or comm_op == "broadcast" or comm_op == "reduce" or comm_op == "gather" or comm_op == "scatter" or comm_op == "barrier":
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0("[Error]: Wrong comm_op specified !!!")  # noqa: F821
        exit(- 2)

    # convert to Gbps
    # tput *= 8
    # busbw *= 8

    # tput /= pow(BYTE_MULTPLE_DOWN, 3)
    # busbw /= pow(BYTE_MULTPLE_DOWN, 3)

    return tput, busbw


def main():
    args = parse_args()
    global PROC_INFO
    PROC_INFO = get_proc_info()
    MASTER_ADDR = os.getenv('MASTER_ADDR', None)
    MASTER_PORT = os.getenv('MASTER_PORT', None)
    init_cluster(PROC_INFO, MASTER_ADDR, MASTER_PORT, backend='NCCL')
    
    torch.cuda.synchronize()
    dist.barrier(group=None, async_op=False)
    print_rank_0(f'[INFO]: Cluster init done !!!')
    
    BW_TABLE = {'meta': MSG_SIZES}
    for coll_comm in COLL_COMMs:
        coll_table = BW_TABLE[coll_comm] = {
            'tput': [],
            'busbw': [],
        }
        print_rank_0(f'COLL_COMM: {abbr2full[coll_comm]}')
        COMM_FUNC =  globals()[abbr2full[coll_comm]]
        for msg_size in MSG_SIZES:
            t_s = time.time()
            kwargs = make_args(coll_comm, msg_size, 0, 0, group=None)
            t_e = time.time()
            # print_rank_0(f'mem alloc t_d: {round(t_e - t_s, 4)} s')

            torch.cuda.synchronize()
            dist.barrier(group=None, async_op=False)
            for _ in range(WARMUP):
                COMM_FUNC(**kwargs)
            
            torch.cuda.synchronize()
            dist.barrier(group=None, async_op=False)

            t_s = time.time()
            for _ in range(TIMES):
                COMM_FUNC(**kwargs)
                torch.cuda.synchronize()
                dist.barrier(group=None, async_op=False)
            torch.cuda.synchronize()
            dist.barrier(group=None, async_op=False)
            t_e = time.time()
            t_d = t_e - t_s
            tput, busbw = calc_bw_log(abbr2full[coll_comm], msg_size, t_d / TIMES)  # B/s
            print_rank_0(f'msg_size: {convert_size(msg_size)}, t_d: {round(t_d, 4)} s, t_d/r: {round(t_d / TIMES, 4)}, ' \
                         f'tput: {convert_throughput(tput)}/s, busbw: {convert_throughput(busbw)}/s')
            torch.cuda.empty_cache()
            coll_table['tput'].append(tput / pow(BYTE_MULTPLE_DOWN, 3))
            coll_table['busbw'].append(busbw / pow(BYTE_MULTPLE_DOWN, 3))
    with open(args.output, 'w') as f:
        f.seek(0)  #定位
        f.truncate()
        json.dump(BW_TABLE, f)        

if __name__ == '__main__':
    main()