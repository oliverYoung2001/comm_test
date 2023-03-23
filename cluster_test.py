import torch
from utils.argparser import get_args
from utils.parallel_context import gpc
# from ..utils.parallel_context import gpc
import torch.distributed as dist
import time
from functools import reduce 
import os
import socket
import sys
# import mpi4py as MPI

MASTER_ADDR = '172.23.18.3'     # nico3 ip, ip of host of rank0
MASTER_PORT = 10378             # idle port

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

def mono_p2p(rank, SIZE, TIMES, GPUID):
    # p2p; 0 -> 1
    if rank == 0:
        a = torch.randn(SIZE, dtype=torch.float32).cuda()
    if rank == 1:
        a = torch.empty(SIZE, dtype=torch.float32).cuda()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[GPUID])

    t0 = time.time()
    if rank == 0:
        for _ in range(TIMES):
            dist.send(a, 1)
    if rank == 1:
        for _ in range(TIMES):
            dist.recv(a, 0)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[GPUID])        # OK ! 手动指定当前进程对应的devices
    
    t1 = time.time()
    if rank == 0:
        t_d = t1 - t0
        print(f'rank: {rank}, time: {t_d}')
        calc = a.nelement() * a.element_size() * TIMES / pow(1024, 3) # GB
        print(f'rank: {rank}, BD: {calc / t_d} GB/s')
        sys.stdout.flush()

def bi_p2p_async(rank, SIZE, TIMES, GPUID):
    # p2p; 0 -> 1, 1 -> 0
    a = torch.randn(SIZE, dtype=torch.float32).cuda()
    b = torch.empty(SIZE, dtype=torch.float32).cuda()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[GPUID])
    t0 = time.time()
    for _ in range(TIMES):
        ops = []
        if rank == 0:
            ops.append(dist.P2POp(dist.isend, a, 1))
            ops.append(dist.P2POp(dist.irecv, b, 1))
        if rank == 1:
            ops.append(dist.P2POp(dist.irecv, b, 0))
            ops.append(dist.P2POp(dist.isend, a, 0))
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[GPUID])        # OK ! 手动指定当前进程对应的devices
    
    t1 = time.time()
    if rank == 0:
        t_d = t1 - t0
        print(f'rank: {rank}, time: {t_d}')
        calc = a.nelement() * a.element_size() * TIMES / pow(1024, 3) # GB
        print(f'rank: {rank}, BD: {calc / t_d} GB/s')
        sys.stdout.flush() 

def bi_p2p_sync(rank, SIZE, TIMES, GPUID):
    # p2p; 0 -> 1, 1 -> 0
    a = torch.randn(SIZE, dtype=torch.float32).cuda()
    b = torch.empty(SIZE, dtype=torch.float32).cuda()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[GPUID])
    t0 = time.time()
    for _ in range(TIMES):
        if rank == 0:
            dist.send(a, 1)
            dist.recv(b, 1)
        if rank == 1:
            dist.recv(b, 0)
            dist.send(a, 0)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[GPUID])        # OK ! 手动指定当前进程对应的devices
    
    t1 = time.time()
    if rank == 0:
        t_d = t1 - t0
        print(f'rank: {rank}, time: {t_d}')
        calc = a.nelement() * a.element_size() * TIMES / pow(1024, 3) # GB
        print(f'rank: {rank}, BD: {calc / t_d} GB/s')
        sys.stdout.flush() 
             
            
def cluster_net_test(rank, local_rank, world_size, tasks_per_node, nodeid, nodes, args):
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank, group_name='cluster_net_test')
    gpc.set_world_size(world_size=world_size)
    gpc.set_rank(rank=rank)
    gpc.set_local_rank(local_rank=local_rank)
    
    GPUIDs = list(map((lambda x: int(x)), args.gpuids.split(',')))
    print(f'GPUIDs: {GPUIDs}')
    assert len(GPUIDs) == nodes
    GPUID = GPUIDs[nodeid]
    
    # print(f'nodeid: {nodeid}, GPUID: {GPUID}')
    if torch.cuda.is_available():
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
        # torch.cuda.set_device(0)
        torch.cuda.set_device(GPUID)
    
    TIMES = 20
    
    for SIZE in SIZES:
        if rank == 0:
            print(f'SIZE: {SIZE}')
            sys.stdout.flush()
            
        # mono_p2p(rank, SIZE, TIMES, GPUID)
        # bi_p2p_async(rank, SIZE, TIMES, GPUID)
        bi_p2p_sync(rank, SIZE, TIMES, GPUID)
        
            
    dist.destroy_process_group()
        

def main():
    args = get_args()
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    ip = os.environ['SLURM_STEP_NODELIST']
    hostname = socket.gethostname()
    hostip = socket.gethostbyname(hostname)
    clustername = os.environ['SLURM_CLUSTER_NAME']
    nodeid = int(os.environ['SLURM_NODEID'])
    nodename = os.environ['SLURMD_NODENAME']
    tasks_per_node = os.environ['SLURM_TASKS_PER_NODE']
    proc_info = {
        'clustername': clustername,
        'hostname': hostname,
        'nodename': nodename,
        'nodeid': nodeid,
        'world_size': world_size,
        'tasks_per_node': tasks_per_node,
        'rank': rank,
        'local_rank': local_rank,
        'hostip': hostip,
        'ip': ip,
    }
    # print(f'{proc_info}')
    # sys.stdout.flush()
    nodes = 2
    cluster_net_test(rank, local_rank, world_size, tasks_per_node, nodeid, nodes, args)
    # torch.multiprocessing.spawn(net_test, nprocs=args.gpus, args=(args.gpus, args))

if __name__ == '__main__':
    main()