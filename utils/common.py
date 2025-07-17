import torch
import torch.distributed as dist
import math
import os
import socket

BYTE_MULTPLE_UP = 1024
BYTE_MULTPLE_DOWN = 1000

def calc_comm_vol_a2a(t: torch.Tensor, world_size): # -> GB
    return t.nelement() * t.element_size() * (world_size - 1) / world_size / pow(1024, 3)

def calc_comm_vol_ag(t: torch.Tensor, world_size):  # -> GB
    return t.nelement() * t.element_size() * (world_size - 1) / pow(1024, 3)

def execute_comm_ops(ops, barrier=False, light_barrier=False):
    if len(ops) > 0:
        # works = dist.batch_isend_irecv(ops)
        works = [op() for op in ops]
        for work in works:
            work.wait()
    if barrier:
        torch.cuda.synchronize()                            # 每一轮是否barrier对性能没有影响
        dist.barrier()
    elif light_barrier:
        torch.cuda.synchronize() 

def append_to_file(message: str, output_file: str, end='\n'):
    if output_file:
        with open(output_file, 'a') as f:
            f.write(message + end)
    
def print_rank_0(message, output_file: str = None):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
            append_to_file(message, output_file)
    else:
        print(message, flush=True)
        append_to_file(message, output_file)

# Helper function to pretty-print message sizes
def convert_size(size_bytes, infix=' ', suffix='B'):
    if size_bytes == 0:
        return "0" + suffix
    # size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(size_bytes, BYTE_MULTPLE_UP)))
    p = math.pow(BYTE_MULTPLE_UP, i)
    s = int(round(size_bytes / p, 2))
    return "%s%s%s%s" % (s, infix, size_name[i], suffix)

# Helper function to pretty-print message sizes
def convert_throughput(size_bytes, round_=3):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, BYTE_MULTPLE_DOWN)))
    p = math.pow(BYTE_MULTPLE_DOWN, i)
    s = round(size_bytes / p, round_)
    return "%s %s" % (s, size_name[i])

def get_proc_info():
    if os.getenv('SLURM_PROCID', None) is not None:    # launch with Slurm
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
        
    elif os.getenv('OMPI_COMM_WORLD_RANK', None) is not None: # launch with OpenMPI
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # ip = os.environ['SLURM_STEP_NODELIST']
        ip = None
        hostname = socket.gethostname()
        hostip = socket.gethostbyname(hostname)
        clustername = os.getenv('CLUSTER_NAME', 'Unknown Cluster')
        # nodeid = int(os.environ['SLURM_NODEID'])
        nodeid = None
        # nodename = os.environ['SLURMD_NODENAME']
        nodename = None
        # tasks_per_node = os.environ['SLURM_TASKS_PER_NODE']
        tasks_per_node = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        
    else:
        raise Exception("Unknown Launcher !!!")
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
    # print(f'proc_info: {proc_info}')
    return proc_info

def init_cluster(PROC_INFO, MASTER_ADDR, MASTER_PORT, backend):
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    # print(f'init_method: {init_method}', flush=True)
    # print(f'world_size: {PROC_INFO["world_size"]}')
    # print(f'MASTER_ADDR: {MASTER_ADDR}')
    # print(f'MASTER_PORT: {MASTER_PORT}')
    # print(f'rank: {PROC_INFO["rank"]}, local_rank: {PROC_INFO["local_rank"]}', flush=True)
    dist.init_process_group(backend=backend, 
                            # init_method=init_method, # [NOTE]: Not necessary, configs like world_size, rank, can be read from env !!!
                            world_size=PROC_INFO['world_size'], 
                            rank=PROC_INFO['rank'])
    # print(f'init_process_group done !!!', flush=True)
    if torch.cuda.is_available():
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
        # torch.cuda.set_device(0)
        torch.cuda.set_device(PROC_INFO['local_rank'])
    else:
        print(f'[Error]: CUDA is not available !!!')
        exit(- 1)
    