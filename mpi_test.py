print(f'hello world 1!', flush=True)
from mpi4py import MPI
print(f'hello world 2!', flush=True)
import os
import socket

def main():
    pass
    # rank = MPI.COMM_WORLD.Get_rank()
    # local_rank = rank
    # world_size = MPI.COMM_WORLD.Get_size()
    # # ip = os.environ['SLURM_STEP_NODELIST']
    # ip = None
    # hostname = socket.gethostname()
    # hostip = socket.gethostbyname(hostname)
    # clustername = os.environ['CLUSTER_NAME']
    # # nodeid = int(os.environ['SLURM_NODEID'])
    # nodeid = None
    # # nodename = os.environ['SLURMD_NODENAME']
    # nodename = None
    # # tasks_per_node = os.environ['SLURM_TASKS_PER_NODE']
    # tasks_per_node = world_size
    # proc_info = {
    #     'clustername': clustername,
    #     'hostname': hostname,
    #     'nodename': nodename,
    #     'nodeid': nodeid,
    #     'world_size': world_size,
    #     'tasks_per_node': tasks_per_node,
    #     'rank': rank,
    #     'local_rank': local_rank,
    #     'hostip': hostip,
    #     'ip': ip,
    # }
    # print(f'proc_info: {proc_info}')
    
if __name__ == '__main__':
    main()