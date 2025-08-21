#!/bin/bash

if [ -z $MASTER_ADDR ]
then
    if [ -z $SLURM_JOB_ID ]
    then
        export MASTER_ADDR=localhost
    else
        export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
    fi
fi
if [ -z $MASTER_PORT ]
then
    export MASTER_PORT=12580
fi

if [ ! -z $OMPI_COMM_WORLD_RANK ]   # OpenMPI
then
    export RANK=$OMPI_COMM_WORLD_RANK
    export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
    export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
elif [ ! -z $SLURM_PROCID ] # SLURM
then
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NPROCS
    export LOCAL_RANK=$SLURM_LOCALID
    # export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
else
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
fi

# echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT, RANK: $RANK, WORLD_SIZE: $WORLD_SIZE"
# echo "NCCL_AVOID_RECORD_STREAMS: $NCCL_AVOID_RECORD_STREAMS"

# # Mask out useless NIC
# if [ $(hostname) == "g0002" ] || [ $(hostname) == "g0004" ] || [ $(hostname) == "g0029" ]; then
#     export NVSHMEM_HCA_LIST=mlx5_0,mlx5_3,mlx5_4,mlx5_5
# else
#     export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_3,mlx5_4
# fi
export NVSHMEM_HCA_LIST="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
export NCCL_IB_HCA=$NVSHMEM_HCA_LIST

# Bind core
if [ "$LOCAL_RANK" -le 3 ]; then numa=0; else numa=1; fi

#   Method1:
if [ $(hostname) == "g0297" ]; then
    if [ "$LOCAL_RANK" -le 3 ]; then cpus="0-31,64-95"; else cpus="32-63,96-127"; fi
elif [ $(hostname) == "g0278" ]; then
    if [ "$LOCAL_RANK" -le 3 ]; then cpus="0-47,96-143"; else cpus="48-95,144-191"; fi
else
    if [ "$LOCAL_RANK" -le 3 ]; then cpus="0-43,88-131"; else cpus="44-87,132-175"; fi
fi
# #   Method2
# cpus=$((LOCAL_RANK*8))-$(((LOCAL_RANK+1)*8-1)),$((LOCAL_RANK*8+64))-$(((LOCAL_RANK+1)*8-1+64))
# #   Method3
# if [ $((LOCAL_RANK/2)) == 1 ]; then
#     cpus=$((LOCAL_RANK*16+32))-$(((LOCAL_RANK%4+1)*16+31))
# elif [ $((LOCAL_RANK/2)) == 2 ]; then
#     cpus=$((LOCAL_RANK*16-32))-$(((LOCAL_RANK+1)*16-33))
# else
#     cpus=$((LOCAL_RANK*16))-$(((LOCAL_RANK+1)*16-1))
# fi
# Conclusion: Method1~3 are the same for performance.

echo "HostName: $(hostname), LOCAL_RANK: $LOCAL_RANK, cpus: $cpus, numa: $numa $(cat /proc/self/status | grep Cpus_allowed_list)"

# # set -x
# numactl --physcpubind=$cpus --membind=$numa \
$@
