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

if [ ! -z $OMPI_COMM_WORLD_RANK ]
then
    export RANK=$OMPI_COMM_WORLD_RANK
    export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
    export localrank=$OMPI_COMM_WORLD_LOCAL_RANK
elif [ ! -z $SLURM_PROCID ]
then
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NPROCS
    # export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
else
    export RANK=0
    export WORLD_SIZE=1
fi

echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT, RANK: $RANK, WORLD_SIZE: $WORLD_SIZE"
# echo "NCCL_AVOID_RECORD_STREAMS: $NCCL_AVOID_RECORD_STREAMS"

# set -x
exec $@
