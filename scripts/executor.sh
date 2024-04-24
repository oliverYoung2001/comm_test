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
    export MASTER_PORT=$((RANDOM % 12000 + 10000))
fi

if [ ! -z $SLURM_PROCID ]
then
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NPROCS
    # export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
else
    export RANK=0
    export WORLD_SIZE=1
fi

# echo "MASTER_ADDR: $MASTER_ADDR"
# echo "RANK: $RANK"
# echo "WORLD_SIZE: $WORLD_SIZE"
# echo "NCCL_AVOID_RECORD_STREAMS: $NCCL_AVOID_RECORD_STREAMS"

# set -x
exec $@
