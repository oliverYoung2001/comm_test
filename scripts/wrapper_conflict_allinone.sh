#!/bin/bash

set -x

# qy:
PARTITION=gpu4-low
HOST="g4003"
HOST=None
MAX_GPU_NUM=8


if [ $MAX_GPU_NUM -le 8 ]; then
   NNODES=1
else
   NNODES=$(($MAX_GPU_NUM / 8))
fi
NGPU_PER_NODE=`expr $MAX_GPU_NUM / $NNODES`


echo "NNODES: $NNODES"
echo "NGPU_PER_NODE: $NGPU_PER_NODE"
# SLURM ARGS:
# MEM_PER_CPU=256G # 
# MEM_PER_NODE=256G
# --mem-per-cpu $MEM_PER_CPU \
# --cpus-per-task $NGPU_PER_NODE \
SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node 1 \
--cpus-per-task 32 \
--gres=gpu:$NGPU_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi
# salloc -n $MAX_GPU_NUM
srun $SLURM_ARGS \
./scripts/conflict_allinone.sh
