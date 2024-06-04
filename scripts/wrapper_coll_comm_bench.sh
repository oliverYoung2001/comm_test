#!/bin/bash

export GPU_NUMs="4 8"
export GPU_NUMs="2 4 8"
export GPU_NUMs="16"
# export GPU_NUMs="24"
export GPU_NUMs="32"
# export GPU_NUMs="64"

# nico:
export CLUSTER_NAME=nico
PARTITION=Mix
HOST="nico1,nico2"
# qy:
export CLUSTER_NAME=qy
PARTITION=gpu4-low
HOST="g4007,g4008"
HOST="g4005,g4006,g4007"
# HOST="g4002,g4003"
# PARTITION=gpu3-2-low
# HOST=None
# wq:
export CLUSTER_NAME=wq
PARTITION=Nvidia_A800
HOST="gpu11,gpu12"
HOST="gpu[11-14]"


export MASTER_PORT=$((RANDOM % 12000 + 10000))

EXECUBLE=coll_comm_bench.py

echo "PARTITION: ${PARTITION}"
echo "HOST: ${HOST}"

for GPU_NUM in $GPU_NUMs; do
echo "GPU_NUM: $GPU_NUM"
OUTPUT=./prof_data/coll_comm_bench_${GPU_NUM}_${HOST}.json

PROC_NUM=$GPU_NUM
if [ $GPU_NUM -le 8 ]; then
   NNODES=1
else
   NNODES=$(($GPU_NUM / 8))
fi
NTASK_PER_NODE=`expr $PROC_NUM / $NNODES`

MEM_PER_CPU=256G
MEM_PER_NODE=256G
# --mem-per-cpu $MEM_PER_CPU \
# --mem $MEM_PER_NODE \
SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node $NTASK_PER_NODE \
--gres=gpu:$NTASK_PER_NODE \
--mem $MEM_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

set -x
srun $SLURM_ARGS \
./scripts/executor.sh \
python $EXECUBLE \
    --output $OUTPUT \

set +x
done