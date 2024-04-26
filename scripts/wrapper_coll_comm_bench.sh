#!/bin/bash

export GPU_NUMs="4 8"
export GPU_NUMs="2 4 8"
export GPU_NUMs="16"
# nico:
# qy:
export CLUSTER_NAME=qy
PARTITION=gpu4-low
HOST="g4004"
PARTITION=gpu3-2-low
HOST=None

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

done