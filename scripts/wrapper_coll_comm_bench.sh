#!/bin/bash

export GPU_NUMs="4 8"
export GPU_NUMs="2 4 8"
export GPU_NUMs="16"
export GPU_NUMs="24"
export GPU_NUMs="64"

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


# NCCL Args:
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=ERROR
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_LEVEL=0   # Disable GDR
export NCCL_IB_DISABLE=0
export NCCL_DEBUG_SUBSYS=NET

# export NCCL_AVOID_RECORD_STREAMS=0  # Disable Warning for P2P
# export CUDA_LAUNCH_BLOCKING=1

# # Launch with Slurm
# set -x
# srun $SLURM_ARGS \
# ./scripts/executor.sh \
# python $EXECUBLE \
#     --output $OUTPUT \

# Launch with MPI
GPU_NUM=32
HOST_CONFIG="g4005:8,g4006:8,g4007:8,g4008:8"
HOST_CONFIG="g3021:8,g3022:8,g3023:8,g3024:8"
# GPU_NUM=16
# HOST_CONFIG="g4007:8,g4008:8"
# # HOST_CONFIG="g3025:8,g4006:8"
# # HOST_CONFIG="g4003:8,g4006:8"
# HOST_CONFIG="g3021:8,g3022:8"
# GPU_NUM=8
# HOST_CONFIG="g4005:8"
# HOST_CONFIG="g3021:8"

export MASTER_ADDR=$(echo $HOST_CONFIG | cut -d',' -f1 | cut -d':' -f1)
export MASTER_ADDR=$(echo $HOST_CONFIG | awk -F',' '{print $1}' | awk -F':' '{print $1}')
echo "MASTER_ADDR: $MASTER_ADDR"
set -x
mpirun --prefix $(dirname `which mpirun`)/../ \
   -x MASTER_ADDR -x MASTER_PORT\
   -x PATH -x LD_LIBRARY_PATH \
   -x NCCL_DEBUG \
   -x NCCL_NET_GDR_LEVEL \
   -x NCCL_DEBUG_SUBSYS \
   -x NCCL_IB_DISABLE \
   -np $GPU_NUM --host $HOST_CONFIG \
   --map-by ppr:4:numa --bind-to core --report-bindings \
python $EXECUBLE \
    --output $OUTPUT \


set +x

done