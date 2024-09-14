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


# NCCL Args:
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=ERROR
export NCCL_NET_GDR_LEVEL=5
# export NCCL_NET_GDR_LEVEL=0   # Disable GDR
export NCCL_IB_DISABLE=0
export NCCL_DEBUG_SUBSYS=NET
# Only for Tusimple, [NOTE]: necessary !!!
export NCCL_SOCKET_IFNAME="=eth1,=eth2,=eth3,=eth4"
export NCCL_IB_GID_INDEX="7"

# export NCCL_AVOID_RECORD_STREAMS=0  # Disable Warning for P2P
# export CUDA_LAUNCH_BLOCKING=1

# # Launch with Slurm
# set -x
# srun $SLURM_ARGS \
# ./scripts/executor.sh \
# python $EXECUBLE \
#     --output $OUTPUT \

# Launch with MPI
# Tusimple
GPU_NUM=8
HOST_CONFIG="10.21.74.94:8"     # Node0
HOST_CONFIG="10.21.77.110:8"    # Node1
HOST_CONFIG="feng-wang-qcsleep2-worker-0:8"    # Node1
GPU_NUM=16
HOST_CONFIG="feng-wang-qcsleep2-worker-0:8,feng-wang-qcsleep2-worker-1:8"     # Node0, 1
# GPU_NUM=24
# HOST_CONFIG="feng-wang-qcsleep2-worker-0:8,feng-wang-qcsleep2-worker-1:8,feng-wang-qcsleep2-worker-2:8"     # Node0, 1, 2
# GPU_NUM=32
# HOST_CONFIG="feng-wang-qcsleep2-worker-0:8,feng-wang-qcsleep2-worker-1:8,feng-wang-qcsleep2-worker-2:8,feng-wang-qcsleep2-worker-3:8"     # Node0, 1, 2, 3

ALLOW_ROOT="--allow-run-as-root"

export MASTER_ADDR=$(echo $HOST_CONFIG | cut -d',' -f1 | cut -d':' -f1)
export MASTER_ADDR=$(echo $HOST_CONFIG | awk -F',' '{print $1}' | awk -F':' '{print $1}')

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
RUNNER_CMD="mpirun $ALLOW_ROOT --prefix $(dirname `which mpirun`)/../ \
    -x MASTER_ADDR -x MASTER_PORT \
    -x LD_LIBRARY_PATH -x PATH \
    -x TRACE_NAME \
    -x NCCL_DEBUG \
    -x NCCL_NET_GDR_LEVEL \
    -x NCCL_DEBUG_SUBSYS \
    -x NCCL_IB_DISABLE \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_GID_INDEX \
    --map-by ppr:4:numa --bind-to core --report-bindings \
    -np $GPU_NUM --host $HOST_CONFIG"
NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${TRACE_NAME}_w${GPU_NUM}_$(date "+%Y%m%d-%H%M%S")"
NSIGHT_CMD=""
set -x
${NSIGHT_CMD} \
${RUNNER_CMD} \
python $EXECUBLE \
    --output $OUTPUT \

set +x
done