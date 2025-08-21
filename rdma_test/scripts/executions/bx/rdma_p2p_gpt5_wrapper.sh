#!/bin/bash

HOST="g0274"
HOST="g0288"
HOST="g0297"
HOST="g0278"
GPUS_PER_NODE=2
NSYS_DIR=logs/bx/nsys
TRACE_NAME="rdma_p2p_gpt5"
mkdir -p $NSYS_DIR

# build
make rdma_p2p_gpt5_clean
make rdma_p2p_gpt5

NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${TRACE_NAME}_w${GPUS_PER_NODE}_$(date "+%Y%m%d-%H%M%S")"

# run
srun --gres=gpu:$GPUS_PER_NODE \
${NSIGHT_CMD} \
./scripts/executions/bx/rdma_p2p_gpt5_executor.sh \
    2>&1 | tee ./logs/bx/rdma_p2p_gpt5.log

# -w $HOST 
