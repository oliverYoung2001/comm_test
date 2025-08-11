#!/bin/bash

HOST="g0274"
HOST="g0288"
HOST="g0297"
HOST="g0278"

# build
make rdma_p2p_gpt5_clean
make rdma_p2p_gpt5

# run
srun --gres=gpu:2 ./scripts/executions/bx/rdma_p2p_gpt5_executor.sh \
    2>&1 | tee ./logs/bx/rdma_p2p_gpt5.log

# -w $HOST 
