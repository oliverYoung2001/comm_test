#!/bin/bash

PARTITION=h01
HOST="g46"

# build
make rdma_p2p_gpt5_clean
make rdma_p2p_gpt5

# run
srun -p $PARTITION -w $HOST --gres=gpu:2 ./scripts/executions/bx/rdma_p2p_gpt5_executor.sh \
    2>&1 | tee ./logs/fit/rdma_p2p_gpt5.log

# -w $HOST 
