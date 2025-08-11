#!/bin/bash

HOST="g0274"
HOST="g0288"
HOST="g0297"

# build
make rdma_p2p_gpt_clean
make rdma_p2p_gpt

# run
srun -w $HOST --gres=gpu:2 ./scripts/executions/bx/rdma_p2p_gpt_executor.sh
