#!/bin/bash

HOST="g0274"

# build
make rdma_p2p_claude_clean
make rdma_p2p_claude

# run
srun -w $HOST --gres=gpu:2 ./scripts/executions/bx/rdma_p2p_claude_executor.sh
