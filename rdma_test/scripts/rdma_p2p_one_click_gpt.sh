#!/bin/bash

HOST="g42"
# HOST="g40"

# build
make rdma_p2p_gpt_clean
make rdma_p2p_gpt

# run
srun -p h01 -w $HOST --gres=gpu:2 ./scripts/rdma_p2p_executor_gpt.sh
