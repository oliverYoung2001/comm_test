#!/bin/bash

HOST="g0274"

# build
make rdma_p2p_gpt_clean
make rdma_p2p_gpt

# run
srun -w $HOST --gres=gpu:2 ./scripts/executions/bx/ib_write_bw_executor.sh
