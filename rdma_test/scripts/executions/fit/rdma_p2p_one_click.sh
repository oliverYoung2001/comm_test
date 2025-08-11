#!/bin/bash

# build
make rdma_client_clean
make rdma_server_clean
make rdma_client
make rdma_server

# run
srun -p h01 -w g42 --gres=gpu:2 ./scripts/rdma_p2p_executor.sh
