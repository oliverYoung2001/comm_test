#!/usr/bin/env bash
set -euo pipefail

# Build
make mpi_test_clean
make mpi_test

# Envs
GPUS_PER_NODE=2
TASKS_PER_NODE=2


# ---- IBGDA / UCX / NVSHMEM 环境建议 ----
# 强制走 IBGDA（GPU 端发起）
export NVSHMEM_IBGDA=1
export NVSHMEM_USE_GPUDIRECT_ASYNC=1

# --mpi=pmix 
srun --mpi=pmi2 -N 1 --ntasks-per-node=$TASKS_PER_NODE --gpus-per-node=$GPUS_PER_NODE \
    ./build/mpi_test
# mpirun -n $TASKS_PER_NODE \
