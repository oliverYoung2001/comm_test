#!/usr/bin/env bash
set -euo pipefail

# Build
make shift_mpi_clean
make shift_mpi

# Envs
GPUS_PER_NODE=2
TASKS_PER_NODE=2


# ---- IBGDA / UCX / NVSHMEM 环境建议 ----
# 强制走 IBGDA（GPU 端发起）
export NVSHMEM_IBGDA=1
export NVSHMEM_USE_GPUDIRECT_ASYNC=1

# --mpi=pmix 
srun --mpi=pmi2 -N 1 --ntasks-per-node=$TASKS_PER_NODE --gpus-per-node=$GPUS_PER_NODE \
    ./build/shift_mpi
# mpirun -n $TASKS_PER_NODE \
