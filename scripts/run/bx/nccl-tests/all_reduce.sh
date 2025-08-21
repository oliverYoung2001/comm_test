#!/bin/bash

PARTITION="H100"
NNODES=2
NPROC_PER_NODE=8
GPUS_PER_NODE=8
# CPUS=176
# CPU_PER_TASK=$((CPUS / NPROC_PER_NODE ))   # [NOTE]: Unnecessary for performance.
HOST=""
# HOST="g[0290,0291]"

SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node=$NPROC_PER_NODE \
--gres=gpu:$GPUS_PER_NODE \
-K \
--cpu-bind=none \
"
# --exclusive \
# --cpu-bind=none \
if [ ! -z "$HOST" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi
if [ ! -z "$CPU_PER_TASK" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        --cpus-per-task=$CPU_PER_TASK \
    "
fi

# NCCL Args
export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=bond0
export NCCL_NVLS_ENABLE=0
# export NCCL_DEBUG=INFO

RUNNER_CMD="srun $SLURM_ARGS"

EXECUTABLE="./scripts/executor.sh \
    ./third_party/nccl-tests/build/all_reduce_perf -b 8M -e 1G -f 2 -g 1
"

set -x
$RUNNER_CMD \
$EXECUTABLE
set +x

# # nccl-tests
# /data/apps/openmpi/bin/mpirun --allow-run-as-root -H g0291:8,g0292:8 --mca pml ucx \
#        -x LD_LIBRARY_PATH -x PATH -x NCCL_DEBUG=info -x NCCL_NVLS_ENABLE=0  -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_4:1 \
#        -x UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_4:1 \
#        -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_DISABLE=0 \
#        -x NCCL_ALGO=Ring /data/apps/nccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1 2>&1 | tee $log_file
