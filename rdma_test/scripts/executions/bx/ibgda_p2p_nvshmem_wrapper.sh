#!/usr/bin/env bash
set -euo pipefail

# Build
make ibgda_p2p_nvshmem_clean
make ibgda_p2p_nvshmem

# Envs
GPUS_PER_NODE=2
TASKS_PER_NODE=2

# 最大消息大小（同时也作为 NVSHMEM 对称堆大小）
MIN_SIZE="256"
MIN_SIZE="16M"
MIN_SIZE="64M"
MAX_SIZE="${1:-64M}"
# MAX_SIZE="${1:-16M}"
export MIN_SIZE
export MAX_SIZE
export ITERS="${2:-200}"
export WARMUP="${3:-50}"

export NVSHMEM_SYMMETRIC_SIZE="${MAX_SIZE}"
# —— 强制 IBGDA + 仅走 IB —— #
export NVSHMEM_IBGDA=1
export NVSHMEM_USE_GPUDIRECT_ASYNC=1
export UCX_TLS=rc,self,cuda_copy  # 禁止 shm/tcp/cuda_ipc
export UCX_IB_GPU_DIRECT_RDMA=y
export UCX_MEMTYPE_CACHE=y
export UCX_IB_PCI_RELAXED_ORDERING=y      # 放开 PCIe 顺序（常见平台安全且提速）

# —— 大消息 RNDV 的分片与并发 —— #
# 强制使用 RDMA 写的 zcopy（对 put 更友好；也试试 get_zcopy，二选一对比）
export UCX_RNDV_SCHEME=put_zcopy           # 备选：get_zcopy / auto

# 分片大小（默认往往较小），拉大避免频繁握手；区间 1m~8m 试一下
export UCX_RNDV_FRAG_SIZE=4m               # 可测 2m/4m/8m

# 发送队列与流控窗口：把在途 WQE/credits 提上去（默认多为 256）
export UCX_RC_TX_QUEUE_LEN=4096
export UCX_RC_FC_WND=4096

# Debug Switch
# export NVSHMEM_INFO=3
export UCX_LOG_LEVEL=debug

export CUDA_VISIBLE_DEVICES=0,1
#
# MPMD：两段 app-context，分别绑定 GPU/NIC
set -x

# srun --mpi=pmi2 -N 1 --ntasks-per-node=2 --gres=gpu:2 \
#      bash -lc '
#        set -euo pipefail
#        if [[ "${SLURM_PROCID}" == "0" ]]; then
#          export UCX_NET_DEVICES=mlx5_0:1
#        else
#          export UCX_NET_DEVICES=mlx5_1:1
#        fi
#        echo "[PE ${SLURM_PROCID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} UCX_NET_DEVICES=${UCX_NET_DEVICES}"
#        ./build/ibgda_p2p_nvshmem --min $MIN_SIZE --max "${MAX_SIZE}" --factor 2 --iters "${ITERS}" --warmup "${WARMUP}"
#      '

NSYS_DIR=logs/bx/nsys
TRACE_NAME=ibgda_p2p
mkdir -p $NSYS_DIR

export NSIGHT_CMD="nsys profile \
  --mpi-impl=mpich \
  --trace=cuda,osrt,nvtx \
  --gpu-metrics-devices=all \
  --gpu-metrics-set=tu10x-gfxt \      
  --gpu-metrics-frequency=1000 \
  --ib-net-info=mlx5_0,mlx5_1 \ 
  --output=${NSYS_DIR}/${TRACE_NAME}_$(date "+%Y%m%d-%H%M%S")"
# NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${TRACE_NAME}_w${GPUS_PER_NODE}_$(date "+%Y%m%d-%H%M%S")"

$NSIGHT_CMD \
mpirun \
  -np 1 \
  -env UCX_NET_DEVICES=mlx5_0:1 \
  ./build/ibgda_p2p_nvshmem --min $MIN_SIZE --max "${MAX_SIZE}" --factor 2 --iters "${ITERS}" --warmup "${WARMUP}" \
  : \
  -np 1 \
  -env UCX_NET_DEVICES=mlx5_1:1 \
  ./build/ibgda_p2p_nvshmem --min $MIN_SIZE --max "${MAX_SIZE}" --factor 2 --iters "${ITERS}" --warmup "${WARMUP}"

set +x
exit 0

# ---- 用 srun 启动两个任务（同一节点）----
# task 0: GPU0 + mlx5_3:1
# task 1: GPU1 + mlx5_4:1
# mpirun -n $TASKS_PER_NODE \
#      --gpu-bind=map_gpu:0,1 \
srun --mpi=pmi2 -N 1 --ntasks-per-node=2 --gres=gpu:2 \
     bash -lc '
       set -euo pipefail
       if [[ "${SLURM_PROCID}" == "0" ]]; then
         export CUDA_VISIBLE_DEVICES=0
         export UCX_NET_DEVICES=mlx5_0:1
       else
         export CUDA_VISIBLE_DEVICES=1
         export UCX_NET_DEVICES=mlx5_1:1
       fi
       echo "[PE ${SLURM_PROCID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} UCX_NET_DEVICES=${UCX_NET_DEVICES}"
       ./build/ibgda_p2p_nvshmem '"$SIZE"'
     '

exit 0

#!/bin/bash

HOST="g0274"
HOST="g0288"
HOST="g0297"
HOST="g0278"
GPUS_PER_NODE=2
NSYS_DIR=logs/bx/nsys
TRACE_NAME="ibgda_p2p_nvshmem"
mkdir -p $NSYS_DIR

# # build
# make ibgda_p2p_nvshmem_clean
# make ibgda_p2p_nvshmem

NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${TRACE_NAME}_w${GPUS_PER_NODE}_$(date "+%Y%m%d-%H%M%S")"
NSIGHT_CMD=""

# run
srun --mpi=pmi2 -N 1 -n 2 --ntasks-per-node=2 --gpus-per-task=1 \
     --gpu-bind=map_gpu:0,1 --gres=gpu:$GPUS_PER_NODE \
${NSIGHT_CMD} \
./scripts/executions/bx/ibgda_p2p_nvshmem_executor.sh

# -w $HOST 
