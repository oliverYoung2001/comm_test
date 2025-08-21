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
# MIN_SIZE="64M"
MAX_SIZE="${1:-64M}"
# MAX_SIZE="${1:-16M}"
ITERS="${2:-200}"
WARMUP="${3:-50}"

export NVSHMEM_SYMMETRIC_SIZE="${MAX_SIZE}"
# —— 只走 NVLink / CUDA-IPC —— #
# 关闭 IBGDA / RDMA / RC
unset NVSHMEM_IBGDA
unset NVSHMEM_USE_GPUDIRECT_ASYNC
unset UCX_NET_DEVICES
unset UCX_IB_GPU_DIRECT_RDMA
unset UCX_RNDV_SCHEME
unset UCX_RNDV_FRAG_SIZE
unset UCX_RC_TX_QUEUE_LEN
unset UCX_RC_FC_WND

# 只允许 CUDA IPC + 必要组件
export UCX_TLS=cuda_ipc,cuda_copy,sm,self

set -x
mpirun \
  -np 1 \
  -x CUDA_VISIBLE_DEVICES=0 \
  ./build/ibgda_p2p_nvshmem --min $MIN_SIZE --max "${MAX_SIZE}" --factor 2 --iters "${ITERS}" --warmup "${WARMUP}" \
  : \
  -np 1 \
  -x CUDA_VISIBLE_DEVICES=1 \
  ./build/ibgda_p2p_nvshmem --min $MIN_SIZE --max "${MAX_SIZE}" --factor 2 --iters "${ITERS}" --warmup "${WARMUP}"

set +x
