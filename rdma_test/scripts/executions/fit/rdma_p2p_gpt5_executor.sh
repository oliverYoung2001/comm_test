#!/bin/bash

# 典型参数（按你的设备名改成 mlx5_3 / mlx5_4）
SRC_PORT=1
DST_PORT=1
SRC_DEV="mlx5_3"
DST_DEV="mlx5_4"

DST_DEV="mlx5_3"
SRC_DEV="mlx5_0"
# DST_DEV="mlx5_0"
DST_DEV="mlx5_1"

SRC_GPU=0
DST_GPU=1
# DST_GPU=0
SIZE="64M"

./build/rdma_p2p_gpt5 \
  --src-dev "${SRC_DEV}" --dst-dev "${DST_DEV}" \
  --src-port ${SRC_PORT} --dst-port ${DST_PORT} \
  --src-gpu ${SRC_GPU} --dst-gpu ${DST_GPU} \
  --size "${SIZE}"

# SRC_DEV="mlx5_3"
# DST_DEV="mlx5_3"
# # SRC_PORT=1
# # DST_PORT=1
# # SRC_GID_IDX=0
# # DST_GID_IDX=0
# SRC_GPU=0
# DST_GPU=1
# SIZE="64M"

# ./build/rdma_p2p_gpt5 \
#   --src-dev "${SRC_DEV}" --dst-dev "${DST_DEV}" \
#   --src-gpu ${SRC_GPU} --dst-gpu ${DST_GPU} \
#   --size "${SIZE}"

  # --src-port ${SRC_PORT} --dst-port ${DST_PORT} \
  # --src-gid-idx ${SRC_GID_IDX} --dst-gid-idx ${DST_GID_IDX} \
