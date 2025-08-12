#!/bin/bash

# nccl-tests
/data/apps/openmpi/bin/mpirun --allow-run-as-root -H g0291:8,g0292:8 --mca pml ucx \
       -x LD_LIBRARY_PATH -x PATH -x NCCL_DEBUG=info -x NCCL_NVLS_ENABLE=0  -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_4:1 \
       -x UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_4:1 \
       -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_DISABLE=0 \
       -x NCCL_ALGO=Ring /data/apps/nccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1 2>&1 | tee $log_file
