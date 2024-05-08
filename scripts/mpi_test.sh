#!/bin/bash

# export OPENMPI_HOME=/home/zhaijidong/yhy/.local/openmpi
# export PATH="$OPENMPI_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"

# hostname

# echo "$OMPI_COMM_WORLD_RANK, $OMPI_COMM_WORLD_LOCAL_RANK, $OMPI_COMM_WORLD_LOCAL_SIZE, `hostname`, $OMPI_COMM_WORLD_HOSTNAME"

echo "$OMPI_COMM_WORLD_CLUSTER_NAME, $OMPI_COMM_WORLD_NODEID, $OMPI_COMM_WORLD_NODENAME, $CUDA_HOME, $LD_LIBRARY_PATH, `hostname`"

# ./csrc/build/conflict_allinone 2 NCCL ./scripts/configs/small.json
