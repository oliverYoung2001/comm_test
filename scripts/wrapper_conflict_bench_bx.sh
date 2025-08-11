#!/bin/bash

# export RECORD_P2P=1
EXECUBLE=conflict_bench.py

BACKENDs="NCCL MPI GLOO UCC"
BACKENDs="GLOO"
BACKENDs="UCC"
# BACKENDs="MPI"
BACKENDs="NCCL"
CP_FILE_NAMEs="small"

# bx
CLUSTER_NAME=bx
GPU_NUMs="8"
# GPU_NUMs="16"
# GPU_NUMs="24"
# GPU_NUMs="32"
# GPU_NUMs="48"
# GPU_NUMs="64"
CP_FILE_NAMEs="small"

export MASTER_PORT=$((RANDOM % 12000 + 10000))

for BACKEND in $BACKENDs; do
for GPU_NUM in $GPU_NUMs; do       # for cudaMemcpy
for CP_FILE_NAME in $CP_FILE_NAMEs; do
CP_FILE_NAME=${CP_FILE_NAME}_${GPU_NUM}
echo "CP_FILE_NAME: ${CP_FILE_NAME}"

PROC_NUM=$GPU_NUM
if [ $GPU_NUM -le 8 ]; then
   NNODES=1
else
   NNODES=$(($GPU_NUM / 8))
fi
NTASK_PER_NODE=`expr $PROC_NUM / $NNODES`
export MLP_WORKER_NUM=$NNODES

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,2
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU_NUM: $GPU_NUM"
echo "PROC_NUM: $PROC_NUM"
echo "NNODES: $NNODES"
echo "NTASK_PER_NODE: $NTASK_PER_NODE"

# Comm Module
COMM_MODULEs="raw-nccl torch-distributed"
# COMM_MODULEs="raw-nccl"


for COMM_MODULE in $COMM_MODULEs; do
echo "COMM_MODULE: ${COMM_MODULE}"

source ./scripts/task_configs/cb_${GPU_NUM}.sh

# [TODO]: Check NUMA affinity

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
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi
if [ ! -z "$CPU_PER_TASK" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        --cpus-per-task=$CPU_PER_TASK \
    "
fi

# # Run with Slurm
# export SLURM_CPU_BIND=verbose
# export SLURM_CPU_BIND_VERBOSE=1
RUNNER_CMD="srun $SLURM_ARGS"

set -x
$RUNNER_CMD \
$EXECUTABLE
set +x

done
done
done
done