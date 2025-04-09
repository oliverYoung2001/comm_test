#!/bin/bash

# export RECORD_P2P=1
EXECUBLE=conflict_bench.py

BACKENDs="NCCL MPI GLOO UCC"
BACKENDs="GLOO"
BACKENDs="UCC"
# BACKENDs="MPI"
BACKENDs="NCCL"
CP_FILE_NAMEs="small"

# # Fit
# CLUSTER_NAME=fit
# PARTITION=a01
# GPU_NUMs="8"
# HOSTs="g01"
# GPU_NUMs="16"
# HOSTs="g01,g07"
# # GPU_NUMs="24"
# # HOSTs="g[01-02],g05"
# # GPU_NUMs="32"
# # HOSTs="g[13-16]"

# PARTITION=h01
# GPU_NUMs="8"
# HOSTs="g40"
# HOSTs="g42"
# GPU_NUMs="16"
# HOSTs="g40,g42"
# HOSTs="g42,g44"
# # GPU_NUMs="24"
# # HOSTs="g40,g42,g44"

# CPU_PER_TASK=13
# # HOSTs="None"

# Zhipu
CLUSTER_NAME=zhipu_planck
CLUSTER_NAME=zhipu_hamming
GPU_NUMs="8"
GPU_NUMs="16"
# GPU_NUMs="24"
# GPU_NUMs="32"
# GPU_NUMs="48"
# GPU_NUMs="64"
CP_FILE_NAMEs="small"

#   Envs for Zhipu
NET_DEVICE=$MLP_SOCKET_IFNAME
MLP_MPI_HOSTFILE=/root/mpi_rack_hostfile
# End


export MASTER_PORT=$((RANDOM % 12000 + 10000))


# mkdir results
mkdir -p results_${CLUSTER_NAME}
MPI_EXTRA=''
# if [ $CLUSTER_NAME == 'zhipu_planck' ] || [ $CLUSTER_NAME == 'zhipu_hamming' ]; then
#    MPI_EXTRA="$MPI_EXTRA \
#    -mca oob_tcp_if_include 10.102.2.0/24 \
#    "
# fi

for BACKEND in $BACKENDs; do
for GPU_NUM in $GPU_NUMs; do       # for cudaMemcpy
for CP_FILE_NAME in $CP_FILE_NAMEs; do
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
# ncu nvprof \

export TRACE_NAME=${CLUSTER_NAME}_${CP_FILE_NAME}
TB_DIR=./tb
mkdir -p $TB_DIR
LOGGING_ARGS=""

# LOGGING_ARGS="${LOGGING_ARGS} \
# --profiler-with-tensorboard \
# --tb-dir $TB_DIR \
# "

# # NCCL Args:
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG=WARN
# # export NCCL_DEBUG=ERROR
# export NCCL_NET_GDR_LEVEL=5
# # export NCCL_NET_GDR_LEVEL=0   # Disable GDR
# export NCCL_IB_DISABLE=0
# export NCCL_DEBUG_SUBSYS=NET

# export NCCL_ALGO=Ring
# export NCCL_PROTO=Simple

# export NCCL_AVOID_RECORD_STREAMS=0  # Disable Warning for P2P
# export CUDA_LAUNCH_BLOCKING=1


# Comm Module
COMM_MODULEs="raw-nccl torch-distributed"
COMM_MODULEs="raw-nccl"


for COMM_MODULE in $COMM_MODULEs; do
echo "COMM_MODULE: ${COMM_MODULE}"

set -x
mpirun -np $((MLP_WORKER_NUM * MLP_GPU)) \
        --hostfile ${MLP_MPI_HOSTFILE} \
        --allow-run-as-root -oversubscribe -map-by ppr:8:node \
        --bind-to numa \
        -mca pml ob1 -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include=${NET_DEVICE} \
        $MPI_EXTRA \
        --output-filename results/${TIMESTAMP} \
        -x NCCL_PXN_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_NET_GDR_LEVEL=4 \
        -x NCCL_IB_RETRY_CNT=7 \
        -x NCCL_IB_TIME_OUT=32 \
        -x NCCL_IB_QPS_PER_CONNECTION=8 \
        -x NCCL_P2P_LEVEL=NVL \
        -x NCCL_DEBUG=VERSION \
        -x PATH \
        -x MASTER_ADDR=$(cat $MLP_MPI_HOSTFILE | head -n 1 | sed -s 's/slots=8//g') \
        -x MASTER_PORT=${MLP_WORKER_0_PORT} \
        -x GLOO_SOCKET_IFNAME=${NET_DEVICE} \
        -x NCCL_SOCKET_IFNAME=${NET_DEVICE} \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        -x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -x NCCL_NVLS_ENABLE=0 \
        -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
        -x CLUSTER_NAME \
        -x PLATFORM \
        -x TRACE_NAME \
        ./scripts/executor.sh \
        python $EXECUBLE \
            --backend $BACKEND \
            --config ./scripts/configs/${CP_FILE_NAME}_${GPU_NUM}.json \
            --comm-module $COMM_MODULE \
            $LOGGING_ARGS \
            2>/dev/null # Disable Warning

set +x
done
done
done
done