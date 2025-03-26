#!/bin/bash

# export RECORD_P2P=1
EXECUBLE=conflict_bench.py

GPU_NUMs="8"
GPU_NUMs="16"
GPU_NUMs="32"
BACKENDs="NCCL MPI GLOO UCC"
BACKENDs="GLOO"
BACKENDs="UCC"
# BACKENDs="MPI"
BACKENDs="NCCL"
CP_FILE_NAMEs="p2p_si p2p_bi"
CP_FILE_NAMEs="p2p_bi"
CP_FILE_NAMEs="conflict_patterns"
# CP_FILE_NAMEs="bad_patterns_3+2"
# CP_FILE_NAMEs="bad_patterns_3+3"
# CP_FILE_NAMEs="bad_patterns_pcie_switch"
# CP_FILE_NAMEs="all2all_4"
# CP_FILE_NAMEs="E2E_4 E2E_8"
CP_FILE_NAMEs="ring"
CP_FILE_NAMEs="small"

# nico:
# PARTITION=SXM
# HOSTs="zoltan"
# PARTITION=V100
# HOSTs="nico3,nico4"
PARTITION=Mix
# HOSTs="nico1,nico2"
HOSTs="nico2"

# qy:
PARTITION=align
PARTITION=rag
GPU_NUMs="8"
HOSTs="g3017"
# GPU_NUMs="16"
# # HOSTs="g4007,g4008"
# # HOSTs="g4002,g4003"
# # HOSTs="g3025,g3026"
# HOSTs="g3027,g3028"
# HOSTs="g1003,g1004"
# HOSTs="g3017,g3018"
# HOSTs="g3015,g3017"
# HOSTs="g3015,g3018"
# GPU_NUMs="24"
# HOSTs="g3015,g3018,g3021"
# GPU_NUMs="32"
# HOSTs="g3011,g3017,g3018,g3022"
# PARTITION=arch
# GPU_NUMs="8"
# HOSTs="g3028"
# GPU_NUMs="16"
# HOSTs="g3027,g3028"
# HOSTs="g3028,g3029"
# GPU_NUMs="24"
# HOSTs="g3024,g3028,g3029"

# GPU_NUMs="32"
# HOSTs="g3024,g3025,g3026,g3027"
# HOSTs="g3025,g3026,g3027,g3028"
# HOSTs="g3027,g3028,g3029,g3030"
# GPU_NUMs="48"
# HOSTs="g3025,g3026,g3027,g3028,g3029,g3030"
# HOSTs="g3023,g3024,g3025,g3026,g3027,g3028"

# PARTITION=hit
# GPU_NUMs="8"
# HOSTs="g4002"
# GPU_NUMs="16"
# HOSTs="g4004,g4005"
# HOSTs="g4001,g4004"
# HOSTs="g4001,g4005"
# GPU_NUMs="24"
# HOSTs="g4001,g4004,g4005"

CPU_PER_TASK=16

# PARTITION=align
# GPU_NUMs="16"
# HOSTs="g1003,g1004"
# # GPU_NUMs="8"
# # HOSTs="g1003"
# CPU_PER_TASK=14

# # wq:
# PARTITION=Nvidia_A800
# # PARTITION=gpu3-2-low
# # HOSTs="g4004"
# HOSTs="gpu21"
# HOSTs="gpu21,gpu22"
# HOSTs="gpu[11-14]"

# Fit
CLUSTER_NAME=fit
PARTITION=a01
GPU_NUMs="8"
HOSTs="g01"
GPU_NUMs="16"
HOSTs="g01,g07"
# GPU_NUMs="24"
# HOSTs="g[01-02],g05"
# GPU_NUMs="32"
# HOSTs="g[13-16]"

PARTITION=h01
GPU_NUMs="8"
HOSTs="g40"
HOSTs="g42"
GPU_NUMs="16"
HOSTs="g40,g42"
HOSTs="g42,g44"
# GPU_NUMs="24"
# HOSTs="g40,g42,g44"

CPU_PER_TASK=13
# HOSTs="None"

export MASTER_PORT=$((RANDOM % 12000 + 10000))


# mkdir results
mkdir -p results_${CLUSTER_NAME}


for BACKEND in $BACKENDs; do
for HOST in $HOSTs; do
echo "HOST: $HOST"
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

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,2
echo "HOST: $HOST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU_NUM: $GPU_NUM"
echo "PROC_NUM: $PROC_NUM"
echo "NNODES: $NNODES"
echo "NTASK_PER_NODE: $NTASK_PER_NODE"
# ncu nvprof \


# SLURM ARGS:
# MEM_PER_CPU=256G
MEM_PER_NODE=256G
# --mem-per-cpu $MEM_PER_CPU \
# --mem $MEM_PER_NODE \

SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node $NTASK_PER_NODE \
--gres=gpu:$NTASK_PER_NODE \
--mem $MEM_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

export TRACE_NAME=${CLUSTER_NAME}_${CP_FILE_NAME}
TB_DIR=./tb
mkdir -p $TB_DIR
LOGGING_ARGS=""

# LOGGING_ARGS="${LOGGING_ARGS} \
# --profiler-with-tensorboard \
# --tb-dir $TB_DIR \
# "

# NCCL Args:
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=ERROR
export NCCL_NET_GDR_LEVEL=5
# export NCCL_NET_GDR_LEVEL=0   # Disable GDR
export NCCL_IB_DISABLE=0
export NCCL_DEBUG_SUBSYS=NET

# export NCCL_ALGO=Ring
# export NCCL_PROTO=Simple

# export NCCL_AVOID_RECORD_STREAMS=0  # Disable Warning for P2P
# export CUDA_LAUNCH_BLOCKING=1


# Comm Module
# COMM_MODULE='torch-distributed'
# COMM_MODULE='raw-nccl'
COMM_MODULEs="raw-nccl torch-distributed"
COMM_MODULEs="raw-nccl"


for COMM_MODULE in $COMM_MODULEs; do
echo "COMM_MODULE: ${COMM_MODULE}"

# Launch with Slurm
# -c 14 \   # WQ
# -c 16 \   # QY
# -c 13 \   # Fit
export SLURM_CPU_BIND=verbose
set -x
srun $SLURM_ARGS \
-c $CPU_PER_TASK \
./scripts/executor.sh \
python $EXECUBLE \
    --backend $BACKEND \
    --config ./scripts/configs/${CP_FILE_NAME}_${GPU_NUM}.json \
    --comm-module $COMM_MODULE \
    $LOGGING_ARGS \
    2>/dev/null # Disable Warning
set +x
# exit 0

# # Launch with MPI
# # salloc -p rag -w g3017 -N 1 -n 128 -t 3600
# # salloc -p rag -w g3017,g3018 -N 2 -n 256 -t 3600
# # salloc -p rag -w g3011,g3017,g3018,g3022 -N 4 -n 512 -t 3600
# GPU_NUM=32
# HOST_CONFIG="g4005:8,g4006:8,g4007:8,g4008:8"
# HOST_CONFIG="g3021:8,g3022:8,g3023:8,g3024:8"
# HOST_CONFIG="g3011:8,g3017:8,g3018:8,g3022:8"
# # GPU_NUM=16
# # # HOST_CONFIG="g4007:8,g4008:8"
# # HOST_CONFIG="g3017:8,g3018:8"
# # # HOST_CONFIG="g4003:8,g4006:8"
# # HOST_CONFIG="g3021:8,g3022:8"
# # GPU_NUM=8
# # HOST_CONFIG="g4005:8"
# GPU_NUM=2
# HOST_CONFIG="g3017:2"
# ALLOW_ROOT=""
# # # Tusimple
# # GPU_NUM=8
# # HOST_CONFIG="10.21.74.94:8"     # Node0
# # HOST_CONFIG="10.21.77.110:8"    # Node1
# # HOST_CONFIG="feng-wang-qcsleep2-worker-0:8"    # Node1
# # GPU_NUM=16
# # HOST_CONFIG="feng-wang-qcsleep2-worker-0:8,feng-wang-qcsleep2-worker-1:8"     # Node0, 1
# # # HOST_CONFIG="10.21.74.94:8,10.21.77.110:8"


# ALLOW_ROOT="--allow-run-as-root"

# export MASTER_ADDR=$(echo $HOST_CONFIG | cut -d',' -f1 | cut -d':' -f1)
# export MASTER_ADDR=$(echo $HOST_CONFIG | awk -F',' '{print $1}' | awk -F':' '{print $1}')
# # export MASTER_ADDR="10.21.77.110"
# # export MASTER_ADDR="feng-wang-qcsleep2-worker-1"
# echo "MASTER_ADDR: $MASTER_ADDR"
# echo "MASTER_PORT: $MASTER_PORT"
# RUNNER_CMD="mpirun $ALLOW_ROOT --prefix $(dirname `which mpirun`)/../ \
#     -x MASTER_ADDR -x MASTER_PORT \
#     -x LD_LIBRARY_PATH -x PATH \
#     -x TRACE_NAME \
#     -x NCCL_DEBUG \
#     -x NCCL_NET_GDR_LEVEL \
#     -x NCCL_DEBUG_SUBSYS \
#     -x NCCL_IB_DISABLE \
#     --map-by ppr:4:numa --bind-to core --report-bindings \
#     -np $GPU_NUM --host $HOST_CONFIG"
# NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${TRACE_NAME}_w${GPU_NUM}_$(date "+%Y%m%d-%H%M%S")"
# NSIGHT_CMD=""
# set -x
# ${NSIGHT_CMD} \
# ${RUNNER_CMD} \
# python $EXECUBLE \
#     --backend $BACKEND \
#     --config ./scripts/configs/${CP_FILE_NAME}_${GPU_NUM}.json \
#     --comm-module $COMM_MODULE \
#     # 2>/dev/null # Disable Warning

done
done
done
done
done
