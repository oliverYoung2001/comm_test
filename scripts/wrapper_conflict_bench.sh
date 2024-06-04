#!/bin/bash

# export RECORD_P2P=1
EXECUBLE=conflict_bench.py

GPU_NUMs="8"
GPU_NUMs="16"
GPU_NUMs="32"
BACKENDs="NCCL MPI"
BACKENDs="MPI"
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
PARTITION=gpu4-low
# PARTITION=gpu3-2-low
# HOSTs="g4004"
GPU_NUMs="16"
HOSTs="g4007,g4008"
HOSTs="g4002,g4003"
GPU_NUMs="8"
HOSTs="g4005"

# wq:
PARTITION=Nvidia_A800
# PARTITION=gpu3-2-low
# HOSTs="g4004"
HOSTs="gpu21"
HOSTs="gpu21,gpu22"
HOSTs="gpu[11-14]"


# HOSTs="None"

export MASTER_PORT=$((RANDOM % 12000 + 10000))


# mkdir results
mkdir -p results


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

# NCCL Args:
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=ERROR
export NCCL_NET_GDR_LEVEL=5
# export NCCL_NET_GDR_LEVEL=0   # Disable GDR
export NCCL_IB_DISABLE=0
export NCCL_DEBUG_SUBSYS=NET

# export NCCL_AVOID_RECORD_STREAMS=0  # Disable Warning for P2P
# export CUDA_LAUNCH_BLOCKING=1

# # Launch with Slurm
# set -x
# srun $SLURM_ARGS \
# ./scripts/executor.sh \
# python $EXECUBLE \
#     --backend $BACKEND \
#     --config ./scripts/configs/${CP_FILE_NAME}_${GPU_NUM}.json \

# Launch with MPI
GPU_NUM=32
HOST_CONFIG="g4005:8,g4006:8,g4007:8,g4008:8"
HOST_CONFIG="g3021:8,g3022:8,g3023:8,g3024:8"
# GPU_NUM=16
# HOST_CONFIG="g4007:8,g4008:8"
# # HOST_CONFIG="g3025:8,g4006:8"
# # HOST_CONFIG="g4003:8,g4006:8"
# HOST_CONFIG="g3021:8,g3022:8"
# GPU_NUM=8
# HOST_CONFIG="g4005:8"
export MASTER_ADDR=$(echo $HOST_CONFIG | cut -d',' -f1 | cut -d':' -f1)
export MASTER_ADDR=$(echo $HOST_CONFIG | awk -F',' '{print $1}' | awk -F':' '{print $1}')
echo "MASTER_ADDR: $MASTER_ADDR"
set -x
mpirun --prefix $(dirname `which mpirun`)/../ \
   -x MASTER_ADDR -x MASTER_PORT\
   -x PATH -x LD_LIBRARY_PATH \
   -x NCCL_DEBUG \
   -x NCCL_NET_GDR_LEVEL \
   -x NCCL_DEBUG_SUBSYS \
   -x NCCL_IB_DISABLE \
   -np $GPU_NUM --host $HOST_CONFIG \
   --map-by ppr:4:numa --bind-to core --report-bindings \
python $EXECUBLE \
    --backend $BACKEND \
    --config ./scripts/configs/${CP_FILE_NAME}_${GPU_NUM}.json \
    2>/dev/null # Disable Warning


set +x

done
done
done
done
