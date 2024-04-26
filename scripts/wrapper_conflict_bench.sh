#!/bin/bash

# export RECORD_P2P=1
EXECUBLE=conflict_bench.py
GPU_NUMs="8"
GPU_NUMs="16"
BACKENDs="NCCL"
CP_FILE_NAMEs="p2p_si p2p_bi"
CP_FILE_NAMEs="p2p_bi"
CP_FILE_NAMEs="conflict_patterns"
# CP_FILE_NAMEs="bad_patterns_3+2"
# CP_FILE_NAMEs="bad_patterns_3+3"
# CP_FILE_NAMEs="bad_patterns_pcie_switch"
# CP_FILE_NAMEs="all2all_4"
# CP_FILE_NAMEs="E2E_4 E2E_8"
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


HOSTs="None"

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

# export CUDA_LAUNCH_BLOCKING=1
set -x
srun $SLURM_ARGS \
./scripts/executor.sh \
python $EXECUBLE \
    --backend $BACKEND \
    --config ./scripts/configs/${CP_FILE_NAME}.json \

done
done
done
done
