#!/bin/bash

BACKENDs="NCCL MPI cudaMemcpy-P cudaMemcpy-nP"
# BACKENDs="NCCL cudaMemcpy"
# BACKENDs="cudaMemcpy"
BACKENDs="NCCL MPI"
BACKENDs="MPI"
# BACKENDs="NCCL"
CP_FILE_NAMEs="c2g"
DIR_MODEs="0 1 2"


# nico:
PARTITION=Mix
# export GPU_NUM=16
GPU_NUMs="8"
# GPU_NUMs="16"
HOSTs="nico3"

# qy:
# PARTITION=gpu4-low
# HOST="g4003"
# GPU_NUM=8

# HOSTs="None"
export MASTER_PORT=$((RANDOM % 12000 + 10000))



EXECUBLE=conflict_allinone_c2g

make clean
make $EXECUBLE

# mkdir results
mkdir -p results

for BACKEND in $BACKENDs; do
echo "BACKEND: $BACKEND"
for HOST in $HOSTs; do
# echo "HOST: $HOST"
for GPU_NUM in $GPU_NUMs; do       # for cudaMemcpy
for CP_FILE_NAME in $CP_FILE_NAMEs; do
echo "CP_FILE_NAME: ${CP_FILE_NAME}"
for DIR_MODE in $DIR_MODEs; do
echo "DIR_MODE: ${DIR_MODE}"

if [ $GPU_NUM -le 8 ]; then
   NNODES=1
else
   NNODES=$(($GPU_NUM / 8))
fi
NGPU_PER_NODE=`expr $GPU_NUM / $NNODES`
# [NOTE]: not support cudaMemcpy for multi machines
if [[ "$BACKEND" =~ "cudaMemcpy" ]]; then # "$str1" =~ "$str2" means whether $str1 contains $str2
   NTASK_PER_NODE=1
else
   NTASK_PER_NODE=$NGPU_PER_NODE
fi

echo "NNODES: $NNODES"
echo "NGPU_PER_NODE: $NGPU_PER_NODE"
# SLURM ARGS:
# MEM_PER_CPU=256G # 
# MEM_PER_NODE=256G
# --mem-per-cpu $MEM_PER_CPU \
# --cpus-per-task $NGPU_PER_NODE \
SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node $NTASK_PER_NODE \
--gres=gpu:$NGPU_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

export SLURM_CPU_BIND=verbose
# --cpu-bind=map_cpu:1,2,3,4,16,17,18,19 \

set -x
# salloc -n $GPU_NUM
srun $SLURM_ARGS \
--cpu-bind=map_cpu:1,2,3,4,16,17,18,19 \
./scripts/executor.sh \
./csrc/build/${EXECUBLE} $GPU_NUM $BACKEND ./scripts/configs/${CP_FILE_NAME}_${GPU_NUM}.json ${DIR_MODE}
set +x

done
done
done
done
done