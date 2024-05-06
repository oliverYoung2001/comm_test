#!/bin/bash

BACKENDs="NCCL MPI cudaMemcpy-P cudaMemcpy-nP"
# BACKENDs="NCCL cudaMemcpy"
# BACKENDs="cudaMemcpy"
BACKENDs="NCCL MPI"
BACKENDs="MPI"
BACKENDs="NCCL"


# nico:
PARTITION=Mix
# export GPU_NUM=16
# GPU_NUMs="8"

# qy:
PARTITION=gpu3-2-low
PARTITION=gpu4-low
# HOST="g4003"
# GPU_NUM=8

GPU_NUMs="16"
HOSTs="None"
export MASTER_PORT=$((RANDOM % 12000 + 10000))



EXECUBLE=coll_comm_bench

make clean
make $EXECUBLE

# mkdir results
mkdir -p results

for BACKEND in $BACKENDs; do
echo "BACKEND: $BACKEND"
for HOST in $HOSTs; do
echo "HOST: $HOST"
for GPU_NUM in $GPU_NUMs; do       # for cudaMemcpy


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

set -x
mpirun --prefix $(dirname `which mpirun`)/../ -x LD_LIBRARY_PATH -np 16 --host g4007:8,g4008:8 \
./csrc/build/${EXECUBLE} 16 $BACKEND

done
done
done