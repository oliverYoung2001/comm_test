#!/bin/bash

# # build nccl
# pushd ./third_party/nccl
# rm -r build
# git checkout master   # for debug
# # git checkout v2.18.6-1
# # git checkout v2.10.3-1      # 性能弱于latest
# make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
# # make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
# popd
# exit -1


# configs:
BACKENDs="NCCL MPI cudaMemcpy-P cudaMemcpy-nP"
# BACKENDs="NCCL cudaMemcpy"
# BACKENDs="cudaMemcpy"
BACKENDs="NCCL MPI"
BACKENDs="MPI"
BACKENDs="NCCL"


# nico:
export CLUSTER_NAME=nico
PARTITION=Mix
HOSTs="nico1,nico2"
# qy:
# PARTITION=gpu3-2-low
PARTITION=gpu4-low
# # HOSTs="g4003"
HOSTs="g4002,g4003"
# # GPU_NUM=8

GPU_NUMs="16"
# HOSTs="None"
export MASTER_PORT=$((RANDOM % 12000 + 10000))



EXECUBLE=coll_comm_bench

# make clean
# make $EXECUBLE

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

# # Use slurm (not support for qy)
# MEM_PER_CPU=256G
# MEM_PER_NODE=256G
# # --mem-per-cpu $MEM_PER_CPU \
# # --mem $MEM_PER_NODE \
# SLURM_ARGS="
# -p $PARTITION \
# -N $NNODES \
# --ntasks-per-node $NTASK_PER_NODE \
# --gres=gpu:$NTASK_PER_NODE \
# --mem $MEM_PER_NODE \
# -K \
# "
# if [ "$HOST" != "None" ]; then
#     SLURM_ARGS="$SLURM_ARGS \
#         -w $HOST \
#     "
# fi

# set -x
# srun $SLURM_ARGS \
# ./scripts/executor.sh \
# ./csrc/build/${EXECUBLE} $GPU_NUM $BACKEND


# # Use mpirun
# [NOTE]: on nico, exec:
# srun -p Mix -N 2 -w nico[1-2] --gres=gpu:8 sleep 10000

# GPU_NUM=16
# HOST_CONFIG="nico1:8,nico2:8"
# set -x
# mpirun \
#    -np $GPU_NUM --host $HOST_CONFIG \
# ./csrc/build/${EXECUBLE} $GPU_NUM $BACKEND

# on qy
GPU_NUM=64
HOST_CONFIG="g4001:8,g4002:8,g4003:8,g4004:8,g4005:8,g4006:8,g4007:8,g4008:8"
# GPU_NUM=32
# HOST_CONFIG="g4005:8,g4006:8,g4007:8,g4008:8"
# GPU_NUM=24
# HOST_CONFIG="g4005:8,g4006:8,g4007:8,g4008:8"
# GPU_NUM=16
# HOST_CONFIG="g4005:8,g4006:8,g4007:8,g4008:8"
# HOST_CONFIG="g3025:8,g3029:8"
# HOST_CONFIG="g4002:8,g4003:8"
set -x
mpirun --prefix $(dirname `which mpirun`)/../ -x LD_LIBRARY_PATH \
   -np $GPU_NUM --host $HOST_CONFIG \
./csrc/build/${EXECUBLE} $GPU_NUM $BACKEND



done
done
done