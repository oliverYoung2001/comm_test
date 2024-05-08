#!/bin/bash

# build nccl
pushd ./third_party/nccl
rm -r build
# git checkout master   # for debug
# git checkout v2.17.1-1
# git checkout v2.10.3-1      # 性能弱于latest
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
popd

# configs:
BACKENDs="NCCL MPI cudaMemcpy-P cudaMemcpy-nP"
# BACKENDs="NCCL cudaMemcpy"
# BACKENDs="cudaMemcpy"
BACKENDs="NCCL MPI"
BACKENDs="MPI"
BACKENDs="NCCL"
# CP_FILE_NAMEs="p2p_si p2p_bi"
# CP_FILE_NAMEs="p2p_si"
CP_FILE_NAMEs="conflict_patterns"
# CP_FILE_NAMEs="bad_patterns_3+2"
# CP_FILE_NAMEs="bad_patterns_3+3"
# CP_FILE_NAMEs="bad_patterns_pcie_switch"
# CP_FILE_NAMEs="all2all_4"
CP_FILE_NAMEs="E2E_4 E2E_8"
CP_FILE_NAMEs="ring_16"
CP_FILE_NAMEs="small"


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



EXECUBLE=conflict_allinone

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
# salloc -n $GPU_NUM
# srun $SLURM_ARGS \
# ./scripts/executor.sh \
# GPU_NUM=2
# mpirun --prefix $(dirname `which mpirun`)/../ -np 2 --host g3022:2 \
# mpirun --prefix $(dirname `which mpirun`)/../ -np 2 --host g3027:1,g3028:1 \
# ./csrc/build/${EXECUBLE} 2 $BACKEND ./scripts/configs/${CP_FILE_NAME}.json
# mpirun --prefix $(dirname `which mpirun`)/../ -x LD_LIBRARY_PATH -np 2 --host g4007:1,g4008:1 \
# ./csrc/build/${EXECUBLE} 2 $BACKEND ./scripts/configs/${CP_FILE_NAME}.json
GPU_NUM=16
mpirun --prefix $(dirname `which mpirun`)/../ -x LD_LIBRARY_PATH -x NCCL_DEBUG=WARN \
   -np $GPU_NUM --host g4007:8,g4008:8,g4005:8 \
./csrc/build/${EXECUBLE} $GPU_NUM $BACKEND ./scripts/configs/${CP_FILE_NAME}_${GPU_NUM}.json

done
done
done
done