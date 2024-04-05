#!/bin/bash

# nico:
# PARTITION=SXM
# HOSTs="zoltan"
# PARTITION=V100
# HOSTs="nico3,nico4"
PARTITION=Big
# HOSTs="nico1,nico2"
HOSTs="nico2"

# qy:
PARTITION=gpu4-low
HOST="g4004"
# HOST=None

NNODES=1

MEM_PER_CPU=256G # 
MEM_PER_NODE=256G
# --mem-per-cpu $MEM_PER_CPU \
# --mem $MEM_PER_NODE \
SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node 1 \
--gres=gpu:8 \
--mem $MEM_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi


for GPU0 in 0 1 2 3 4 5 6 7; do
for GPU1 in 0 1 2 3 4 5 6 7; do
# if [ $GPU0 -lt $GPU1 ]; then
if [ $GPU0 -ne $GPU1 ]; then
echo "GPUs: ${GPU0}, ${GPU1}"
CUDA_VISIBLE_DEVICES=${GPU0},${GPU1} \
srun $SLURM_ARGS \
python binet_test.py \
    --gpus 2 \
    --gpuids $GPU0,$GPU1 \
    --excel_file ./results/binet_test_$HOST.xlsx

fi
done
done