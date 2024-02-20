# CUDA_LAUNCH_BLOCKING=1 \
GPU_NUM=4
NODE_NUM=1
NTASK_PER_NODE=`expr $GPU_NUM / $NODE_NUM`
echo "GPU_NUM: $GPU_NUM"
# export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
srun -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w nico3 --exclusive \
python all2all_test.py \
    --gpus $GPU_NUM
