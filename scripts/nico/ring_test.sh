# CUDA_LAUNCH_BLOCKING=1 \
GPU_NUM=4
echo "GPU_NUM: $GPU_NUM"
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
srun -N 1 -w nico3 --exclusive \
python ring_test.py \
    --gpus $GPU_NUM
