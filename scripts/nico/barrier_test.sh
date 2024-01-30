# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
HOST=nico3
echo "HOST: ${HOST}"
for GPUs in 1 2 3 4 5 6 7 8; do
srun -N 1 -w $HOST --exclusive \
python barrier_test.py \
    --gpus $GPUs
done