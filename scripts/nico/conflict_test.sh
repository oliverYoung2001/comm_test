# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
HOST=nico3
echo "HOST: ${HOST}"
for GPUs in 8; do
echo "GPUs: $GPUs"
srun -N 1 -w $HOST --exclusive \
python conflit_test.py \
    --gpus $GPUs \
    --excel_file ./results/binet_test_${HOST}_${GPUs}.xlsx

done