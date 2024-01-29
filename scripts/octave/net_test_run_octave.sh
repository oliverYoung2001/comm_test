
# srun -N 1 -w nico3 \
# for GPU0 in 0 1 2 3 4 5 6; do
# for GPU1 in 0 1 2 3 4 5 6; do
for GPU0 in 0 1 2 3 4 5 6; do
for GPU1 in 1 2 3 4 5 6; do
if [ $GPU0 -lt $GPU1 ]; then
echo "GPUs: ${GPU0}, ${GPU1}"
CUDA_VISIBLE_DEVICES=${GPU0},${GPU1} \
python net_test.py \
    --gpus 2 \
    --gpuids $GPU0,$GPU1

fi
done
done