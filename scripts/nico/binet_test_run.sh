HOST=nico3
for GPU0 in 0 1 2 3 4 5 6 7; do
for GPU1 in 0 1 2 3 4 5 6 7; do
# if [ $GPU0 -lt $GPU1 ]; then
if [ $GPU0 -ne $GPU1 ]; then
echo "GPUs: ${GPU0}, ${GPU1}"
CUDA_VISIBLE_DEVICES=${GPU0},${GPU1} \
srun -N 1 -w $HOST --exclusive \
python binet_test.py \
    --gpus 2 \
    --gpuids $GPU0,$GPU1 \
    --excel_file ./results/binet_test_$HOST.xlsx

fi
done
done