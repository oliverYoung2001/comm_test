for GPU0 in 4 0; do
for GPU1 in 3 7; do
python cluster_test.py \
    --gpus 2 \
    --gpuids ${GPU0},${GPU1}; \

done
done