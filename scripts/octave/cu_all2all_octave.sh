# build nccl
pushd /home/yhy/.local/nccl
rm -r build
git checkout master   # for debug
# git checkout v2.17.1-1
# git checkout v2.10.3-1      # 性能弱于latest
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
popd

# build src
make clean
make

# mkdir results
mkdir -p results

# CUDA_LAUNCH_BLOCKING=1 \
for GPU_NUM in 4; do

echo "GPU_NUM: $GPU_NUM"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=2,3,4,5
export CUDA_VISIBLE_DEVICES=3,4,5,6
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
# srun -N 1 -n $GPU_NUM -w nico3 --exclusive \
mpirun -np $GPU_NUM ./csrc/all2all_test
python utils/plot.py --gpus $GPU_NUM

done
