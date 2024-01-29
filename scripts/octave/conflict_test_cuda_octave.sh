# build nccl
# pushd /home/yhy/.local/nccl
# rm -r build
# git checkout master   # for debug
# # git checkout v2.17.1-1
# # git checkout v2.10.3-1      # 性能弱于latest
# make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
# popd

EXECUBLE=conflict_test_cuda

make clean
make $EXECUBLE

# mkdir results
mkdir -p results

GPUs=4

# CUDA_LAUNCH_BLOCKING=1 \
for GPU_NUM in 1; do
# for GPU_NUM in 4 8 16; do
if [ $GPU_NUM -le 8 ]; then
   NODE_NUM=1
else
   NODE_NUM=2
fi
# NODE_NUM=2
# HOST=nico4
NTASK_PER_NODE=`expr $GPU_NUM / $NODE_NUM`
echo "GPU_NUM: $GPU_NUM"
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# export CUDA_VISIBLE_DEVICES=0,2
echo "HOST: $HOST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPUs: $GPUs"
# srun -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w $HOST --exclusive \
./csrc/${EXECUBLE} $GPUs
# python utils/plot.py --gpus $GPU_NUM

done
