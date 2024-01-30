set -o xtrace

EXECUBLE=all2all_test
GPU_NUMs="16"
# BACKEND=NCCL    # Baseline
# BACKEND=cudaMemcpy
# BACKEND=MPI     # MPI_Isend/MPI_Irecv 时内存泄露？
# BACKENDs="NCCL cudaMemcpy MPI"
# BACKENDs="NCCL cudaMemcpy"
# BACKENDs="cudaMemcpy"
# BACKENDs="NCCL"
# CP_FILE_NAMEs="p2p_si p2p_bi"
# CP_FILE_NAMEs="conflict_patterns"
# CP_FILE_NAMEs="bad_patterns_3+3"
# CP_FILE_NAMEs="bad_patterns_pcie_switch"
# CP_FILE_NAMEs="all2all"
# CP_FILE_NAMEs="small"
# PARTITION=SXM
# export HOST=zoltan
PARTITION=V100
HOSTs="nico4"
# PARTITION=Big
# HOSTs="nico2"


make clean
make $EXECUBLE

# mkdir results
mkdir -p results

# CUDA_LAUNCH_BLOCKING=1 \
# for GPU_NUM in $GPUs; do # for NCCL, MPI
# for BACKEND in $BACKENDs; do
for HOST in $HOSTs; do
export HOST
for GPU_NUM in $GPU_NUMs; do       # for cudaMemcpy
# for CP_FILE_NAME in $CP_FILE_NAMEs; do
# echo "CP_FILE_NAME: ${CP_FILE_NAME}"

PROC_NUM=$GPU_NUM
if [ $GPU_NUM -le 8 ]; then
   NODE_NUM=1
else
   NODE_NUM=2
fi
# NODE_NUM=2
NTASK_PER_NODE=`expr $PROC_NUM / $NODE_NUM`

# export CUDA_VISIBLE_DEVICES=2,3,4,5
# export CUDA_VISIBLE_DEVICES=0,2
echo "HOST: $HOST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU_NUM: $GPU_NUM"
echo "PROC_NUM: $PROC_NUM"
# /home/spack/opt/spack/linux-debian11-cascadelake/gcc-10.2.1/cuda-11.3.1-ufjgm56ezd6xzo2wzmja56jev7nbolec/bin/ncu nvprof \
# srun -p $PARTITION -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w $HOST --exclusive \
srun -p $PARTITION -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w nico3,nico4 --exclusive \
./csrc/${EXECUBLE} $GPU_NUM $BACKEND csrc/configs/${CP_FILE_NAME}.json
python utils/plot.py --gpus $GPU_NUM

# done
# done
done
done

# make clean
# make

# # mkdir results
# mkdir -p results

# # CUDA_LAUNCH_BLOCKING=1 \
# for GPU_NUM in 8; do
# # for GPU_NUM in 4 8 16; do
# if [ $GPU_NUM -le 8 ]; then
#    NODE_NUM=1
# else
#    NODE_NUM=2
# fi
# NODE_NUM=2
# NTASK_PER_NODE=`expr $GPU_NUM / $NODE_NUM`
# echo "GPU_NUM: $GPU_NUM"
# # export CUDA_VISIBLE_DEVICES=2,3,4,5
# # export CUDA_VISIBLE_DEVICES=0,2
# echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
# # srun -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w nico3,nico4 \
# srun -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w nico3 \
# ./csrc/all2all_test
# python utils/plot.py --gpus $GPU_NUM

# done
