# build nccl
# pushd /home/yhy/.local/nccl
# rm -r build
# git checkout master   # for debug
# # git checkout v2.17.1-1
# # git checkout v2.10.3-1      # 性能弱于latest
# make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
# popd

EXECUBLE=net_test_cpu
# GPU_NUMs="8"
# BACKEND=NCCL    # Baseline
# BACKEND=cudaMemcpy
# BACKEND=MPI     # MPI_Isend/MPI_Irecv 时内存泄露？
# BACKENDs="NCCL cudaMemcpy MPI"
# BACKENDs="cudaMemcpy MPI"
# CP_FILE_NAMEs="p2p_si p2p_bi"
# CP_FILE_NAMEs="conflict_patterns"
# CP_FILE_NAMEs="small"
# PARTITION=SXM
# export HOST=zoltan
PARTITION=V100
export HOST=nico3
NODE_NUM=1
PROC_NUM=2

make clean
make $EXECUBLE

# mkdir results
mkdir -p results

# CUDA_LAUNCH_BLOCKING=1 \
# for GPU_NUM in $GPUs; do # for NCCL, MPI
# for CP_FILE_NAME in $CP_FILE_NAMEs; do
# echo "CP_FILE_NAME: ${CP_FILE_NAME}"
# for BACKEND in $BACKENDs; do
# for GPU_NUM in $GPU_NUMs; do       # for cudaMemcpy

# if [ "$BACKEND" == "cudaMemcpy" ]; then
#    PROC_NUM=1
# else
#    PROC_NUM=$GPU_NUM
# fi
# if [ $GPU_NUM -le 8 ]; then
#    NODE_NUM=1
# else
#    NODE_NUM=2
# fi
# # NODE_NUM=2
NTASK_PER_NODE=`expr $PROC_NUM / $NODE_NUM`
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# export CUDA_VISIBLE_DEVICES=0,2
echo "HOST: $HOST"
# echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
# echo "GPU_NUM: $GPU_NUM"
echo "NODE_NUM: $NODE_NUM"
echo "PROC_NUM: $PROC_NUM"
# srun -p $PARTITION -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w $HOST --exclusive \
srun -p $PARTITION -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w $HOST --cpu-bind=none --exclusive \
./scripts/wrapper.sh \
./csrc/${EXECUBLE}

# done
# done
# done