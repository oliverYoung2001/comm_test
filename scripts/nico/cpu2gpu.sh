# build nccl
# pushd /home/yhy/.local/nccl
# rm -r build
# git checkout master   # for debug
# # git checkout v2.17.1-1
# # git checkout v2.10.3-1      # 性能弱于latest
# make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
# popd

EXECUBLE=cpu2gpu
GPU_NUMs="8"
# BACKEND=NCCL    # Baseline
# BACKEND=cudaMemcpy
# BACKEND=MPI     # MPI_Isend/MPI_Irecv 时内存泄露？
# BACKENDs="NCCL cudaMemcpy MPI"
BACKENDs="cudaMemcpy"
# CP_FILE_NAMEs="p2p_bi p2p_si"
# CP_FILE_NAMEs="conflict_patterns"
# CP_FILE_NAMEs="small"
CP_FILE_NAMEs="cpu2gpu"
# PARTITION=SXM
# export HOST=zoltan
# PARTITION=V100
# HOSTs="nico4"
PARTITION=Big
HOSTs="nico2"

make clean
make $EXECUBLE

# mkdir results
mkdir -p results

# CUDA_LAUNCH_BLOCKING=1 \
# for GPU_NUM in $GPUs; do # for NCCL, MPI
for BACKEND in $BACKENDs; do
for HOST in $HOSTs; do
export HOST
for GPU_NUM in $GPU_NUMs; do       # for cudaMemcpy
for CP_FILE_NAME in $CP_FILE_NAMEs; do
echo "CP_FILE_NAME: ${CP_FILE_NAME}"

if [ "$BACKEND" == "cudaMemcpy" ]; then
   PROC_NUM=1
else
   PROC_NUM=$GPU_NUM
fi
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
srun -p $PARTITION -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w $HOST --exclusive \
./csrc/${EXECUBLE} $GPU_NUM $BACKEND csrc/configs/${CP_FILE_NAME}.json
# ./scripts/wrapper.sh ./csrc/${EXECUBLE} $GPU_NUM $BACKEND csrc/configs/${CP_FILE_NAME}.json
# python utils/build_excel.py \
#    --input_file_name "P2P_${BACKEND}_${GPU_NUM}_${HOST}" \

done
done
done
done