# # build nccl
# pushd /home/yhy/.local/nccl
# # rm -r build
# git checkout master   # for debug
# # git checkout v2.17.1-1
# # git checkout v2.10.3-1      # 性能弱于latest
# make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
# popd

# export RECORD_P2P=1
EXECUBLE=all2allv_allinone
GPU_NUMs="4"
# BACKEND=NCCL    # Baseline
# BACKEND=cudaMemcpy
# BACKEND=MPI     # MPI_Isend/MPI_Irecv 时内存泄露？
# BACKENDs="cudaMemcpy-P cudaMemcpy-nP NCCL MPI"
# BACKENDs="NCCL cudaMemcpy"
# BACKENDs="cudaMemcpy"
# BACKENDs="NCCL MPI"
# BACKENDs="NCCL"
BACKENDs="MPI NCCL"
CP_FILE_NAMEs="all2allv_4"
PARTITION=SXM
HOSTs="zoltan"
# PARTITION=V100
# HOSTs="nico3,nico4"
# PARTITION=Big
# # HOSTs="nico1,nico2"
# HOSTs="nico2"

make clean
make $EXECUBLE

# mkdir results
mkdir -p results


for BACKEND in $BACKENDs; do
for HOST in $HOSTs; do
export HOST
for GPU_NUM in $GPU_NUMs; do       # for cudaMemcpy
for CP_FILE_NAME in $CP_FILE_NAMEs; do
echo "CP_FILE_NAME: ${CP_FILE_NAME}"

if [[ "$BACKEND" =~ "cudaMemcpy" ]]; then # "$str1" =~ "$str2" means whether $str1 contains $str2
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

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,2
echo "HOST: $HOST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU_NUM: $GPU_NUM"
echo "PROC_NUM: $PROC_NUM"
echo "NODE_NUM: $NODE_NUM"
echo "NTASK_PER_NODE: $NTASK_PER_NODE"
# ncu nvprof \
srun -p $PARTITION -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w $HOST --exclusive \
./csrc/${EXECUBLE} $GPU_NUM $BACKEND csrc/configs/${CP_FILE_NAME}.json

done
done
done
done