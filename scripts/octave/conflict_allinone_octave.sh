# # build nccl
# pushd /home/yhy/.local/nccl
# # rm -r build
# git checkout master   # for debug
# # git checkout v2.17.1-1
# # git checkout v2.10.3-1      # 性能弱于latest
# make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
# popd

# export RECORD_P2P=1
EXECUBLE=conflict_allinone
GPU_NUMs="8"
# BACKEND=NCCL    # Baseline
# BACKEND=cudaMemcpy
# BACKEND=MPI     # MPI_Isend/MPI_Irecv 时内存泄露？
# BACKENDs="NCCL MPI cudaMemcpy-P cudaMemcpy-nP"
# BACKENDs="cudaMemcpy-P cudaMemcpy-nP"
# BACKENDs="NCCL cudaMemcpy"
# BACKENDs="cudaMemcpy"
# BACKENDs="NCCL MPI"
# BACKENDs="NCCL"
# BACKENDs="MPI"
BACKENDs="cudaMemcpy-P"
# CP_FILE_NAMEs="p2p_si p2p_bi"
CP_FILE_NAMEs="conflict_patterns"
# CP_FILE_NAMEs="bad_patterns_3+2"
# CP_FILE_NAMEs="bad_patterns_3+3"
# CP_FILE_NAMEs="bad_patterns_pcie_switch"
# CP_FILE_NAMEs="all2all_4"
# CP_FILE_NAMEs="small"
HOSTs="octave"
# PARTITION=SXM
# HOSTs="zoltan"
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
# srun -p $PARTITION -N $NODE_NUM --ntasks-per-node $NTASK_PER_NODE -w $HOST --exclusive \
mpirun -np $NTASK_PER_NODE \
./csrc/build/${EXECUBLE} $GPU_NUM $BACKEND csrc/configs/${CP_FILE_NAME}.json

if [ "$RECORD_P2P" ]; then
   python utils/build_excel.py \
      --input_file_name "P2P_${BACKEND}_${GPU_NUM}_${HOST}" \
      # --output_file_excel "results/P2P_${BACKEND}_${GPU_NUM}_${HOST}.xlsx"
fi

done
done
done
done