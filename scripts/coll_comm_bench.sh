export GPU_NUMs="4 8"
export GPU_NUMs="2 4 8"
# qy:
export CLUSTER_NAME=qy
PARTITION=gpu4-low
HOST="g4004"
# HOST=None

MEM_PER_CPU=256G
MEM_PER_NODE=256G
NNODES=1

EXECUBLE=coll_comm_bench.py

echo "PARTITION: ${PARTITION}"
echo "HOST: ${HOST}"

for GPU_NUM in $GPU_NUMs; do
echo "GPU_NUM: $GPU_NUM"

# --mem-per-cpu $MEM_PER_CPU \
# --mem $MEM_PER_NODE \
SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node $GPU_NUM \
--gres=gpu:$GPU_NUM \
--mem $MEM_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

srun $SLURM_ARGS \
python $EXECUBLE

done