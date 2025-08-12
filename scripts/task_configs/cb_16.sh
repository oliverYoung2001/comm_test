CLUSTER_NAME=${CLUSTER_NAME:-'UNKNOWN-CLUSTER'}
PLATFORM=${PLATFORM:-'UNKNOWN-PLATFORM'}
EXP_NAME=kernel_profile_${CLUSTER_NAME}
PYTHON_EXECUTABLE=conflict_bench.py

# Parallel Parameters:
PARTITION=H100
GPUS_PER_NODE=8
NPROC_PER_NODE=$GPUS_PER_NODE
NNODES=2
# HOST=${HOST:-'g[0278,0297]'}
HOST=${HOST:-'g[0297,0310]'}
HOST=${HOST:-'g[0291,0292]'}
# HOST=${HOST:-'g[0288,0290]'}
HOST=${HOST:-'g[0275,0276]'}
HOST=${HOST:-'g[0276,0278]'}
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
CPUS=176
CPU_PER_TASK=$((CPUS / NPROC_PER_NODE ))   # [NOTE]: Unnecessary for performance.
export MASTER_PORT=$((RANDOM % 12000 + 10000))

# Task-specific ARGS
BACKEND=${BACKEND:-"NCCL"}
CP_FILE_NAME=${CP_FILE_NAME:-small_${WORLD_SIZE}}
COMM_MODULE=${COMM_MODULE:-"raw-nccl"}
#   NCCL Args:
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=ERROR
# export NCCL_NET_GDR_LEVEL=5 # the same as 3
# export NCCL_NET_GDR_LEVEL=0   # Disable GDR
export NCCL_NET_GDR_LEVEL=1 # Best performance: NCCL_NET_GDR_LEVEL=1/2(default)
export NCCL_IB_DISABLE=0
export NCCL_DEBUG_SUBSYS=NET
export NCCL_NET=IB
# NCCL_P2P_LEVEL    # Unrelated to cross-machine communication
# export NCCL_TOPO_DUMP_FILE=../../database/bingxing/H800/m_configs/nccl_topo.json
# export NCCL_PROTO=Simple    # Useless

# # Check whether GDR(GPUDirect RDMA) is enabled.
# export NCCL_DEBUG=INFO  # If you find 'NET/IB/2/GDRDMA', GDR is enabled.

# Output location
OUTPUT_DIR=${OUTPUT_DIR:-./prof_data/tmp}
mkdir -p $OUTPUT_DIR
OUTPUT_FILE_NAME=cb_${WORLD_SIZE}_${HOST}_all.log
OUTPUT_FILE=${OUTPUT_DIR}/${OUTPUT_FILE_NAME}
rm -f $OUTPUT_FILE

EXECUTABLE="./scripts/executor.sh \
    python $PYTHON_EXECUTABLE \
        --backend $BACKEND \
        --config ./scripts/configs/${CP_FILE_NAME}.json \
        --comm-module $COMM_MODULE \
        --output $OUTPUT_FILE \
"
        # 2>/dev/null \
        # $LOGGING_ARGS \

