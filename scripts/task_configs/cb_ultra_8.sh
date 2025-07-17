CLUSTER_NAME=${CLUSTER_NAME:-'UNKNOWN-CLUSTER'}
PLATFORM=${PLATFORM:-'UNKNOWN-PLATFORM'}
EXP_NAME=kernel_profile_${CLUSTER_NAME}
PYTHON_EXECUTABLE=conflict_bench.py

# Parallel Parameters:
GPUS_PER_NODE=8
NPROC_PER_NODE=$GPUS_PER_NODE
NNODES=1
HOST='g0002'
HOST='g0030'
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
CPUS=128
CPU_PER_TASK=$((CPUS / NPROC_PER_NODE ))   # [NOTE]: Unnecessary for performance.
export MASTER_PORT=$((RANDOM % 12000 + 10000))

# Task-specific ARGS
BACKEND="NCCL"
CP_FILE_NAME="ultra_8"
COMM_MODULE="raw-nccl"

# [TODO]: here !!!
OUTPUT_DIR=${OUTPUT_DIR:-./prof_data/tmp}
mkdir -p $OUTPUT_DIR
OUTPUT_FILE_NAME=cb_${WORLD_SIZE}_${HOST}.log
OUTPUT_FILE=${OUTPUT_DIR}/${OUTPUT_FILE_NAME}
rm $OUTPUT_FILE

EXECUTABLE="./scripts/executor.sh \
    python $PYTHON_EXECUTABLE \
        --backend $BACKEND \
        --config ./scripts/configs/${CP_FILE_NAME}.json \
        --comm-module $COMM_MODULE \
        --output $OUTPUT_FILE \
"
        # 2>/dev/null \
        # $LOGGING_ARGS \

