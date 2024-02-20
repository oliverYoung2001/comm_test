#!/bin/bash

RANK=$SLURM_PROCID

# LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK # for OpenMPI
# LOCAL_RANK=$MPI_LOCALRANKID # for Intel MPI
LOCAL_RANK=$SLURM_LOCALID # for SLURM

# LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE # for OpenMPI
# LOCAL_SIZE=$MPI_LOCALNRANKS # for Intel MPI
LOCAL_SIZE=$SLURM_TASKS_PER_NODE # for SLURM

NCPUS=$(nproc --all) # eg: 28
NUM_NUMA=2

# calculate binding parameters
# bind to sequential cores in a NUMA domain
# CORES_PER_PROCESS=$(($NCPUS / $LOCAL_SIZE)) # eg: 7 when LOCAL_SIZE=4
# NUMA_ID=$(($LOCAL_RANK / $NUM_NUMA)) # eg: 0, 0, 1, 1
# NUMA_OFFSET=$(($LOCAL_RANK % $NUM_NUMA)) # 0, 1, 0, 1
# CORE_START=$(($NUMA_ID * $CORES_PER_PROCESS * $NUM_NUMA + $NUMA_OFFSET)) # eg: 0, 1, 14, 15
# CORE_END=$((($NUMA_ID + 1) * $CORES_PER_PROCESS * $NUM_NUMA - $NUM_NUMA + $NUMA_OFFSET)) # eg: 12, 13, 26, 27
# CORES=$(seq -s, $CORE_START $NUM_NUMA $CORE_END) # eg: 0,2,4,6,8,10,12 for rank 0

NUMA_ID=$LOCAL_RANK
NUMA_ID=0
if [ $LOCAL_RANK == 0 ]; then
    CORES=37
else
    CORES=56
fi
# CORES=`expr ${LOCAL_RANK} \* 16`
# execute command with specific cores
if [ $RANK == 0 ]; then
    echo "TOTAL CPUs: $NCPUS"
fi
echo "Process $LOCAL_RANK on $(hostname) bound to NUMA domain $NUMA_ID & core $CORES"
exec numactl -m "$NUMA_ID" -C "$CORES" $@
# exec numactl -m "$NUMA_ID" $@
# exec $@