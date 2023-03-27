# COCONET_FLAGS = -I../../src ../../src/codegen.cpp ../../src/dsl.cpp ../../src/pipeline.cpp ../../src/utils.cpp
# SCHEDULE_FLAGS = -I../
COMM_TEST_PATH = /home/yhy/fold/comm_test
# COCONET_PATH = /home/yhy/AI_Compiler/coconet
NCCL_PATH = /home/yhy/.local/nccl
NCCL_BUILD_PATH = $(NCCL_PATH)/build
# NCCL_OVERLAP_PATH = $(COCONET_PATH)/nccl-overlap
# NCCL_OVERLAP_BUILD_PATH = $(NCCL_OVERLAP_PATH)/build
# MPI_CXX = /usr/bin/mpicxx
MPI_CXX = "`which mpicxx`"
GENCODE = "-gencode=arch=compute_80,code=sm_80"
