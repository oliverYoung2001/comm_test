include ../common.mk
.PHONY: clean

all: all2all_test net_test net_test_cuda conflict_test_cuda net_test_mpi conflict_allinone net_test_cpu nccl_test

%.o: %.cpp
	gcc $< -std=c++11 -c -o $@

%.o: %.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ $(GENCODE) -lnccl -lcudart -c -o $@

all2all_test: all2all_test.o comm.o jsoncpp.o
	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

net_test: net_test.o comm.o jsoncpp.o
	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

net_test_cuda: net_test_cuda.o comm.o jsoncpp.o
	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

conflict_test_cuda: conflict_test_cuda.o comm.o jsoncpp.o
	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

net_test_mpi: net_test_mpi.o comm.o jsoncpp.o
	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

conflict_allinone: conflict_allinone.o comm.o jsoncpp.o
	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

net_test_cpu: net_test_cpu.o jsoncpp.o
	$(MPI_CXX) $? -std=c++11 -fopenmp -Wall -o $@

nccl_test: nccl_test.o comm.o jsoncpp.o
	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib


clean:
	rm all2all_test net_test net_test_cuda conflict_test_cuda net_test_mpi conflict_allinone net_test_cpu nccl_test
	rm `ls *.o | grep -v jsoncpp.o`