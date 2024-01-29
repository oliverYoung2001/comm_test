include ../common.mk
.PHONY: clean

all: all2all_test net_test net_test_cuda conflict_test_cuda net_test_mpi conflict_allinone net_test_cpu nccl_test cpu2gpu \
	 benchmark_p2p ring_patterns all2all_baseline single_turn mixed_p2p netflow all2allv_MPI all2allv_allinone

%_clean:
	rm $@.o
	rm build/$@

clean:
	rm -f `ls *.o | grep -v jsoncpp.o`
	rm -f all2all_test net_test net_test_cuda conflict_test_cuda net_test_mpi conflict_allinone net_test_cpu nccl_test cpu2gpu \
	   benchmark_p2p ring_patterns all2all_baseline single_turn mixed_p2p netflow all2allv_MPI all2allv_allinone
	rm -rf build

%.o: %.cpp
	gcc $< -std=c++11 -c -o $@

%.o: %.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ $(GENCODE) -lnccl -lcudart -c -o $@

%: %.o comm.o jsoncpp.o utils.o
	mkdir -p build
	$(MPI_CXX) $? -no-pie -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o build/$@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib


# all2all_test: all2all_test.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# net_test: net_test.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# net_test_cuda: net_test_cuda.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# conflict_test_cuda: conflict_test_cuda.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# net_test_mpi: net_test_mpi.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# conflict_allinone: conflict_allinone.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -no-pie -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# net_test_cpu: net_test_cpu.o jsoncpp.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -Wall -o $@

# nccl_test: nccl_test.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# cpu2gpu: cpu2gpu.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# benchmark_p2p: benchmark_p2p.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# ring_patterns: ring_patterns.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# all2all_baseline: all2all_baseline.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# single_turn: single_turn.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# mixed_p2p: mixed_p2p.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# netflow: netflow.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -no-pie -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# all2allv_MPI: all2allv_MPI.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -no-pie -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib

# all2allv_allinone: all2allv_allinone.o comm.o jsoncpp.o utils.o
# 	$(MPI_CXX) $? -no-pie -std=c++11 -fopenmp -L$(NCCL_BUILD_PATH)/lib -lcudart -lnccl -Wall -o $@ -Wl,-rpath=$(NCCL_BUILD_PATH)/lib
