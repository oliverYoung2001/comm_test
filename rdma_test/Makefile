.PHONY: clean

build_dir:
	mkdir -p build

rdma_server: build_dir
	nvcc -std=c++17 rdma_server.cpp -o build/rdma_server -libverbs -lrdmacm

rdma_client: build_dir
	nvcc -std=c++17 rdma_client.cpp -o build/rdma_client -libverbs -lrdmacm

rdma_p2p_gpt: build_dir
	nvcc -std=c++17 rdma_p2p_gpt.cpp -o build/rdma_p2p_gpt -libverbs -lrdmacm

rdma_p2p_gpt5: build_dir
	nvcc -std=c++17 rdma_p2p_gpt5.cpp -o build/rdma_p2p_gpt5 -libverbs -lrdmacm

rdma_p2p_claude: build_dir
	nvcc -std=c++17 rdma_p2p_claude.cpp -o build/rdma_p2p_claude -libverbs -lrdmacm

topo_discover: build_dir
	nvcc topo_discover.cpp -o build/topo_discover -libverbs -lrdmacm

NVSHMEM_HOME ?= $(shell dirname $(shell which nvshmem-info))/..
MPI_HOME ?= $(shell dirname $(shell which mpicxx))/..

ibgda_p2p_nvshmem: build_dir
	nvcc -O3 -std=c++17 -arch=sm_90 -rdc=true -ccbin=mpicxx \
	-I"${NVSHMEM_HOME}/include" \
	$@.cu \
	-L"${NVSHMEM_HOME}/lib" -lnvshmem_host -lnvshmem_device \
	-Xlinker -rpath -Xlinker "${NVSHMEM_HOME}/lib" \
	-o build/$@

shift_mpi: build_dir
	nvcc -O3 -std=c++17 -arch=sm_90 -rdc=true -ccbin=mpicxx \
	-I"${NVSHMEM_HOME}/include" \
	$@.cu \
	-L"${NVSHMEM_HOME}/lib" -lnvshmem_host -lnvshmem_device \
	-Xlinker -rpath -Xlinker "${NVSHMEM_HOME}/lib" \
	-o build/$@

mpi_test: build_dir
	nvcc -O3 -std=c++17 -arch=sm_90 -rdc=true -ccbin=mpicxx \
	$@.cu \
	-o build/$@

%_clean:
	rm -f $*.o
	rm -f build/$*

clean:
	rm -f `ls *.o | grep -v jsoncpp.o`
# 	rm -f all2all_test net_test net_test_cuda conflict_test_cuda net_test_mpi conflict_allinone net_test_cpu nccl_test cpu2gpu \
# 	   benchmark_p2p ring_patterns all2all_baseline single_turn mixed_p2p netflow all2allv_MPI all2allv_allinone
	rm -rf build
