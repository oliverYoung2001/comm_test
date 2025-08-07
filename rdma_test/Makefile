.PHONY: clean

build_dir:
	mkdir -p build

rdma_server: build_dir
	nvcc -std=c++17 rdma_server.cpp -o build/rdma_server -libverbs -lrdmacm

rdma_client: build_dir
	nvcc -std=c++17 rdma_client.cpp -o build/rdma_client -libverbs -lrdmacm

rdma_p2p_gpt: build_dir
	nvcc -std=c++17 rdma_p2p_gpt.cpp -o build/rdma_p2p_gpt -libverbs -lrdmacm

topo_discover: build_dir
	nvcc topo_discover.cpp -o build/topo_discover -libverbs -lrdmacm

%_clean:
	rm -f $*.o
	rm -f build/$*

clean:
	rm -f `ls *.o | grep -v jsoncpp.o`
# 	rm -f all2all_test net_test net_test_cuda conflict_test_cuda net_test_mpi conflict_allinone net_test_cpu nccl_test cpu2gpu \
# 	   benchmark_p2p ring_patterns all2all_baseline single_turn mixed_p2p netflow all2allv_MPI all2allv_allinone
	rm -rf build
