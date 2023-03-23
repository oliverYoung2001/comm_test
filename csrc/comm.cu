// #include "nccl.h"
#include "comm.h"

void all2all_SC0(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < comm_size; ++ i) {
        // if (i != rank) {
        if (true) {
            NCCL_CHECK(ncclSend(input_list[i], CHUNK_SIZE, ncclDataType, i, comm, stream));
            NCCL_CHECK(ncclRecv(output_list[i], CHUNK_SIZE, ncclDataType, i, comm, stream));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
    if (async_op == false) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
