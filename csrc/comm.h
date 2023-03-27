#ifndef __COMM_H__
#define __COMM_H__
#include "nccl.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "utils.h"
typedef long long LL;
void all2all_SC0(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, \
                 ncclComm_t comm, ncclDataType_t ncclDataType,  cudaStream_t stream, bool async_op);
                 
void all2all_SC1(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, \
                 ncclComm_t comm, ncclDataType_t ncclDataType,  cudaStream_t stream, bool async_op);

void all2all_SC4(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, \
                 ncclComm_t comm, ncclDataType_t ncclDataType,  cudaStream_t stream, bool async_op);

void all2all_BRUCK(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, \
                 ncclComm_t comm, ncclDataType_t ncclDataType,  cudaStream_t stream, bool async_op);

void all2all_RD(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, \
                 ncclComm_t comm, ncclDataType_t ncclDataType,  cudaStream_t stream, bool async_op);

void all2all_2DMESH(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, \
                 ncclComm_t comm, ncclDataType_t ncclDataType,  cudaStream_t stream, bool async_op);

void all2all_3DMESH(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, \
                 ncclComm_t comm, ncclDataType_t ncclDataType,  cudaStream_t stream, bool async_op);

#endif