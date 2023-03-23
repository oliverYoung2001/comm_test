#ifndef __UTILS_H__
#define __UTILS_H__
#include <cstdio>
#include "cuda_runtime.h"
#include "nccl.h"

// #define CUBLASCHECK(cmd) do {                       \
//   cublasStatus_t e = cmd;                           \
//   if (e != CUBLAS_STATUS_SUCCESS) {                 \
//     printf("Failed: CUBLAS error %s: %d '%d'\n",    \
//            __FILE__, __LINE__, cmd);                \
//     assert(false);                                  \
//   }                                                 \
// } while(0)                                          \

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#endif