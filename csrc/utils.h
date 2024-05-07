#ifndef __UTILS_H__
#define __UTILS_H__
#include <cstdio>
#include "cuda_runtime.h"
#include "json/json.h"
#include <cstdlib>
#include "nccl.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cstdint>
#include <curand.h>
#include <mpi.h>
#include <string>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <curand_kernel.h>
#include <cmath>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include "assert.h" 
#include<algorithm>
#include "constant.h"
#include <set>

typedef long long LL;

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


struct PROC_PARAMS {
    std::string host;
    int rank;
    int local_rank;
    int cur_rank;
    int comm_size;
    int nodes;
    int nodeid;
    int tasks_per_node;
    int N_GPUs;
    std::string ip;
    std::string clustername;
    std::string nodename;
    std::string RECORD_P2P;
    bool ENABLE_GPU_P2P;        // 性能不一定好！！！ 单个P2P更好，但多个P2P不一定好

    std::string BACKEND;
    ncclComm_t comm;
    ncclComm_t cur_comm;
    std::set<int> r_s;

    cudaStream_t* streams;
    MPI_Request* mpi_requests;

    // void (*XXX_comm)(int** send_buf, int** recv_buf, LL SIZE, \
    //            cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request);

    PROC_PARAMS(int _N_GPUs = 0) {
        host = "";
        rank = local_rank = cur_rank = 0;
        nodes = 1;
        nodeid = 0;
        comm_size = tasks_per_node = _N_GPUs;
        N_GPUs = _N_GPUs;
        ip = "";
        clustername = "";
        nodename = "";
        ENABLE_GPU_P2P = false;

        streams = NULL;
        mpi_requests = NULL;
    }

    ~PROC_PARAMS() {
        if (streams != NULL) {
            delete[] streams;
        }
        if (mpi_requests != NULL) {
            delete[] mpi_requests;
        }
    }

    void init_cudaStream(int STREAM_NUM) {
        streams = new cudaStream_t[STREAM_NUM];
        for (int i = 0; i < STREAM_NUM; ++ i) {
            cudaStreamCreate(&streams[i]);
        }
    }

    void init_MPI_Request(int REQ_NUM) {
        mpi_requests = new MPI_Request[REQ_NUM];
    }

    // void set_N_GPUs(int _N_GPUs = 0) {
    //     host = "";
    //     rank = local_rank = 0;
    //     nodes = 1;
    //     nodeid = 0;
    //     comm_size = tasks_per_node = N_GPUs = _N_GPUs;
    //     ip = "";
    //     clustername = "";
    //     nodename = "";
    // }
};

void create_comm_group_from_pattern(PROC_PARAMS*& pp, Json::Value& pairs);

void barrier(std::string& BACKEND, int N_GPUs);

void enableP2P(Json::Value& pairs);

void disableP2P(Json::Value& pairs);

void enableP2P(int ngpus);

void disableP2P(int ngpus);

void enableP2P(int i, int j);

void disableP2P(int i, int j);

void check_UVA(int ngpus);

// cudaMemcpy_comm: 不适用于多机
void cudaMemcpy_comm(PROC_PARAMS*& pp, Json::Value& pairs, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request);
// NCCL_comm
void NCCL_comm(PROC_PARAMS*& pp, Json::Value& pairs, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request);
// MPI_comm
void MPI_comm(PROC_PARAMS*& pp, Json::Value& pairs, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request);

int parse_env(std::string key, std::string& value);

int parse_env2int(std::string key, int& value);

void get_proc_params(PROC_PARAMS* pp);

void setup_env(PROC_PARAMS*& pp, int argc, char** argv);

#endif