#include <stdio.h>
#include "cuda_runtime.h"
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
#include "comm.h"
#include <assert.h>
#include <cmath>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include "json/json.h"
#include <fstream>
#include "assert.h" 
#include<algorithm>
#include "utils.h"

// #include <format>    // need c++20
typedef long long LL;

PROC_PARAMS* pp;

// #define CHECK_RESULT
// #define PRINT_JSON
int TIMES = 10;
int WARMUP = 5;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZEIDX_START = 5;
const int SIZEIDX_END = 9;

const int SIZES_LEN = 10;
const LL SIZES[SIZES_LEN] = {   // int = 4B
    1LL * 256,                  // 1KB      // 打不满带宽
    1LL * 1024 * 1,             // 4KB      // 打不满带宽
    1LL * 1024 * 2,             // 8KB     // 会高一些!!! (仅在某些情况下)
    1LL * 1024 * 4,             // 16KB     // 会高一些!!!  （最好）
    // 1LL * 1024 * 8,             // 32KB     // 会高一些!!!  （最好）
    1LL * 1024 * 16,            // 64KB     // 会高一些!!!  （最好）
    // 1LL * 1024 * 64,            // 256KB    // 趋于稳定
    1LL * 1024 * 256,           // 1MB
    1LL * 1024 * 1024 * 1,      // 4MB      // 打不满带宽
    1LL * 1024 * 1024 * 32,     // 128MB
    1LL * 1024 * 1024 * 64,     // 256MB
    1LL * 1024 * 1024 * 128,    // 512MB
    // 1LL * 1024 * 1024 * 256,    // 1GB
    // 1LL * 1024 * 1024 * 512,    // 2GB
    // 1LL * 1024 * 1024 * 1024,   // 4GB      // 用cudaMemcpy，竟然有性能下降！！！
    // 1LL * 1024 * 1024 * 2048,   // 8GB
    // 1LL * 1024 * 1024 * 4096,   // 16GB
    // 1LL * 1024 * 1024 * 8192,   // OOM
};

// const int SIZES_LEN = 26;
// const LL SIZES[SIZES_LEN] = {   // int = 4B
//     1LL * 256,                  // 1KB
//     1LL * 512,                  // 2KB
//     1LL * 1024 * 1,             // 4KB
//     1LL * 1024 * 2,             // 8KB
//     1LL * 1024 * 4,             // 16KB
//     1LL * 1024 * 8,             // 32KB
//     1LL * 1024 * 16,            // 64KB
//     1LL * 1024 * 32,            // 128KB
//     1LL * 1024 * 64,            // 256KB
//     1LL * 1024 * 128,           // 512KB
//     1LL * 1024 * 256,           // 1MB
//     1LL * 1024 * 512,           // 2MB
//     1LL * 1024 * 1024 * 1,      // 4MB
//     1LL * 1024 * 1024 * 2,      // 8MB
//     1LL * 1024 * 1024 * 4,      // 16MB
//     1LL * 1024 * 1024 * 8,      // 32MB
//     1LL * 1024 * 1024 * 16,     // 64MB
//     1LL * 1024 * 1024 * 32,     // 128MB
//     1LL * 1024 * 1024 * 64,     // 256MB
//     1LL * 1024 * 1024 * 128,    // 512MB
//     1LL * 1024 * 1024 * 256,    // 1GB
//     1LL * 1024 * 1024 * 512,    // 2GB
//     1LL * 1024 * 1024 * 1024,   // 4GB
//     1LL * 1024 * 1024 * 2048,   // 8GB
//     1LL * 1024 * 1024 * 4096,   // 16GB
//     1LL * 1024 * 1024 * 8192,   // OOM
// };

// const int SIZES_LEN = 18;
// const LL SIZES[SIZES_LEN] = {           // int = 4B
//     (LL)MAGIC_FACTOR * 1,               // 590.6KB
//     (LL)MAGIC_FACTOR * 2,               
//     (LL)MAGIC_FACTOR * 4,           
//     (LL)MAGIC_FACTOR * 8,            
//     (LL)MAGIC_FACTOR * 16,             
//     (LL)MAGIC_FACTOR * 32,
//     (LL)MAGIC_FACTOR * 64,
//     (LL)MAGIC_FACTOR * 128,
//     (LL)MAGIC_FACTOR * 256,
//     (LL)MAGIC_FACTOR * 512,
//     (LL)MAGIC_FACTOR * 1024,
//     (LL)MAGIC_FACTOR * 1024 * 2,
//     (LL)MAGIC_FACTOR * 1024 * 4,
//     (LL)MAGIC_FACTOR * 1024 * 8,
//     (LL)MAGIC_FACTOR * 1024 * 16,
//     (LL)MAGIC_FACTOR * 1024 * 32,       // 18.46GB
//     (LL)MAGIC_FACTOR * 1024 * 64,       // 36.91GB
//     (LL)MAGIC_FACTOR * 1024 * 128,      // 73.82GB
// };

bool check_pattern(Json::Value pattern, int N_GPUs) {
    for (int k = 0; k < pattern.size(); ++ k) {
        if (std::max(pattern[k][0].asInt(), pattern[k][1].asInt()) >= N_GPUs) {
            return false;
        }
    }
    return true;
}

void devicesSyncAll(int N_GPUs) {
    for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
        CUDA_CHECK(cudaSetDevice(gpuid));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void barrier(std::string& BACKEND, int N_GPUs) {
    if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
        for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
            CUDA_CHECK(cudaSetDevice(gpuid));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    if (BACKEND.compare("NCCL") == 0 || BACKEND.compare("MPI") == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
}

// cudaMemcpy_comm
void cudaMemcpy_comm(int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    for (int i = 0; i < nranks; ++ i) {
        for (int j = 0; j < nranks; ++ j) {
            CUDA_CHECK(cudaMemcpyPeerAsync(recv_buf[j], j, \
                                   send_buf[i], i, \
                                   SIZE * sizeof(int), streams[i * nranks + j])); 
        }
    }
}

// NCCL_comm
void NCCL_comm(int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    // printf("rank: %d", rank);
    // Json::FastWriter sw;
    // std::cout << sw.write(pairs);
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < nranks; ++ i) {
        NCCL_CHECK(ncclSend(send_buf[rank], SIZE, ncclInt32, i, comm, streams[0]));
        NCCL_CHECK(ncclRecv(recv_buf[rank], SIZE, ncclInt32, i, comm, streams[0]));
    }
    NCCL_CHECK(ncclGroupEnd());
}

void MPI_comm(int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    int req_num = 0;
    for (int i = 0; i < nranks; ++ i) {
        MPI_Isend(send_buf[rank], SIZE, MPI_INT, i, 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
        MPI_Irecv(recv_buf[rank], SIZE, MPI_INT, i, 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
    }
    MPI_Waitall(req_num, mpi_request , nullptr);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Need at least 3 args: \"<command> <gpus> <backend>\"\n");
        return - 1;
    }
    setup_env(pp, argc, argv);

    void (*XXX_comm)(int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request);
    if (pp->BACKEND.compare("NCCL") == 0) {
        XXX_comm = NCCL_comm;
    } else if (pp->BACKEND.compare("MPI") == 0) {
        XXX_comm = MPI_comm;
    } else if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
        XXX_comm = cudaMemcpy_comm;
    } else {
        printf("Error BACKEND !!!");
        return - 1;
    }

    // Init cudaStream
    pp->init_cudaStream(pp->N_GPUs * pp->N_GPUs);

    // Init MPI_Request
    pp->init_MPI_Request(pp->N_GPUs * pp->N_GPUs);

    if (pp->BACKEND.find("cudaMemcpy") != std::string::npos && pp->ENABLE_GPU_P2P) {
        enableP2P(pp->N_GPUs);
    }

    for (int i = SIZEIDX_START; i < SIZEIDX_END; ++ i) {
        LL SIZE = SIZES[i];
        int** send_buf = new int*[pp->N_GPUs];
        int** recv_buf = new int*[pp->N_GPUs];
        if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
            for (int gpuid = 0; gpuid < pp->N_GPUs; ++ gpuid) {
                CUDA_CHECK(cudaSetDevice(gpuid));
                CUDA_CHECK(cudaMalloc(&send_buf[gpuid], SIZE * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&recv_buf[gpuid], SIZE * sizeof(int)));
            }
        }
        if (pp->BACKEND.compare("NCCL") == 0 || pp->BACKEND.compare("MPI") == 0) {
            CUDA_CHECK(cudaMalloc(&send_buf[pp->rank], SIZE * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&recv_buf[pp->rank], SIZE * sizeof(int)));
        }

        // WARMUP
        for (int _ = 0; _ < WARMUP; ++ _) {
            XXX_comm(send_buf, recv_buf, SIZE, pp->streams, pp->rank, pp->N_GPUs, pp->comm, pp->mpi_requests);
            barrier(pp->BACKEND, pp->N_GPUs);
            // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
        }
        barrier(pp->BACKEND, pp->N_GPUs);

        // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int _ = 0; _ < TIMES; ++ _) {
            XXX_comm(send_buf, recv_buf, SIZE, pp->streams, pp->rank, pp->N_GPUs, pp->comm, pp->mpi_requests); 
            barrier(pp->BACKEND, pp->N_GPUs);
            // CUDA_CHECK(cudaDeviceSynchronize());    // light-barrier, [WHY]: 会有性能提升！！！ 减少 comm contention ?
            // MPI_Barrier(MPI_COMM_WORLD);            // cpu-barrier, 没有意义
            // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)            
        }
        barrier(pp->BACKEND, pp->N_GPUs);

        auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT
        // CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start_a2a, stop_a2a));    // ms
        // if (true) {
        if (pp->rank == 0) {
            // // double t_d = (double)elapsedTime / 1000;    // s
            double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
            double calc = pp->N_GPUs * (pp->N_GPUs - 1) * (double)SIZE * sizeof(int) * TIMES;      // B
            double avg_bd = calc / t_d;
            // printf(": time %lf s, REAL_BD %lf GB/s, SIZE %lf KB, comm_vol %lf KB\n", \
            //         t_d, avg_bd / pow(1024, 3) , (double)SIZE * sizeof(int) / pow(1024, 1), calc / pow(1024, 1));
            printf("time %lf s, REAL_BD %lf GB/s, SIZE %lf KB\n", \
                    t_d, avg_bd / pow(1024, 3) , (double)SIZE * sizeof(int) / pow(1024, 1));
            fflush(stdout);
        }


        if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
            for (int gpuid = 0; gpuid < pp->N_GPUs; ++ gpuid) {
                CUDA_CHECK(cudaFree(recv_buf[gpuid]));
                CUDA_CHECK(cudaFree(send_buf[gpuid]));
            }
        }
        if (pp->BACKEND.compare("NCCL") == 0 || pp->BACKEND.compare("MPI") == 0) {
            CUDA_CHECK(cudaFree(recv_buf[pp->rank]));
            CUDA_CHECK(cudaFree(send_buf[pp->rank]));
        }
        
        delete[] recv_buf;
        delete[] send_buf;
    }

    if (pp->BACKEND.find("cudaMemcpy") != std::string::npos && pp->ENABLE_GPU_P2P) {
        disableP2P(pp->N_GPUs);
    }
    // MPI_Finalize();
    return 0;
}