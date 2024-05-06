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
#include "utils.h"
#include "constant.h"

// #include <format>    // need c++20
typedef long long LL;

PROC_PARAMS* pp;

int TIMES = 3;
int WARMUP = 2;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZEIDX_START = 6;
const int SIZEIDX_END = 10;

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
    // 1LL * 1024 * 1024 * 1,      // 1MB      // 打不满带宽
    1LL * 1024 * 1024 * 16,     // 16MB
    // 1LL * 1024 * 1024 * 32,     // 32MB
    1LL * 1024 * 1024 * 64,     // 64MB
    // 1LL * 1024 * 1024 * 128,    // 128MB
    1LL * 1024 * 1024 * 256,    // 256MB
    // 1LL * 1024 * 1024 * 512,    // 512MB
    1LL * 1024 * 1024 * 512,    // 1GB
    // 1LL * 1024 * 1024 * 1024,   // 4GB      // 用cudaMemcpy，竟然有性能下降！！！
    // 1LL * 1024 * 1024 * 2048,   // 8GB
    // 1LL * 1024 * 1024 * 4096,   // 16GB
    // 1LL * 1024 * 1024 * 8192,   // OOM
};

// const int SIZES_LEN = 27;
// const LL SIZES[SIZES_LEN] = {   // int = 4B
//     178956970 / 4,
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

std::pair<double, double> calc_bw_log(PROC_PARAMS*& pp, const char* op, LL size, double duration) {
    double n = pp->N_GPUs;
    double tput = 0, busbw = 0;
    if (strcmp(op, "ag") == 0 || strcmp(op, "rs") == 0) {
        size *= n;
        tput = (size / duration);
        busbw = (size / duration) * ((n - 1) / n);
    } else if (strcmp(op, "ar") == 0) {
        size *= n;
        tput = (size / duration) * 2;
        busbw = (size / duration) * (2 * (n - 1) / n);
    }
    tput = tput / 1000 / 1000 / 1000;
    busbw = busbw / 1000 / 1000 / 1000;
    return std::make_pair(tput, busbw);
}

void bench_op(PROC_PARAMS*& pp, const char* op, char* tensor, char* tensor_list, LL SIZE) {
    for (int _ = 0; _ < WARMUP; ++ _) {
        if (strcmp(op, "ag") == 0) {
            ncclAllGather(tensor, tensor_list, SIZE, ncclChar, pp->comm, pp->streams[0]);
        } else if (strcmp(op, "rs") == 0) {
            ncclReduceScatter(tensor_list, tensor, SIZE, ncclChar, ncclSum, pp->comm, pp->streams[0]);
        } else if (strcmp(op, "ar") == 0) {
            ncclAllReduce(tensor_list, tensor_list, SIZE * pp->N_GPUs, ncclChar, ncclSum, pp->comm, pp->streams[0]);
        }
        barrier(pp->BACKEND, pp->N_GPUs);
    }
    barrier(pp->BACKEND, pp->N_GPUs);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int _ = 0; _ < TIMES; ++ _) {
        if (strcmp(op, "ag") == 0) {
            ncclAllGather(tensor, tensor_list, SIZE, ncclChar, pp->comm, pp->streams[0]);
        } else if (strcmp(op, "rs") == 0) {
            ncclReduceScatter(tensor_list, tensor, SIZE, ncclChar, ncclSum, pp->comm, pp->streams[0]);
        } else if (strcmp(op, "ar") == 0) {
            ncclAllReduce(tensor_list, tensor_list, SIZE * pp->N_GPUs, ncclChar, ncclSum, pp->comm, pp->streams[0]);
        }
        barrier(pp->BACKEND, pp->N_GPUs);
    }
    barrier(pp->BACKEND, pp->N_GPUs);
    auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT

    if (pp->rank == 0) {
        // double t_d = (double)elapsedTime / 1000;    // s
        double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
        // double calc = root[cp].size() * (double)SIZE * sizeof(int) * TIMES;      // B
        // double avg_bd = calc / t_d / pow(1024, 3);
        auto bw = calc_bw_log(pp, op, SIZE, t_d / TIMES);   // GB/s
        printf("msg_size %lf KB, time %lf s, tput %lf GB/s, busbw %lf GB/s\n", \
               (double)SIZE / pow(1024, 1), t_d, bw.first, bw.second);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Need at least 3 args: \"<command> <gpus> <backend>\"\n");
        return - 1;
    }
    setup_env(pp, argc, argv);

    // Init cudaStream
    pp->init_cudaStream(pp->N_GPUs);

    // Init MPI_Request
    pp->init_MPI_Request(pp->N_GPUs);


    // check_UVA(N_GPUs);        // 我理解，统一内存编址是为了方便，而不是性能

    const int COMM_NUM = 3;
    std::string ops[COMM_NUM] = {"ag", "rs", "ar"};
    for (int comm_id = 0; comm_id < COMM_NUM; ++ comm_id) {
        char* op = (char*)ops[comm_id].c_str();
        if (pp->rank == 0) {
            printf("Comm_Op: %s\n", op);
        }
        for (int i = SIZEIDX_START; i < SIZEIDX_END; ++ i) {
            LL SIZE = SIZES[i];
            char* tensor;
            char* tensor_list;

            CUDA_CHECK(cudaMalloc(&tensor, SIZE));
            CUDA_CHECK(cudaMalloc(&tensor_list, SIZE * pp->N_GPUs));

            bench_op(pp, op, tensor, tensor_list, SIZE);
            
            CUDA_CHECK(cudaFree(tensor));
            CUDA_CHECK(cudaFree(tensor_list));
        }
    }

    delete pp;
    // MPI_Finalize();
    return 0;
}