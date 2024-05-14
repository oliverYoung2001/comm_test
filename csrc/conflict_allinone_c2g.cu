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

const int SIZEIDX_START = 5;
const int SIZEIDX_END = 9;

const int SIZES_LEN = 9;
const LL SIZES[SIZES_LEN] = {   // int = 4B
    1LL * 256,                  // 1KB      // 打不满带宽
    1LL * 1024 * 1,             // 4KB      // 打不满带宽
    1LL * 1024 * 2,             // 8KB     // 会高一些!!! (仅在某些情况下)
    1LL * 1024 * 4,             // 16KB     // 会高一些!!!  （最好）
    // 1LL * 1024 * 8,             // 32KB     // 会高一些!!!  （最好）
    1LL * 1024 * 16,            // 64KB     // 会高一些!!!  （最好）
    // 1LL * 1024 * 64,            // 256KB    // 趋于稳定
    1LL * 1024 * 256,           // 1MB
    // 1LL * 1024 * 1024 * 1,      // 4MB      // 打不满带宽
    1LL * 1024 * 1024 * 32,     // 128MB
    // 1LL * 1024 * 1024 * 64,     // 256MB
    1LL * 1024 * 1024 * 128,    // 512MB
    // 1LL * 1024 * 1024 * 256,    // 1GB
    1LL * 1024 * 1024 * 512,    // 2GB
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

bool check_pattern(Json::Value pattern, int N_GPUs) {
    for (int k = 0; k < pattern.size(); ++ k) {
        if (pattern[k].asInt() >= N_GPUs) {
            return false;
        }
    }
    return true;
}

// void devicesSyncAll(int N_GPUs) {
//     for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
//         CUDA_CHECK(cudaSetDevice(gpuid));
//         CUDA_CHECK(cudaDeviceSynchronize());
//     }
// }

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Need at least 5 args: \"<command> <gpus> <backend> <cp_file> <unidirection/bidirection(1/2)>\"\n");
        return - 1;
    }
    setup_env(pp, argc, argv);
    std::string cp_file = argv[3];
    std::string dir_mode = argv[4];
    // 0 stands for c2g
    // 1 stands for g2c
    // 2 stands for c2g + g2c
    assert(dir_mode == "0" || dir_mode == "1" || dir_mode == "2");

    // double result_table_si[pp->N_GPUs][pp->N_GPUs];
    // double result_table_bi[pp->N_GPUs][pp->N_GPUs];
    // memset(result_table_si, 0, sizeof(result_table_si));
    // memset(result_table_bi, 0, sizeof(result_table_bi));

    Json::Reader reader;
	Json::Value root;
    // std::string cp_file = "csrc/configs/conflict_patterns.json";
    std::ifstream in(cp_file.c_str(), std::ios::binary);
    if (! in.is_open()) {
		std::cout << "Error OPENING FILE\n";
		return - 1;
	}
    if (! reader.parse(in, root)) {
        std::cout << "Error READING FILE\n";
		return - 2;
    }

    // // Init cudaStream
    // int max_pair_num = 0;
    // for (int cp = 0; cp < root.size(); ++ cp) {
    //     max_pair_num = std::max(max_pair_num, (int)root[cp].size());
    // }
    // pp->init_cudaStream(std::max(pp->N_GPUs, max_pair_num));

    // // Init MPI_Request
    // pp->init_MPI_Request(std::max(pp->N_GPUs, max_pair_num));


    // check_UVA(N_GPUs);        // 我理解，统一内存编址是为了方便，而不是性能
    // if (true) {
    for (int cp = 0; cp < root.size(); ++ cp) {
        if (! check_pattern(root[cp], pp->N_GPUs)) {
            continue;
        }
        if (pp->rank == 0) {
            // Json::StyledWriter sw;
            Json::FastWriter sw;
            std::cout << sw.write(root[cp]);
            fflush(stdout);
        }
        // create_comm_group_from_pattern(pp, root[cp]);
        bool is_rank_in_pattern = false;
        for (int k = 0; k < root[cp].size(); ++ k) {
            if (root[cp][k].asInt() == pp->rank) {
                is_rank_in_pattern = true;
            }
        }
        // printf("rank%d, is_rank_in_pattern: %d\n", pp->rank, is_rank_in_pattern);
        // fflush(stdout);

        for (int i = SIZEIDX_START; i < SIZEIDX_END; ++ i) {
            LL SIZE = SIZES[i];
            int** send_buf = new int*[pp->N_GPUs];
            int** recv_buf = new int*[pp->N_GPUs];
            int** send_buf_cpu = new int*[pp->N_GPUs];
            int** recv_buf_cpu = new int*[pp->N_GPUs];

            if (pp->BACKEND.compare("NCCL") == 0 || pp->BACKEND.compare("MPI") == 0) {
                CUDA_CHECK(cudaMalloc(&send_buf[pp->rank], SIZE * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&recv_buf[pp->rank], SIZE * sizeof(int)));
                send_buf_cpu[pp->rank] = (int*)malloc(SIZE * sizeof(int));
                recv_buf_cpu[pp->rank] = (int*)malloc(SIZE * sizeof(int));
            }
            
            // WARMUP
            for (int _ = 0; _ < WARMUP; ++ _) {
                if (is_rank_in_pattern) {
                    if (dir_mode == "0" || dir_mode == "2") {
                        CUDA_CHECK(cudaMemcpy(send_buf_cpu[pp->rank], recv_buf[pp->rank], SIZE * sizeof(int), cudaMemcpyDefault));
                    }
                    if (dir_mode == "1" || dir_mode == "2") {
                        CUDA_CHECK(cudaMemcpy(send_buf[pp->rank], recv_buf_cpu[pp->rank], SIZE * sizeof(int), cudaMemcpyDefault));
                    }
                }
                barrier(pp->BACKEND, pp->N_GPUs);
            }
            barrier(pp->BACKEND, pp->N_GPUs);

            // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int _ = 0; _ < TIMES; ++ _) {
                if (is_rank_in_pattern) {
                    if (dir_mode == "0" || dir_mode == "2") {
                        CUDA_CHECK(cudaMemcpy(send_buf_cpu[pp->rank], recv_buf[pp->rank], SIZE * sizeof(int), cudaMemcpyDefault));
                    }
                    if (dir_mode == "1" || dir_mode == "2") {
                        CUDA_CHECK(cudaMemcpy(send_buf[pp->rank], recv_buf_cpu[pp->rank], SIZE * sizeof(int), cudaMemcpyDefault));
                    }
                }
                barrier(pp->BACKEND, pp->N_GPUs);
            }
            barrier(pp->BACKEND, pp->N_GPUs);

            auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT
            if (pp->rank == 0) {
                // double t_d = (double)elapsedTime / 1000;    // s
                double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
                double calc = root[cp].size() * (double)SIZE * sizeof(int) * TIMES * (dir_mode == "2" ? 2 : 1);      // B
                double avg_bd = calc / t_d / pow(1024, 3);
                printf("time %lf s, REAL_BD %lf GB/s, SIZE %lf KB, comm_vol %lf KB\n", \
                        t_d, avg_bd, (double)SIZE * sizeof(int) / pow(1024, 1), calc / pow(1024, 1));
                fflush(stdout);
            }
            
            if (pp->BACKEND.compare("NCCL") == 0 || pp->BACKEND.compare("MPI") == 0) {
                CUDA_CHECK(cudaFree(recv_buf[pp->rank]));
                CUDA_CHECK(cudaFree(send_buf[pp->rank]));
                free(recv_buf_cpu[pp->rank]);
                free(send_buf_cpu[pp->rank]);
            }
            
            delete[] recv_buf;
            delete[] send_buf;
            delete[] recv_buf_cpu;
            delete[] send_buf_cpu;
        }
        
        ncclCommDestroy(pp->cur_comm);
    }

    delete pp;
    MPI_Finalize();
    return 0;
}