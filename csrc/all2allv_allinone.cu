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

// #include <format>    // need c++20
typedef long long LL;

PROC_PARAMS* pp;

// #define CHECK_RESULT
// #define PRINT_JSON
int TIMES = 2000;
int WARMUP = 10;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZEIDX_START = 4;
const int SIZEIDX_END = 20;

// const int SIZES_LEN = 8;
// const LL SIZES[SIZES_LEN] = {   // int = 4B
//     1LL * 256,                  // 1KB      // 打不满带宽
//     1LL * 1024 * 1,             // 4KB      // 打不满带宽
//     1LL * 1024 * 2,             // 8KB     // 会高一些!!! (仅在某些情况下)
//     1LL * 1024 * 4,             // 16KB     // 会高一些!!!  （最好）
//     // 1LL * 1024 * 8,             // 32KB     // 会高一些!!!  （最好）
//     1LL * 1024 * 16,            // 64KB     // 会高一些!!!  （最好）
//     // 1LL * 1024 * 64,            // 256KB    // 趋于稳定
//     1LL * 1024 * 256,           // 1MB
//     // 1LL * 1024 * 1024 * 1,      // 4MB      // 打不满带宽
//     1LL * 1024 * 1024 * 32,     // 128MB
//     // 1LL * 1024 * 1024 * 64,     // 256MB
//     1LL * 1024 * 1024 * 128,    // 512MB
//     // 1LL * 1024 * 1024 * 256,    // 1GB
//     // 1LL * 1024 * 1024 * 512,    // 2GB
//     // 1LL * 1024 * 1024 * 1024,   // 4GB      // 用cudaMemcpy，竟然有性能下降！！！
//     // 1LL * 1024 * 1024 * 2048,   // 8GB
//     // 1LL * 1024 * 1024 * 4096,   // 16GB
//     // 1LL * 1024 * 1024 * 8192,   // OOM
// };

const int SIZES_LEN = 26;
const LL SIZES[SIZES_LEN] = {   // int = 4B
    1LL * 256,                  // 1KB
    1LL * 512,                  // 2KB
    1LL * 1024 * 1,             // 4KB
    1LL * 1024 * 2,             // 8KB
    1LL * 1024 * 4,             // 16KB
    1LL * 1024 * 8,             // 32KB
    1LL * 1024 * 16,            // 64KB
    1LL * 1024 * 32,            // 128KB
    1LL * 1024 * 64,            // 256KB
    1LL * 1024 * 128,           // 512KB
    1LL * 1024 * 256,           // 1MB
    1LL * 1024 * 512,           // 2MB
    1LL * 1024 * 1024 * 1,      // 4MB
    1LL * 1024 * 1024 * 2,      // 8MB
    1LL * 1024 * 1024 * 4,      // 16MB
    1LL * 1024 * 1024 * 8,      // 32MB
    1LL * 1024 * 1024 * 16,     // 64MB
    1LL * 1024 * 1024 * 32,     // 128MB
    1LL * 1024 * 1024 * 64,     // 256MB
    1LL * 1024 * 1024 * 128,    // 512MB
    1LL * 1024 * 1024 * 256,    // 1GB
    1LL * 1024 * 1024 * 512,    // 2GB
    1LL * 1024 * 1024 * 1024,   // 4GB
    1LL * 1024 * 1024 * 2048,   // 8GB
    1LL * 1024 * 1024 * 4096,   // 16GB
    1LL * 1024 * 1024 * 8192,   // OOM
};

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

// void devicesSyncAll(int N_GPUs) {
//     for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
//         CUDA_CHECK(cudaSetDevice(gpuid));
//         CUDA_CHECK(cudaDeviceSynchronize());
//     }
// }

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

void MPI_v(int** sendcounts, int** recvcounts, int** sdispls, int** rdispls, int** send_buf, int** recv_buf, \
              cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request) {
    MPI_Alltoallv(send_buf[rank], sendcounts[rank], sdispls[rank], MPI_BYTE, \
                  recv_buf[rank], recvcounts[rank], rdispls[rank], MPI_BYTE, MPI_COMM_WORLD);
}

void NCCL_v(int** sendcounts, int** recvcounts, int** sdispls, int** rdispls, int** send_buf, int** recv_buf, \
              cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request) {
    NCCL_CHECK(ncclGroupStart());
    for (int k = 0; k < pp->N_GPUs; ++ k) {
        NCCL_CHECK(ncclSend((char*)send_buf[rank] + sdispls[rank][k], sendcounts[rank][k], ncclChar, k, comm, streams[0]));
        NCCL_CHECK(ncclRecv((char*)recv_buf[rank] + rdispls[rank][k], recvcounts[rank][k], ncclChar, k, comm, streams[0]));
    }
    NCCL_CHECK(ncclGroupEnd());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Need at least 4 args: \"<command> <gpus> <backend> <cp_file>\"\n");
        return - 1;
    }
    setup_env(pp, argc, argv);
    std::string cp_file = argv[3];

    void (*XXX_v)(int** sendcounts, int** recvcounts, int** sdispls, int** rdispls, int** send_buf, int** recv_buf, \
              cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request);
    if (pp->BACKEND.compare("NCCL") == 0) {
        XXX_v = NCCL_v;
    } else if (pp->BACKEND.compare("MPI") == 0) {
        XXX_v = MPI_v;
    } else if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
        // XXX_comm = cudaMemcpy_comm;
    } else {
        printf("Error BACKEND !!!");
        exit(- 1);
    }

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

    // Init cudaStream
    int max_pair_num = 0;
    for (int cp = 0; cp < root.size(); ++ cp) {
        max_pair_num = std::max(max_pair_num, (int)root[cp].size());
    }
    pp->init_cudaStream(std::max(pp->N_GPUs, max_pair_num));

    // Init MPI_Request
    pp->init_MPI_Request(std::max(pp->N_GPUs, max_pair_num));


    // check_UVA(N_GPUs);        // 我理解，统一内存编址是为了方便，而不是性能

    int** sendcounts = new int*[pp->N_GPUs];
    int** recvcounts = new int*[pp->N_GPUs];
    int** sdispls = new int*[pp->N_GPUs];
    int** rdispls = new int*[pp->N_GPUs];
    for (int i = 0; i < pp->N_GPUs; ++ i) {
        sendcounts[i] = new int[pp->N_GPUs];
        recvcounts[i] = new int[pp->N_GPUs];
        sdispls[i] = new int[pp->N_GPUs + 1];
        rdispls[i] = new int[pp->N_GPUs + 1];
    }

    for (int cp = 0; cp < root.size(); ++ cp) {
        if (pp->rank == 0) {
            // Json::StyledWriter sw;
            Json::FastWriter sw;
            std::cout << sw.write(root[cp]);
            fflush(stdout);
        }
        if (pp->BACKEND.find("cudaMemcpy") != std::string::npos && pp->ENABLE_GPU_P2P) {
            printf("[ERROR]: No Implementation !!!");
            exit(- 1);
            // enableP2P(root[cp]);
        }

        Json::Value commv = root[cp];
        int commv_sum = 0;
        for (int i = 0; i < pp->N_GPUs; ++ i) {
            for (int j = 0; j < pp->N_GPUs; ++ j) {
                // commv_max = std::max(commv_max, commv[i][j].asInt());
                commv_sum += commv[i][j].asInt();
            }
        }
        if (pp->rank == 0) {
            printf("commv_sum: %d\n", commv_sum);
        }

        // for (int __ = SIZEIDX_START; __ < SIZEIDX_END; ++ __) {

            for (int i = 0; i < pp->N_GPUs; ++ i) {
                sdispls[i][0] = 0;
                rdispls[i][0] = 0;
                for (int j = 0; j < pp->N_GPUs; ++ j) {
                    sendcounts[i][j] = commv[i][j].asInt();
                    recvcounts[i][j] = commv[j][i].asInt();
                    sdispls[i][j + 1] = sdispls[i][j] + sendcounts[i][j];
                    rdispls[i][j + 1] = rdispls[i][j] + recvcounts[i][j];
                }
                // if (pp->rank == 0) {
                //     printf("%d %d\n", sdispls[i][pp->N_GPUs], rdispls[i][pp->N_GPUs]);
                // }
            }

            int** send_buf = new int*[pp->N_GPUs];
            int** recv_buf = new int*[pp->N_GPUs];
            if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
                printf("[ERROR]: No Implementation !!!");
                exit(- 1);
                // for (int gpuid = 0; gpuid < pp->N_GPUs; ++ gpuid) {
                //     CUDA_CHECK(cudaSetDevice(gpuid));
                //     CUDA_CHECK(cudaMalloc(&send_buf[gpuid], SIZE * sizeof(int)));
                //     CUDA_CHECK(cudaMalloc(&recv_buf[gpuid], SIZE * sizeof(int)));
                // }
            }
            if (pp->BACKEND.compare("NCCL") == 0 || pp->BACKEND.compare("MPI") == 0) {
                CUDA_CHECK(cudaMalloc(&send_buf[pp->rank], sdispls[pp->rank][pp->N_GPUs]));
                CUDA_CHECK(cudaMalloc(&recv_buf[pp->rank], rdispls[pp->rank][pp->N_GPUs]));
            }

            // WARMUP
            for (int _ = 0; _ < WARMUP; ++ _) {
                XXX_v(sendcounts, recvcounts, sdispls, rdispls, \
                        send_buf, recv_buf, pp->streams, pp->rank, pp->comm, pp->mpi_requests);
                barrier(pp->BACKEND, pp->N_GPUs);
            }

            barrier(pp->BACKEND, pp->N_GPUs);

            auto t0 = std::chrono::high_resolution_clock::now();

            for (int _ = 0; _ < TIMES; ++ _) {
                XXX_v(sendcounts, recvcounts, sdispls, rdispls, \
                        send_buf, recv_buf, pp->streams, pp->rank, pp->comm, pp->mpi_requests);
                barrier(pp->BACKEND, pp->N_GPUs);
            }
            barrier(pp->BACKEND, pp->N_GPUs);

            auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT
            // CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start_a2a, stop_a2a));    // ms
            // if (true) {
            if (pp->rank == 0) {
                // double t_d = (double)elapsedTime / 1000;    // s
                double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
                double calc = 1 * (double)commv_sum * 1 * TIMES;      // B
                double avg_bd = calc / t_d / pow(1024, 3);
                printf("time %lf s, REAL_BD %lf GB/s, SIZE %lf KB, comm_vol %lf KB\n", \
                        t_d, avg_bd, (double)commv_sum * 1 / pow(1024, 1), calc / pow(1024, 1));
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
            printf("[ERROR]: No Implementation !!!");
            exit(- 1);
            // disableP2P(root[cp]);
        }
    // }

    for (int i = 0; i < pp->N_GPUs; ++ i) {
        delete[] sendcounts[i];
        delete[] recvcounts[i];
        delete[] sdispls[i];
        delete[] rdispls[i];
    }
    delete[] sendcounts;
    delete[] recvcounts;
    delete[] sdispls;
    delete[] rdispls;
    
    delete pp;
    // MPI_Finalize();
    return 0;
}