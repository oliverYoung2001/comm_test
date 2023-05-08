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
#include <algorithm>
#include <limits>
#include "utils.h"

std::vector<int*> ringid_arrays;

// #include <format>    // need c++20
typedef long long LL;

PROC_PARAMS* pp;

// #define CHECK_RESULT
// #define PRINT_JSON
// #define RECORD_TABLE
// #define ENABLE_GPU_P2P       // 性能不一定好！！！ 单个P2P更好，但多个P2P不一定好
int TIMES = 3;
int WARMUP = 1;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZEIDX_START = 6;
const int SIZEIDX_END = 7;

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
    // 1LL * 1024 * 1024 * 8,      // 32MB
    1LL * 1024 * 1024 * 16,      // 64MB
    1LL * 1024 * 1024 * 32,     // 128MB
    // 1LL * 1024 * 1024 * 64,     // 256MB
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

void cudaMemcpy_comm(int* ids, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    int k = 0;
    int ring_st = ids[0];
    int dst;
    int stream_id = 0;
    while (true) {
        if (ids[k] < 0) {
            if (ids[k] == - 2) {
                break;
            }
            ring_st = ids[++ k];
            continue;
        }
        dst = ids[k + 1] < 0 ? ring_st : ids[k + 1];
        CUDA_CHECK(cudaMemcpyPeerAsync(recv_buf[dst], dst, \
                                   send_buf[ids[k]], ids[k], \
                                   SIZE * sizeof(int), streams[stream_id ++]));
                                //    SIZE * sizeof(int), streams[k]));
        ++ k;
    }
}

void NCCL_comm(int* ids, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    int k = 0;
    int ring_st = ids[0];
    int src, dst;
    NCCL_CHECK(ncclGroupStart());
    int stream_id = 0;
    while (true) {
        if (ids[k] < 0) {
            if (ids[k] == - 2) {
                break;
            }
            ring_st = ids[++ k];
            continue;
        }
        src = ids[k];
        dst = ids[k + 1] < 0 ? ring_st : ids[k + 1];
        if (rank == src) {
            NCCL_CHECK(ncclSend(send_buf[rank], SIZE, ncclInt32, dst, comm, streams[stream_id ++]));    // 用不同的stream效果相同
        }
        if (rank == dst) {
            NCCL_CHECK(ncclRecv(recv_buf[rank], SIZE, ncclInt32, src, comm, streams[stream_id ++]));
        }
        ++ k;
    }
    NCCL_CHECK(ncclGroupEnd());
}

// MPI_comm
void MPI_comm(int* ids, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    int k = 0;
    int ring_st = ids[0];
    int src, dst;
    int req_num = 0;
    while (true) {
        if (ids[k] < 0) {
            if (ids[k] == - 2) {
                break;
            }
            ring_st = ids[++ k];
            continue;
        }
        src = ids[k];
        dst = ids[k + 1] < 0 ? ring_st : ids[k + 1];
        if (rank == src) {
            MPI_Isend(send_buf[rank], SIZE, MPI_INT, dst, 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
        }
        if (rank == dst) {
            MPI_Irecv(recv_buf[rank], SIZE, MPI_INT, src, 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
        }
        ++ k;
    }
    MPI_Waitall(req_num, mpi_request , nullptr);

    // int req_num = 0;
    // for (int k = 0; k < pairs.size(); ++ k) {
    //     if (rank == pairs[k][0].asInt()) {
    //         MPI_Isend(send_buf[rank], SIZE, MPI_INT, pairs[k][1].asInt(), 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
    //     }
    //     if (rank == pairs[k][1].asInt()) {
    //         MPI_Irecv(recv_buf[rank], SIZE, MPI_INT, pairs[k][0].asInt(), 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
    //     }
    // }
    // MPI_Waitall(req_num, mpi_request , nullptr);
}

void dfs(int turn, int id, int* tmp_ids, bool* P) {
    // printf("turn: %d, id: %d\n", turn, id);
    if (id >= pp->N_GPUs) {
        // ringid_arrays
        int* new_ids = new int[pp->N_GPUs + turn + 1];
        memcpy(new_ids, tmp_ids, (pp->N_GPUs + turn) * sizeof(int));
        new_ids[pp->N_GPUs + turn] = - 2;
        ringid_arrays.push_back(new_ids);
        // for (int i = 0; i < N_GPUs + turn + 1; ++ i) {
        //     printf("%d ", new_ids[i]);
        // }
        // puts("");
        return;
    }
    for (int i = 0; i < pp->N_GPUs; ++ i) {
        if (! P[i]) {
            tmp_ids[id + turn] = i;
            P[i] = 1;
            // old ring
            dfs(turn, id + 1, tmp_ids, P);

            // new ring
            if (tmp_ids[id + turn - 1] >= 0 && pp->N_GPUs - (id + 1) >= 2) {      // size of each ring >= 2
                tmp_ids[id + turn + 1] = - 1;
                for (int j = 0; j < pp->N_GPUs; ++ j) {         // smallist
                    if (! P[j]) {
                        tmp_ids[id + 1 + turn + 1] = j;
                        P[j] = 1;
                        dfs(turn + 1, id + 2, tmp_ids, P);
                        P[j] = 0;
                        break;
                    }
                }
            }
            P[i] = 0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Need at least 2 args: \"<command> <gpus> <backend>\"\n");
        return - 1;
    }
    setup_env(pp, argc, argv);

    void (*XXX_comm)(int* ring_ids, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request);
    if (pp->BACKEND.compare("NCCL") == 0) {
        XXX_comm = NCCL_comm;
    } else if (pp->BACKEND.compare("MPI") == 0) {
        XXX_comm = MPI_comm;
    } else if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
        XXX_comm = cudaMemcpy_comm;
    } else {
        printf("Error BACKEND !!!");
        exit(- 1);
    }

    // Init cudaStream
    pp->init_cudaStream(pp->N_GPUs * pp->N_GPUs);

    // Init MPI_Request
    pp->init_MPI_Request(pp->N_GPUs);

    if (pp->BACKEND.find("cudaMemcpy") != std::string::npos && pp->ENABLE_GPU_P2P) {
        enableP2P(pp->N_GPUs);
    }

    LL SIZE = SIZES[SIZEIDX_START];
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

    bool P[pp->N_GPUs];
    memset(P, 0, sizeof(P));

    int tmp_ids[pp->N_GPUs << 1];
    P[0] = 1;
    tmp_ids[0] = 0;
    dfs(0, 1, tmp_ids, P);
    P[0] = 0;
    if (pp->rank == 0) {
        printf("ringid_arrays.size(): %d\n", ringid_arrays.size());
        fflush(stdout);
    }
    double max_BD = 0, min_BD = std::numeric_limits<double>::max();
    for (auto ring_ids : ringid_arrays) {
        // WARMUP
        for (int _ = 0; _ < WARMUP; ++ _) {
            XXX_comm(ring_ids, send_buf, recv_buf, SIZE, pp->streams, pp->rank, pp->N_GPUs, pp->comm, pp->mpi_requests);
            barrier(pp->BACKEND, pp->N_GPUs);
            // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
        }
        barrier(pp->BACKEND, pp->N_GPUs);

        // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int _ = 0; _ < TIMES; ++ _) {
            XXX_comm(ring_ids, send_buf, recv_buf, SIZE, pp->streams, pp->rank, pp->N_GPUs, pp->comm, pp->mpi_requests);
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
            double calc = pp->N_GPUs * (double)SIZE * sizeof(int) * TIMES;      // B
            double avg_bd = calc / t_d;
            for (int k = 0; ; ++ k) {
                if (ring_ids[k] == - 1) {
                    putchar('|');
                } else if (ring_ids[k] == - 2) {
                    break;
                } else {
                    printf("%d", ring_ids[k]);
                }
            }
            max_BD = std::max(max_BD, avg_bd);
            min_BD = std::min(min_BD, avg_bd);
            printf(": time %lf s, REAL_BD %lf GB/s, SIZE %lf KB, max_BD %lf GB/s\n", \
                    t_d, avg_bd / pow(1024, 3) , (double)SIZE * sizeof(int) / pow(1024, 1), max_BD / pow(1024, 3));
            fflush(stdout);
        }
    }
    // } while (std::next_permutation(ring_ids + 1, ring_ids + N_GPUs));


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

    if (pp->BACKEND.find("cudaMemcpy") != std::string::npos && pp->ENABLE_GPU_P2P) {
        disableP2P(pp->N_GPUs);
    }
    delete pp;
    // MPI_Finalize();
    return 0;
}