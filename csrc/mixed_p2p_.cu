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
// #include <format>    // need c++20
typedef long long LL;
int comm_size, rank;
int N_GPUs, GPU_VISIBLE;

PROC_PARAMS* pp;

// #define CHECK_RESULT
// #define PRINT_JSON
// #define RECORD_TABLE
// #define ENABLE_GPU_P2P       // 性能不一定好！！！ 单个P2P更好，但多个P2P不一定好

int TIMES = 1;
int WARMUP = 0;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZEIDX_START = 6;
const int SIZEIDX_END = 7;

const int SIZES_LEN = 8;
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
    // if (BACKEND.compare("cudaMemcpy") == 0) {
    //     for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
    //         CUDA_CHECK(cudaSetDevice(gpuid));
    //         CUDA_CHECK(cudaDeviceSynchronize());
    //     }
    // }
    if (BACKEND.compare("NCCL") == 0 || BACKEND.compare("MPI") == 0) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
}

// cudaMemcpy_comm
void cudaMemcpy_comm(Json::Value& pairs, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request) {
    for (int k = 0; k < pairs.size(); ++ k) {
        // CUDA_CHECK(cudaMemcpyAsync(recv_buf[pairs[k][1].asInt()], send_buf[pairs[k][0].asInt()], 
        //                             SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[k]));
        CUDA_CHECK(cudaMemcpyPeerAsync(recv_buf[pairs[k][1].asInt()], pairs[k][1].asInt(), \
                                   send_buf[pairs[k][0].asInt()], pairs[k][0].asInt(), \
                                   SIZE * sizeof(int), streams[k]));                                // 两者性能相似
    }
}

// NCCL_comm
void NCCL_comm(Json::Value& pairs, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request) {
    // printf("rank: %d", rank);
    // Json::FastWriter sw;
    // std::cout << sw.write(pairs);
    NCCL_CHECK(ncclGroupStart());
    for (int k = 0; k < pairs.size(); ++ k) {
        if (rank == pairs[k][0].asInt()) {
            NCCL_CHECK(ncclSend(send_buf[rank], SIZE, ncclInt32, pairs[k][1].asInt(), comm, streams[0]));
        }
        if (rank == pairs[k][1].asInt()) {
            NCCL_CHECK(ncclRecv(recv_buf[rank], SIZE, ncclInt32, pairs[k][0].asInt(), comm, streams[0]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
}

// MPI_comm
void MPI_comm(Json::Value& pairs, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request) {
    int req_num = 0;
    for (int k = 0; k < pairs.size(); ++ k) {
        if (rank == pairs[k][0].asInt()) {
            MPI_Isend(send_buf[rank], SIZE, MPI_INT, pairs[k][1].asInt(), 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
        }
        if (rank == pairs[k][1].asInt()) {
            MPI_Irecv(recv_buf[rank], SIZE, MPI_INT, pairs[k][0].asInt(), 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
        }
    }
    // MPI_Wait(mpi_request, NULL);
    MPI_Waitall(req_num, mpi_request , nullptr);

}


void mixed_p2p_comm(Json::Value& pairs, int p2p_id, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request) {
    // printf("rank: %d", rank);
    // Json::FastWriter sw;
    // std::cout << sw.write(pairs);

    CUDA_CHECK(cudaSetDevice(rank));    // just for NCCL
    NCCL_CHECK(ncclGroupStart());
    for (int k = 0; k < pairs.size(); ++ k) {
        if (p2p_id / (int)pow(3, k) % 3 == 0) {     // NCCL
            if (rank == pairs[k][0].asInt()) {
                NCCL_CHECK(ncclSend(send_buf[rank], SIZE, ncclInt32, pairs[k][1].asInt(), comm, streams[0]));
            }
            if (rank == pairs[k][1].asInt()) {
                NCCL_CHECK(ncclRecv(recv_buf[rank], SIZE, ncclInt32, pairs[k][0].asInt(), comm, streams[0]));
            }
        } else {                                    // DMA
            if (rank != 0) {
                continue;
            }
            CUDA_CHECK(cudaMemcpyPeerAsync(recv_buf[pairs[k][1].asInt()], pairs[k][1].asInt(), \
                                   send_buf[pairs[k][0].asInt()], pairs[k][0].asInt(), \
                                   SIZE * sizeof(int), streams[k]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Need at least 2 args: \"<command> <gpus> <backend> <cp_file>\"\n");
        return - 1;
    }
    //Get number of gpus in the node
    CUDA_CHECK(cudaGetDeviceCount(&GPU_VISIBLE));
    N_GPUs = std::stoi(argv[1]);
    assert(N_GPUs <= GPU_VISIBLE);
    std::string BACKEND = argv[2];
    std::string cp_file = argv[3];

    pp = new PROC_PARAMS(N_GPUs);
    pp->BACKEND = BACKEND;

    setup_env(pp);

    // Read patterns
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
    int STREAM_NUM = std::max(N_GPUs, max_pair_num) + N_GPUs;       // for both DMA and NCCL
    cudaStream_t* streams = new cudaStream_t[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; ++ i) {
        cudaStreamCreate(&streams[i]);
    }

    // Init MPI_Request
    MPI_Request mpi_request[std::max(N_GPUs, max_pair_num)];        // for cuda-aware MPI

    for (int cp = 0; cp < root.size(); ++ cp) {
        if (! check_pattern(root[cp], N_GPUs)) {
            continue;
        }
        if (rank == 0) {
            // Json::StyledWriter sw;
            Json::FastWriter sw;
            std::cout << sw.write(root[cp]);
            fflush(stdout);
        }
        for (int i = SIZEIDX_START; i < SIZEIDX_END; ++ i) {
            LL SIZE = SIZES[i];
            int** send_buf = new int*[N_GPUs];
            int** recv_buf = new int*[N_GPUs];
            // if (BACKEND.compare("cudaMemcpy") == 0) {
            //     for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
            //         CUDA_CHECK(cudaSetDevice(gpuid));
            //         CUDA_CHECK(cudaMalloc(&send_buf[gpuid], SIZE * sizeof(int)));
            //         CUDA_CHECK(cudaMalloc(&recv_buf[gpuid], SIZE * sizeof(int)));
            //     }
            // }
            if (BACKEND.compare("NCCL") == 0 || BACKEND.compare("MPI") == 0) {
                if (rank == 0) {
                    for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
                        CUDA_CHECK(cudaSetDevice(gpuid));
                        CUDA_CHECK(cudaMalloc(&send_buf[gpuid], SIZE * sizeof(int)));
                        CUDA_CHECK(cudaMalloc(&recv_buf[gpuid], SIZE * sizeof(int)));
                    }
                } else {
                    CUDA_CHECK(cudaSetDevice(rank));
                    CUDA_CHECK(cudaMalloc(&send_buf[rank], SIZE * sizeof(int)));
                    CUDA_CHECK(cudaMalloc(&recv_buf[rank], SIZE * sizeof(int)));
                }
            }
            
            // cudaEvent_t start_a2a, stop_a2a;
            // float elapsedTime;
            // CUDA_CHECK(cudaEventCreate(&start_a2a));
            // CUDA_CHECK(cudaEventCreate(&stop_a2a));
            
            double max_BD = 0, min_BD = std::numeric_limits<double>::max();
            int MIXED_P2P_NUM = pow(3, root[cp].size());
            if (rank == 0) {
                printf("MIXED_P2P_NUM: %d\n", MIXED_P2P_NUM);
            }
            for (int p2p_id = 0; p2p_id < MIXED_P2P_NUM; ++ p2p_id) {
                bool has_0 = false;
                for (int k = 0; k < root[cp].size(); ++ k) {        // no NCCL
                    if (p2p_id / (int)pow(3, k) % 3 == 0) {
                        has_0 = true;
                        break;
                    }
                }
                if (has_0) {
                    continue;
                }

                // enableP2P
                if (rank == 0) {
                    for (int k = 0; k < root[cp].size(); ++ k) {
                        if (p2p_id / (int)pow(3, k) % 3 == 1) {
                            enableP2P(root[cp][k][0].asInt(), root[cp][k][1].asInt());
                        }
                    }
                }
                // WARMUP
                for (int _ = 0; _ < WARMUP; ++ _) {
                    mixed_p2p_comm(root[cp], p2p_id, send_buf, recv_buf, SIZE, streams, rank, pp->comm, mpi_request);
                    // CUDA_CHECK(cudaGetLastError());
                    barrier(BACKEND, N_GPUs);
                }

                // CUDA_CHECK(cudaDeviceSynchronize());
                // MPI_Barrier(MPI_COMM_WORLD);
                // devicesSyncAll(N_GPUs);
                barrier(BACKEND, N_GPUs);

                // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
                auto t0 = std::chrono::high_resolution_clock::now();

                for (int _ = 0; _ < TIMES; ++ _) {
                    mixed_p2p_comm(root[cp], p2p_id, send_buf, recv_buf, SIZE, streams, rank, pp->comm, mpi_request);
                    // CUDA_CHECK(cudaDeviceSynchronize());    // light-barrier, [WHY]: 会有性能提升！！！ 减少 comm contention ?
                    // MPI_Barrier(MPI_COMM_WORLD);            // cpu-barrier, 没有意义
                    // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
                    // barrier(BACKEND, N_GPUs);
                }
                barrier(BACKEND, N_GPUs);

                auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT
                // CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start_a2a, stop_a2a));    // ms
                // if (true) {
                if (rank == 0) {
                    // double t_d = (double)elapsedTime / 1000;    // s
                    double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
                    double calc = root[cp].size() * (double)SIZE * sizeof(int) * TIMES;      // B
                    double avg_bd = calc / t_d;
                    for (int k = 0; k < N_GPUs; ++ k) {
                        printf("%d", p2p_id / (int)pow(3, k) % 3);
                    }
                    max_BD = std::max(max_BD, avg_bd);
                    min_BD = std::min(min_BD, avg_bd);
                    printf(": time %lf s, REAL_BD %lf GB/s, SIZE %lf KB, max_BD: %lf GB/s\n", \
                            t_d, avg_bd / pow(1024, 3) , (double)SIZE * sizeof(int) / pow(1024, 1), max_BD / pow(1024, 3));
                    fflush(stdout);
                }

                // disableP2P
                if (rank == 0) {
                    for (int k = 0; k < root[cp].size(); ++ k) {
                        if (p2p_id / (int)pow(3, k) % 3 == 1) {
                            disableP2P(root[cp][k][0].asInt(), root[cp][k][1].asInt());
                        }
                    }
                }
            }


            // if (BACKEND.compare("cudaMemcpy") == 0) {
            //     for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
            //         CUDA_CHECK(cudaFree(recv_buf[gpuid]));
            //         CUDA_CHECK(cudaFree(send_buf[gpuid]));
            //     }
            // }
            if (BACKEND.compare("NCCL") == 0 || BACKEND.compare("MPI") == 0) {
                if (rank == 0) {
                    for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
                        CUDA_CHECK(cudaFree(recv_buf[gpuid]));
                        CUDA_CHECK(cudaFree(send_buf[gpuid]));
                    }
                } else {
                    CUDA_CHECK(cudaFree(recv_buf[rank]));
                    CUDA_CHECK(cudaFree(send_buf[rank]));
                }
            }
            
            delete[] recv_buf;
            delete[] send_buf;
        }
    }
    for (int i = 0; i < STREAM_NUM; ++ i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    // MPI_Finalize();
    return 0;
}