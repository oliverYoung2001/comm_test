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
#include <pthread.h>

// #include <format>    // need c++20
typedef long long LL;

PROC_PARAMS* pp;

// #define CHECK_RESULT
// #define PRINT_JSON
// int TIMES = 200;
// int WARMUP = 10;
int TIMES = 1;
int WARMUP = 0;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZEIDX_START = 7;
const int SIZEIDX_END = 8;

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
        printf("Rank: %d, after sync !!!\n", pp->rank);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Rank: %d, after barrier !!!\n", pp->rank);
        fflush(stdout);
    }
}

// MPI_netflow(send_buf, recv_buf, SIZE, CHUNK_SIZE, flowpath, pp->streams, pp->rank, pp->comm, pp->mpi_requests);

struct MPI_line_args{
    int** send_buf;
    int** recv_buf;
    LL CHUNK_SIZE;
    int NUM;
    int src;
    int dst;
    MPI_Request* mpi_request;
};

void* MPI_line(void* threadargs) {
    sleep(1);
    cudaSetDevice(pp->rank);
    struct MPI_line_args* args = (struct MPI_line_args*)threadargs;
    int** send_buf = args->send_buf;
    int** recv_buf = args->recv_buf;
    LL CHUNK_SIZE = args->CHUNK_SIZE;
    int NUM = args->NUM;
    int src = args->src;
    int dst = args->dst;
    MPI_Request* mpi_request = args->mpi_request;
    // MPI_Request* mpi_request = new MPI_Request[1];
    // printf("(rank, src, dst, NUM, CHUNK_SIZE): (%d, %d, %d, %d, %lld)\n", pp->rank, src, dst, NUM, CHUNK_SIZE);
    // fflush(stdout);
    if (src < 0) {
        MPI_Isend(send_buf[pp->rank], CHUNK_SIZE, MPI_INT, dst, 
                             0/*tag*/, MPI_COMM_WORLD, mpi_request);
    } else if (dst < 0) {
        MPI_Irecv(send_buf[pp->rank], CHUNK_SIZE, MPI_INT, src, 
                             0/*tag*/, MPI_COMM_WORLD, mpi_request);
    } else {
        printf("Error !!!\n");
        exit(- 1);
    }
    int ret = MPI_Wait(mpi_request, nullptr);
    printf("Rank: %d, ret: %d\n", pp->rank, ret);
    fflush(stdout);
    // printf("rank: %d, done !!!\n", pp->rank);
    // fflush(stdout);
    return 0;
}

void MPI_netflow(int** send_buf, int** recv_buf, LL SIZE, LL CHUNK_SIZE, std::vector<int>& flowpath, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request) {
    // 0->2, 0->7
    // struct MPI_line_args* ta = new MPI_line_args[2];
    struct MPI_line_args ta[2];
    for (int i = 0; i < 2; ++ i) {
        ta[i].send_buf = send_buf;
        ta[i].recv_buf = recv_buf;
        ta[i].CHUNK_SIZE = SIZE;
        ta[i].NUM = 1;
    }
    if (pp->rank == 0) {
        pthread_t tids[2];
        ta[0].src = - 1;
        ta[0].dst = 2;
        ta[0].mpi_request = mpi_request;
        ta[1].src = - 1;
        ta[1].dst = 7;
        ta[1].mpi_request = mpi_request + 1;
        pthread_create(&tids[0], NULL, MPI_line, (void*)&ta[0]);
        pthread_create(&tids[1], NULL, MPI_line, (void*)&ta[1]);
        pthread_join(tids[0], nullptr);
        pthread_join(tids[1], nullptr);
        printf("Rank: %d, tids: %lld, %lld\n", pp->rank, tids[0], tids[1]);
        fflush(stdout);
        // MPI_Isend(send_buf[rank], SIZE, MPI_INT, 2, 
        //                      0/*tag*/, MPI_COMM_WORLD, mpi_request);
        // MPI_Isend(send_buf[rank], SIZE, MPI_INT, 7, 
        //                      0/*tag*/, MPI_COMM_WORLD, mpi_request + 1);
    }
    if (pp->rank == 2) {
        pthread_t tids[1];
        ta[0].src = 0;
        ta[0].dst = - 1;
        ta[0].mpi_request = mpi_request;
        pthread_create(&tids[0], NULL, MPI_line, (void*)&ta[0]);
        pthread_join(tids[0], nullptr);
        printf("Rank: %d, tids: %lld\n", pp->rank, tids[0]);
        fflush(stdout);
        // MPI_Recv(send_buf[rank], SIZE, MPI_INT, 0, 
        //                      0/*tag*/, MPI_COMM_WORLD, nullptr);
    }
    if (pp->rank == 7) {
        pthread_t tids[1];
        ta[0].src = 0;
        ta[0].dst = - 1;
        ta[0].mpi_request = mpi_request;
        pthread_create(&tids[0], NULL, MPI_line, (void*)&ta[0]);
        pthread_join(tids[0], nullptr);
        printf("Rank: %d, tids: %lld\n", pp->rank, tids[0]);
        fflush(stdout);
        // MPI_Recv(send_buf[rank], SIZE, MPI_INT, 0, 
        //                      0/*tag*/, MPI_COMM_WORLD, nullptr);
    }
    // delete[] ta;
    cudaSetDevice(pp->rank);
    printf("Rank: %d, out MPI_netflow\n", pp->rank);
    fflush(stdout);
    return;
    
    // // 0->2, 0->7
    // if (pp->rank == 0) {
    //     MPI_Isend(send_buf[rank], SIZE, MPI_INT, 2, 
    //                          0/*tag*/, MPI_COMM_WORLD, mpi_request);
    //     MPI_Isend(send_buf[rank], SIZE, MPI_INT, 7, 
    //                          0/*tag*/, MPI_COMM_WORLD, mpi_request + 1);
    // }
    // if (pp->rank == 2) {
    //     MPI_Recv(send_buf[rank], SIZE, MPI_INT, 0, 
    //                          0/*tag*/, MPI_COMM_WORLD, nullptr);
    // }
    // if (pp->rank == 7) {
    //     MPI_Recv(send_buf[rank], SIZE, MPI_INT, 0, 
    //                          0/*tag*/, MPI_COMM_WORLD, nullptr);
    // }
    // return;

    // 2->0, 7->0
    // MPI_Waitall(2, mpi_request, nullptr);    // Error
    // if (rank == 0) {
    //     MPI_Irecv(send_buf[rank], SIZE, MPI_INT, 2, 
    //                          0/*tag*/, MPI_COMM_WORLD, mpi_request);
    //     MPI_Irecv(send_buf[rank], SIZE, MPI_INT, 7, 
    //                          0/*tag*/, MPI_COMM_WORLD, mpi_request + 1);
    //     MPI_Waitall(2, mpi_request, nullptr);
    // }
    // if (rank == 2) {
    //     MPI_Send(send_buf[rank], SIZE, MPI_INT, 0, 
    //                          0/*tag*/, MPI_COMM_WORLD);
    // }
    // if (rank == 7) {
    //     MPI_Send(send_buf[rank], SIZE, MPI_INT, 0, 
    //                          0/*tag*/, MPI_COMM_WORLD);
    // }
    // return;
    
    // CHUNK_SIZE = SIZE;
    // if (rank == flowpath[0]) {
    //     MPI_Send(send_buf[rank], CHUNK_SIZE, MPI_INT, flowpath[1], 
    //                          0/*tag*/, MPI_COMM_WORLD);
    // }
    // if (rank == flowpath[1]) {
    //     MPI_Sendrecv(send_buf[rank], CHUNK_SIZE, MPI_INT, flowpath[2], 0, \
    //                  send_buf[rank], CHUNK_SIZE, MPI_INT, flowpath[0], 0, \
    //                  MPI_COMM_WORLD, nullptr);
    // }
    // if (rank == flowpath[2]) {
    //     MPI_Recv(send_buf[rank], CHUNK_SIZE, MPI_INT, flowpath[1], 
    //                          0/*tag*/, MPI_COMM_WORLD, nullptr);
    // }
    // return;

    // if (rank == flowpath[0]) {
    //     MPI_Send(send_buf[rank], CHUNK_SIZE, MPI_INT, flowpath[1], 
    //                          0/*tag*/, MPI_COMM_WORLD);
    // }
    // if (rank == flowpath[1]) {
    //     MPI_Recv(send_buf[rank], CHUNK_SIZE, MPI_INT, flowpath[0], 
    //                          0/*tag*/, MPI_COMM_WORLD, nullptr);
    // }

    // return;

    // int NUM = SIZE / CHUNK_SIZE;


    return;

    int NUM = SIZE / CHUNK_SIZE;
    if (NUM == 1) {
        for (int i = 0; i < flowpath.size(); ++ i) {
            if (rank == flowpath[i]) {
                if (i == 0) {
                    MPI_Send(send_buf[rank] + 0 * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i + 1], 
                             0/*tag*/, MPI_COMM_WORLD);
                } else if (i + 1 == flowpath.size()) {
                    MPI_Recv(send_buf[rank] + 0 * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i - 1], 
                             0/*tag*/, MPI_COMM_WORLD, nullptr);
                } else {
                    MPI_Recv(send_buf[rank] + 0 * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i - 1], 
                             0/*tag*/, MPI_COMM_WORLD, nullptr);
                    MPI_Send(send_buf[rank] + 0 * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i + 1], 
                             0/*tag*/, MPI_COMM_WORLD);
                }
            }
        }
        return;
    }
    
    for (int n = 0; n < NUM; ++ n) {
        for (int i = 0; i < flowpath.size(); ++ i) {
            if (rank == flowpath[i]) {
                // printf("%d, %d: in\n", rank, n);
                // fflush(stdout);
                if (i == 0) {
                    MPI_Send(send_buf[rank] + n * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i + 1], 
                             n * flowpath.size() + i/*tag*/, MPI_COMM_WORLD);
                } else if (i + 1 == flowpath.size()) {
                    MPI_Recv(send_buf[rank] + n * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i - 1], 
                             n * flowpath.size() + i - 1/*tag*/, MPI_COMM_WORLD, nullptr);
                } else {
                    if (n == 0) {
                        MPI_Recv(send_buf[rank] + n * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i - 1], 
                                 n * flowpath.size() + i - 1/*tag*/, MPI_COMM_WORLD, nullptr);
                    } else if (n + 1 == NUM) {
                        MPI_Sendrecv(send_buf[rank] + (n - 1) * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i + 1], (n - 1) * flowpath.size() + i, \
                                     send_buf[rank] + n * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i - 1], n * flowpath.size() + i - 1, \
                                     MPI_COMM_WORLD, nullptr);
                        MPI_Send(send_buf[rank] + n * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i + 1], 
                                 n * flowpath.size() + i/*tag*/, MPI_COMM_WORLD);
                    } else {
                        MPI_Sendrecv(send_buf[rank] + (n - 1) * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i + 1], (n - 1) * flowpath.size() + i, \
                                     send_buf[rank] + n * CHUNK_SIZE, CHUNK_SIZE, MPI_INT, flowpath[i - 1], n * flowpath.size() + i - 1, \
                                     MPI_COMM_WORLD, nullptr);
                    }
                }
                // printf("%d, %d: out\n", rank, n);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Need at least 4 args: \"<command> <gpus> <backend> <cp_file>\"\n");
        return - 1;
    }
    setup_env(pp, argc, argv);
    std::string cp_file = argv[3];

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

    // int CHUNK_SIZE = 1024 * 1024 / 4;    // 1MB
    // int CHUNK_SIZE_S = 1024 * 1024 / 8;
    int CHUNK_SIZE_S = 1024 * 1024 * 128;
    int CHUNK_SIZE_E = 1024 * 1024 * 128;
    for (int CHUNK_SIZE = CHUNK_SIZE_S; CHUNK_SIZE <= CHUNK_SIZE_E; CHUNK_SIZE <<= 1) {
        for (int i = SIZEIDX_START; i < SIZEIDX_END; ++ i) {
            LL SIZE = SIZES[i];
    #ifdef CHECK_RESULT
            SIZE = comm_size * comm_size;
    #endif 
            // const LL SSIZE = SIZE / comm_size;
            // const LL CHUNK_SIZE = SIZE / (comm_size * comm_size);
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
            
            // 2 link:
            // std::vector<int> flowpath = {1, 6, 4};       // 40.9, 16MB
            // std::vector<int> flowpath = {3, 1, 6, 4};    // 42.2, 16MB

            // 1 link:
            // std::vector<int> flowpath = {0, 1, 2};          // 21.5, 8MB 
            // std::vector<int> flowpath = {0, 1, 2, 5};       // 21.2, 8MB
            // std::vector<int> flowpath = {0, 1, 2, 5, 6};       // 20.8, 8MB

            // mix link:
            // std::vector<int> flowpath = {1, 0, 7, 4};       // 21.4, 8MB
            // std::vector<int> flowpath = {1, 3, 2, 5, 4};       // 21.2, 8MB
            std::vector<int> flowpath = {1, 2, 0, 7, 5};       // 21.4, 8MB

            // WARMUP
            for (int _ = 0; _ < WARMUP; ++ _) {
                MPI_netflow(send_buf, recv_buf, SIZE, CHUNK_SIZE, flowpath, pp->streams, pp->rank, pp->comm, pp->mpi_requests);
                // XXX_comm(root[cp], send_buf, recv_buf, SIZE, pp->streams, pp->rank, pp->comm, pp->mpi_requests);
                barrier(pp->BACKEND, pp->N_GPUs);
                // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
            }

            barrier(pp->BACKEND, pp->N_GPUs);

            // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int _ = 0; _ < TIMES; ++ _) {
                MPI_netflow(send_buf, recv_buf, SIZE, CHUNK_SIZE, flowpath, pp->streams, pp->rank, pp->comm, pp->mpi_requests);
                barrier(pp->BACKEND, pp->N_GPUs);
                // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
            }
            // CUDA_CHECK(cudaEventRecord(stop_a2a, stream));
            // CUDA_CHECK(cudaEventSynchronize(stop_a2a));
            // still async !!!
            // CUDA_CHECK(cudaStreamSynchronize(stream));
            // CUDA_CHECK(cudaDeviceSynchronize());
            // MPI_Barrier(MPI_COMM_WORLD);
            // devicesSyncAll(N_GPUs);
            barrier(pp->BACKEND, pp->N_GPUs);

            auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT
            // CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start_a2a, stop_a2a));    // ms
            // if (true) {
            if (pp->rank == 0) {
                // double t_d = (double)elapsedTime / 1000;    // s
                double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
                // double calc = root[cp].size() * (double)SIZE * sizeof(int) * TIMES;      // B
                double calc = 1 * (double)SIZE * sizeof(int) * TIMES;      // B
                // double calc = 1 * (double)CHUNK_SIZE * sizeof(int) * TIMES;      // B
                double avg_bd = calc / t_d / pow(1024, 3);
                printf("time %lf s, REAL_BD %lf GB/s, SIZE %lf KB, CHUNK_SIZE %lf KB, comm_vol %lf KB\n", \
                        t_d, avg_bd, (double)SIZE * sizeof(int) / pow(1024, 1), \
                        (double)CHUNK_SIZE * sizeof(int) / pow(1024, 1), calc / pow(1024, 1));
#ifdef PRINT_JSON
                root[method]["time"].append(Json::Value(t_d));
                root[method]["REAL_BD"].append(Json::Value(avg_bd));
                root[method]["SIZE"].append(Json::Value((double)SIZE * sizeof(int)));
                root[method]["comm_vol"].append(Json::Value(calc));
#endif
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
    }
    delete pp;
    // MPI_Finalize();
    return 0;
}