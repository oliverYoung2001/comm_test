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

std::vector<int*> ringid_arrays;
int N_GPUs;

const int RING_SIZE = 8;
int TURNs = 0;
// #include <format>    // need c++20
typedef long long LL;

// #define CHECK_RESULT
// #define PRINT_JSON
// #define RECORD_TABLE
#define ENABLE_GPU_P2P       // 性能不一定好！！！ 单个P2P更好，但多个P2P不一定好
int TIMES = 4;
int WARMUP = 2;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZEIDX_START = 6;
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

void devicesSyncAll(int N_GPUs) {
    for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
        CUDA_CHECK(cudaSetDevice(gpuid));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void barrier(std::string& BACKEND, int N_GPUs) {
    if (BACKEND.compare("cudaMemcpy") == 0) {
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

void check_UVA(int ngpus) {
    for (int gpuid = 0; gpuid < ngpus; ++ gpuid) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuid));
        printf("GPU%d: %s unified addressing\n", gpuid, prop.unifiedAddressing ? "supports" : "does not support");
        fflush(stdout);
    }
    
}
inline void enableP2P(Json::Value& pairs) {
    for (int k = 0; k < pairs.size(); ++ k) {
        int src = pairs[k][0].asInt();
        int dst = pairs[k][1].asInt();
        CUDA_CHECK(cudaSetDevice(src));
        int peer_access_available = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&peer_access_available, src, dst));
        if (peer_access_available) {
            CUDA_CHECK(cudaDeviceEnablePeerAccess(dst, 0));
        } else {
            printf("> GPU%d disabled direct access to GPU%d !!!\n", src, dst);
            fflush(stdout);
        }
    }
}

void disableP2P(Json::Value& pairs) {
    for (int k = 0; k < pairs.size(); ++ k) {
        int src = pairs[k][0].asInt();
        int dst = pairs[k][1].asInt();
        CUDA_CHECK(cudaSetDevice(src));
        CUDA_CHECK(cudaDeviceDisablePeerAccess(dst));
    }
}

inline void enableP2P(int ngpus) {
    for (int i = 0; i < ngpus; ++ i) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < ngpus; ++ j) {
            if (i == j) {
                continue;
            }
            int peer_access_available = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&peer_access_available, i, j));
            
            if (peer_access_available) {
                CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
                // printf("> GPU%d enabled direct access to GPU%d\n", i, j);
                // fflush(stdout);
            } else {
                printf("> GPU%d disabled direct access to GPU%d !!!\n", i, j);
                fflush(stdout);
            }
        }
    }
}

void disableP2P(int ngpus) {
    for (int i = 0; i < ngpus; ++ i) {
        for (int j = 0; j < ngpus; ++ j) {
            if (i == j) {
                continue;
            }
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaDeviceDisablePeerAccess(j));
        }
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

void cudaMemcpy_comm(int* ids, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    int dst;
    for (int k = 0; k < nranks; ++ k) {
        dst = (k + 1) % RING_SIZE == 0 ? ids[k + 1 - RING_SIZE] : ids[k + 1];
        CUDA_CHECK(cudaMemcpyPeerAsync(recv_buf[dst], dst, \
                                   send_buf[ids[k]], ids[k], \
                                   SIZE * sizeof(int), streams[k]));  
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

void NCCL_comm(int* ids, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    NCCL_CHECK(ncclGroupStart());
    int src, dst;
    for (int k = 0; k < nranks; ++ k) {
        src = ids[k];
        dst = (k + 1) % RING_SIZE == 0 ? ids[k + 1 - RING_SIZE] : ids[k + 1];
        if (rank == src) {
            NCCL_CHECK(ncclSend(send_buf[rank], SIZE, ncclInt32, dst, comm, streams[0]));
        }
        if (rank == dst) {
            NCCL_CHECK(ncclRecv(recv_buf[rank], SIZE, ncclInt32, src, comm, streams[0]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
    // printf("rank: %d", rank);
    // Json::FastWriter sw;
    // std::cout << sw.write(pairs);
    // NCCL_CHECK(ncclGroupStart());
    // for (int k = 0; k < pairs.size(); ++ k) {
    //     if (rank == pairs[k][0].asInt()) {
    //         NCCL_CHECK(ncclSend(send_buf[rank], SIZE, ncclInt32, pairs[k][1].asInt(), comm, streams[0]));
    //     }
    //     if (rank == pairs[k][1].asInt()) {
    //         NCCL_CHECK(ncclRecv(recv_buf[rank], SIZE, ncclInt32, pairs[k][0].asInt(), comm, streams[0]));
    //     }
    // }
    // NCCL_CHECK(ncclGroupEnd());
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

void MPI_comm(int* ids, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request) {
    // int req_num = 0;
    // for (int k = 0; k < pairs.size(); ++ k) {
    //     if (rank == pairs[k][0].asInt()) {
    //         MPI_Isend(send_buf[rank], SIZE, MPI_INT, pairs[k][1].asInt(), 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
    //     }
    //     if (rank == pairs[k][1].asInt()) {
    //         MPI_Irecv(recv_buf[rank], SIZE, MPI_INT, pairs[k][0].asInt(), 0/*tag*/, MPI_COMM_WORLD, mpi_request + (req_num ++));
    //     }
    // }
    // // MPI_Wait(mpi_request, NULL);
    // MPI_Waitall(req_num, mpi_request , nullptr);
}

void dfs(int turn, int id, int* tmp_ids, bool* P) {
    // printf("turn: %d, id: %d\n", turn, id);
    if (turn >= TURNs) {
        // ringid_arrays
        int* new_ids = new int[N_GPUs];
        memcpy(new_ids, tmp_ids, N_GPUs * sizeof(int));
        ringid_arrays.push_back(new_ids);
        return;
    }
    for (int i = 0; i < N_GPUs; ++ i) {
        if (! P[i]) {
            tmp_ids[turn * RING_SIZE + id] = i;
            P[i] = 1;
            dfs(turn + (id + 1 == RING_SIZE), (id + 1) % RING_SIZE, tmp_ids, P);
            P[i] = 0;
            if (id == 0) {
                return;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Need at least 2 args: \"<command> <gpus> <backend>\"\n");
        return - 1;
    }
    //Get number of gpus in the node
    int GPU_VISIBLE;
    CUDA_CHECK(cudaGetDeviceCount(&GPU_VISIBLE));
    N_GPUs = std::stoi(argv[1]);
    assert(N_GPUs <= GPU_VISIBLE);
    std::string BACKEND = argv[2];
    assert(N_GPUs % RING_SIZE == 0);
    TURNs = N_GPUs / RING_SIZE;

    void (*XXX_comm)(int* ring_ids, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, int nranks, ncclComm_t comm, MPI_Request* mpi_request);
    if (BACKEND.compare("NCCL") == 0) {
        XXX_comm = NCCL_comm;
    } else if (BACKEND.compare("MPI") == 0) {
        XXX_comm = MPI_comm;
    } else if (BACKEND.compare("cudaMemcpy") == 0) {
        XXX_comm = cudaMemcpy_comm;
    } else {
        printf("Error BACKEND !!!");
        return - 1;
    }

    // Init MPI
    int comm_size, rank;
    if (BACKEND.compare("NCCL") == 0 || BACKEND.compare("MPI") == 0) {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        CUDA_CHECK(cudaSetDevice(rank % comm_size));
        assert(N_GPUs == comm_size);
    }
    // Init NCCL
    ncclComm_t comm;
    if (BACKEND.compare("NCCL") == 0) {
        ncclUniqueId id;
        if (rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclCommInitRank(&comm, comm_size, id, rank);
    }
    if (BACKEND.compare("cudaMemcpy") == 0) {
        comm_size = 0;
        rank = 0;
#ifdef ENABLE_GPU_P2P
        enableP2P(N_GPUs);   // disable 后会有较大的性能下降，因为会走CPU memory
#endif
    }
    if (rank == 0) {
        printf("BACKEND: %s\n", BACKEND.c_str());
#ifdef ENABLE_GPU_P2P
        printf("ENABLE_GPU_P2P !!!\n");
#else
        printf("DISABLE_GPU_P2P !!!\n");
#endif
    }
    if (rank == 0) {
        printf("N_GPUs: %d, TURNs: %d, RING_SIZE: %d\n", N_GPUs, TURNs, RING_SIZE);
    }


    // Init cudaStream
    // int max_pair_num = 0;
    // for (int cp = 0; cp < root.size(); ++ cp) {
    //     // int tmp = 
    //     max_pair_num = std::max(max_pair_num, (int)root[cp].size());
    // }
    // int STREAM_NUM = std::max(N_GPUs, max_pair_num);
    int STREAM_NUM = N_GPUs;
    cudaStream_t* streams = new cudaStream_t[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; ++ i) {
        cudaStreamCreate(&streams[i]);
    }
    MPI_Request mpi_request[N_GPUs];


    LL SIZE = SIZES[SIZEIDX_START];
    int** send_buf = new int*[N_GPUs];
    int** recv_buf = new int*[N_GPUs];
    if (BACKEND.compare("cudaMemcpy") == 0) {
        for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
            CUDA_CHECK(cudaSetDevice(gpuid));
            CUDA_CHECK(cudaMalloc(&send_buf[gpuid], SIZE * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&recv_buf[gpuid], SIZE * sizeof(int)));
        }
    }
    if (BACKEND.compare("NCCL") == 0 || BACKEND.compare("MPI") == 0) {
        CUDA_CHECK(cudaMalloc(&send_buf[rank], SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&recv_buf[rank], SIZE * sizeof(int)));
    }

    bool P[N_GPUs];
    memset(P, 0, sizeof(P));

    int tmp_ids[N_GPUs];
    dfs(0, 0, tmp_ids, P);
    if (rank == 0) {
        printf("ringid_arrays.size(): %d\n", ringid_arrays.size());
    }
    for (auto ring_ids : ringid_arrays) {
        // WARMUP
        for (int _ = 0; _ < WARMUP; ++ _) {
            XXX_comm(ring_ids, send_buf, recv_buf, SIZE, streams, rank, N_GPUs, comm, mpi_request);
            barrier(BACKEND, N_GPUs);
            // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
        }
        barrier(BACKEND, N_GPUs);

        // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int _ = 0; _ < TIMES; ++ _) {
            XXX_comm(ring_ids, send_buf, recv_buf, SIZE, streams, rank, N_GPUs, comm, mpi_request);
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
            // // double t_d = (double)elapsedTime / 1000;    // s
            double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
            double calc = N_GPUs * (double)SIZE * sizeof(int) * TIMES;      // B
            double avg_bd = calc / t_d;
            for (int k = 0; k < N_GPUs; ++ k) {
                printf("%d", ring_ids[k]);
            }
            // printf(": time %lf s, REAL_BD %lf GB/s, SIZE %lf KB, comm_vol %lf KB\n", \
            //         t_d, avg_bd / pow(1024, 3) , (double)SIZE * sizeof(int) / pow(1024, 1), calc / pow(1024, 1));
            printf(": time %lf s, REAL_BD %lf GB/s, SIZE %lf KB\n", \
                    t_d, avg_bd / pow(1024, 3) , (double)SIZE * sizeof(int) / pow(1024, 1));
            fflush(stdout);
        }
    }
    // } while (std::next_permutation(ring_ids + 1, ring_ids + N_GPUs));


    if (BACKEND.compare("cudaMemcpy") == 0) {
        for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
            CUDA_CHECK(cudaFree(recv_buf[gpuid]));
            CUDA_CHECK(cudaFree(send_buf[gpuid]));
        }
    }
    if (BACKEND.compare("NCCL") == 0 || BACKEND.compare("MPI") == 0) {
        CUDA_CHECK(cudaFree(recv_buf[rank]));
        CUDA_CHECK(cudaFree(send_buf[rank]));
    }
    
    delete[] recv_buf;
    delete[] send_buf;

    if (BACKEND.compare("cudaMemcpy") == 0) {
#ifdef ENABLE_GPU_P2P
        disableP2P(N_GPUs);
#endif
    }
    for (int i = 0; i < STREAM_NUM; ++ i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    // MPI_Finalize();
    return 0;
}