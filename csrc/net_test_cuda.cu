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
// #include <format>    // need c++20
typedef long long LL;

// #define CHECK_RESULT
// #define PRINT_JSON
#define RECORD_TABLE
int TIMES = 20;
int WARMUP = 10;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZES_LEN = 9;
const LL SIZES[SIZES_LEN] = {   // int = 4B
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

void devicesSyncAll(int N_GPUs) {
    for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
        CUDA_CHECK(cudaSetDevice(gpuid));
        CUDA_CHECK(cudaDeviceSynchronize());
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

int main(int argc, char** argv) {
    //Get number of gpus in the node
    int N_GPUs;
    CUDA_CHECK(cudaGetDeviceCount(&N_GPUs));
    N_GPUs = 4;
    // MPI_Init(&argc, &argv);
    // int comm_size, rank;
    // MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // ncclComm_t comm;
    // CUDA_CHECK(cudaSetDevice(rank % N_GPUs));
    //initializing NCCL
    // ncclUniqueId id;
    // if (rank == 0) ncclGetUniqueId(&id);
    // MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    // ncclCommInitRank(&comm, comm_size, id, rank);
    
    int STREAM_NUM = N_GPUs;
    cudaStream_t* streams = new cudaStream_t[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; ++ i) {
        cudaStreamCreate(&streams[i]);
    }


    check_UVA(N_GPUs);        // 我理解，统一内存编址是为了方便，而不是性能
    enableP2P(N_GPUs);

    // printf("rank %d: initializing done !!!\n", rank);
    // fflush(stdout);

    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC0"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC1"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC4"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC5"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"BRUCK"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"RD"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"2DMESH"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"3DMESH"}};
    // const int METHOD_NUM = 2; std::string methods[METHOD_NUM] = {{"SC0"}, {"SC1"}};
    // const int METHOD_NUM = 7; std::string methods[METHOD_NUM] = {{"SC0"}, {"SC1"}, {"SC4"}, {"BRUCK"}, {"RD"}, {"2DMESH"}, {"3DMESH"}};
    // const int ALL_METHOD_NUM = 2; const std::string ALL_METHODS[ALL_METHOD_NUM] = {{"SC0"}, {"SC1"}};

    // result_json = {};
#ifdef PRINT_JSON
    Json::Value root;
#endif
#ifdef RECORD_TABLE
    double result_table[N_GPUs][N_GPUs];
#endif
    // for (int m_id = 0; m_id < METHOD_NUM; ++ m_id) {
    for (int src = 0; src < N_GPUs; ++ src) {
    // for (int src = 4; src < 5; ++ src) {
    for (int dst = 0; dst < N_GPUs; ++ dst) {
        if (src == dst || std::max(src, dst) >= N_GPUs) {
            continue;
        }
        if (true) {
        // if (rank == 0) {
            printf("(%d, %d)\n", src, dst);
        }
        // for (int i = 0; i < SIZES_LEN - 2; ++ i) {
        int SIZEIDX_START = 0;
        int SIZEIDX_END = 6;
        for (int i = SIZEIDX_START; i < SIZEIDX_END; ++ i) {
            LL SIZE = SIZES[i];
    #ifdef CHECK_RESULT
            SIZE = comm_size * comm_size;
    #endif 
            // const LL SSIZE = SIZE / comm_size;
            // const LL CHUNK_SIZE = SIZE / (comm_size * comm_size);
            int* send_buf_cpu = new int[SIZE];
            int* recv_buf_cpu = new int[SIZE];
            int** send_buf = new int*[N_GPUs];
            int** recv_buf = new int*[N_GPUs];
            CUDA_CHECK(cudaSetDevice(src));
            CUDA_CHECK(cudaMalloc(&send_buf[src], SIZE * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&recv_buf[src], SIZE * sizeof(int)));
            CUDA_CHECK(cudaSetDevice(dst));
            CUDA_CHECK(cudaMalloc(&send_buf[dst], SIZE * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&recv_buf[dst], SIZE * sizeof(int)));
            // for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
            //     CUDA_CHECK(cudaSetDevice(gpuid));
            //     CUDA_CHECK(cudaMalloc(&send_buf[gpuid], SIZE * sizeof(int)));
            //     CUDA_CHECK(cudaMalloc(&recv_buf[gpuid], SIZE * sizeof(int)));
            // }

    #ifdef CHECK_RESULT
            TIMES = 1;
            for (int j = 0; j < comm_size; ++ j) {
                int v = rank * comm_size + j;
                for (int k = 0; k < CHUNK_SIZE; ++ k) {
                    input_list_cpu[j][k] = v;
                }
            }
            printf("rank %d, send_buf_cpu:\n", rank);
            for (int j = 0; j < SSIZE; ++ j) {
                printf("%d ", send_buf_cpu[j]);
            }
            puts("");
            CUDA_CHECK(cudaMemcpy(send_buf, send_buf_cpu, SSIZE * sizeof(int), cudaMemcpyDefault));
    #endif
            // cudaEvent_t start_a2a, stop_a2a;
            // float elapsedTime;
            // CUDA_CHECK(cudaEventCreate(&start_a2a));
            // CUDA_CHECK(cudaEventCreate(&stop_a2a));
            
            // WARMUP
            for (int _ = 0; _ < WARMUP; ++ _) {
                cudaMemcpyAsync(recv_buf[dst], send_buf[src], SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[0]);
                cudaMemcpyAsync(recv_buf[src], send_buf[dst], SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[1]);
                // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
            }

            // CUDA_CHECK(cudaDeviceSynchronize());
            // MPI_Barrier(MPI_COMM_WORLD);
            devicesSyncAll(N_GPUs);

            // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int _ = 0; _ < TIMES; ++ _) {
                cudaMemcpyAsync(recv_buf[dst], send_buf[src], SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[0]);
                cudaMemcpyAsync(recv_buf[src], send_buf[dst], SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[1]);
                // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
                // CUDA_CHECK(cudaDeviceSynchronize());    // light-barrier, [WHY]: 会有性能提升！！！ 减少 comm contention ?
                // MPI_Barrier(MPI_COMM_WORLD);            // cpu-barrier, 没有意义
            }
            // CUDA_CHECK(cudaEventRecord(stop_a2a, stream));
            // CUDA_CHECK(cudaEventSynchronize(stop_a2a));
            // still async !!!
            // CUDA_CHECK(cudaStreamSynchronize(stream));
            // CUDA_CHECK(cudaDeviceSynchronize());
            // MPI_Barrier(MPI_COMM_WORLD);
            devicesSyncAll(N_GPUs);

            auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT
            // CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start_a2a, stop_a2a));    // ms
            if (true) {
            // if (rank == 0) {
                // double t_d = (double)elapsedTime / 1000;    // s
                double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
                double calc = (double)SIZE * sizeof(int) * TIMES / pow(1024, 3);
                double avg_bd = calc / t_d;
                printf("time %lf s, REAL_BD %lf GB/s, SIZE %lf GB, comm_vol %lf GB\n", \
                        t_d, avg_bd, (double)SIZE * sizeof(int) / pow(1024, 3), calc);
#ifdef RECORD_TABLE
                if (i + 1 == SIZEIDX_END) {
                    result_table[src][dst] = avg_bd;
                }
#endif
#ifdef PRINT_JSON
                root[method]["time"].append(Json::Value(t_d));
                root[method]["REAL_BD"].append(Json::Value(avg_bd));
                root[method]["SIZE"].append(Json::Value((double)SIZE * sizeof(int)));
                root[method]["comm_vol"].append(Json::Value(calc));
#endif
                fflush(stdout);
            }
            

    #ifdef CHECK_RESULT
            // CUDA_CHECK(cudaMemcpy(recv_buf_cpu, recv_buf, SSIZE * sizeof(int), cudaMemcpyDefault));
            for (int j = 0; j < comm_size; ++ j) {
                CUDA_CHECK(cudaMemcpy(recv_buf_cpu + j * CHUNK_SIZE, output_list[j], 
                                      CHUNK_SIZE * sizeof(int), cudaMemcpyDefault));
            }
            printf("rank %d, recv_buf_cpu:\n", rank);
            for (int j = 0; j < SSIZE; ++ j) {
                printf("%d ", recv_buf_cpu[j]);
            }
            puts("");
            for (int j = 0; j < comm_size; ++ j) {
                int v = j * comm_size + rank;
                for (int k = 0; k < CHUNK_SIZE; ++ k) {
                    if (output_list_cpu[j][k] != v) {
                        printf("Failed: Comm(All2All) error %s:%d '%d!=%d'\n",             \
                                __FILE__,__LINE__,output_list_cpu[j][k],v);   \
                        exit(EXIT_FAILURE); 
                    }
                    // assert(output_list[j][k] == v)
                }
            }
            printf("rank %d: ALL2ALL CORRECT !!!\n", rank);
            fflush(stdout);
    #endif

            CUDA_CHECK(cudaSetDevice(src));
            CUDA_CHECK(cudaFree(send_buf[src]));
            CUDA_CHECK(cudaFree(recv_buf[src]));
            CUDA_CHECK(cudaSetDevice(dst));
            CUDA_CHECK(cudaFree(send_buf[dst]));
            CUDA_CHECK(cudaFree(recv_buf[dst]));
            // for (int gpuid = 0; gpuid < N_GPUs; ++ gpuid) {
            //     CUDA_CHECK(cudaFree(send_buf[gpuid]));
            //     CUDA_CHECK(cudaFree(recv_buf[gpuid]));
            // }
            delete[] recv_buf;
            delete[] send_buf;
            // CUDA_CHECK(cudaFree(recv_buf));
            // CUDA_CHECK(cudaFree(send_buf));
            // delete[] output_list;
            // delete[] input_list;
            // delete[] output_list_cpu;
            // delete[] input_list_cpu;
            delete[] recv_buf_cpu;
            delete[] send_buf_cpu;
        }
    }
    }

#ifdef PRINT_JSON
    if (rank == 0) {
        Json::StyledWriter sw;
        // std::cout << "StyledWriter:" << std::endl;
        // std::cout << sw.write(root) << std::endl << std::endl;

        // string output_file = std::format("./results/{}cu_all2all.json", comm_size);
        std::string output_file = "results/" + std::to_string(comm_size) + "cu_all2all.json";
        std::ofstream os(output_file.c_str(), std::ios::out);    // 覆盖写
        if (! os.is_open()) {
            std::cout << "error: can not find or create the file which named \" demo.json\"." << std::endl;
        }
        os << sw.write(root);
        os.close();
    }
#endif
#ifdef RECORD_TABLE
    printf("RESULT_TABLE: \n");
    for (int src = 0; src < N_GPUs; ++ src) {
        for (int dst = 0; dst < N_GPUs; ++ dst) {
            printf("%lf ", result_table[src][dst]);
        }
        puts("");
    }
    puts("");
#endif
    
    disableP2P(N_GPUs);
    for (int i = 0; i < STREAM_NUM; ++ i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    // MPI_Finalize();
    return 0;
}