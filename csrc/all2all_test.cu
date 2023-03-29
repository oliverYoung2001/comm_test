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
int TIMES = 20;
int WARMUP = 10;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

// const int SIZES_LEN = 9;
// const LL SIZES[SIZES_LEN] = {   // int = 4B
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


int main(int argc, char** argv) {
    //Get number of gpus in the node
    int N_GPUs;
    CUDA_CHECK(cudaGetDeviceCount(&N_GPUs));
    MPI_Init(&argc, &argv);
    int comm_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ncclComm_t comm;
    CUDA_CHECK(cudaSetDevice(rank % N_GPUs));
    //initializing NCCL
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, comm_size, id, rank);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // printf("rank %d: initializing done !!!\n", rank);
    // fflush(stdout);

    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC0"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC1"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC4"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"BRUCK"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"RD"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"2DMESH"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"3DMESH"}};
    // const int METHOD_NUM = 2; std::string methods[METHOD_NUM] = {{"SC0"}, {"SC1"}};
    const int METHOD_NUM = 7; std::string methods[METHOD_NUM] = {{"SC0"}, {"SC1"}, {"SC4"}, {"BRUCK"}, {"RD"}, {"2DMESH"}, {"3DMESH"}};
    // const int ALL_METHOD_NUM = 2; const std::string ALL_METHODS[ALL_METHOD_NUM] = {{"SC0"}, {"SC1"}};

    // result_json = {};
    Json::Value root;
    for (int m_id = 0; m_id < METHOD_NUM; ++ m_id) {
        std::string method = methods[m_id];

        void (*all2all_SCX)(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, \
                            ncclComm_t comm, ncclDataType_t ncclDataType,  cudaStream_t stream, bool async_op);
        if (method.compare("SC0") == 0) {
            all2all_SCX = all2all_SC0;
        } else if (method.compare("SC1") == 0) {
            all2all_SCX = all2all_SC1;
        } else if (method.compare("SC4") == 0) {
            all2all_SCX = all2all_SC4;
        } else if (method.compare("BRUCK") == 0) {
            all2all_SCX = all2all_BRUCK;
        } else if (method.compare("RD") == 0) {
            all2all_SCX = all2all_RD;
        } else if (method.compare("2DMESH") == 0) {
            all2all_SCX = all2all_2DMESH;
        } else if (method.compare("3DMESH") == 0) {
            all2all_SCX = all2all_3DMESH;
        } else {
            printf("Wrong method: '%s' !!!\n", method.c_str());
            exit(- 1);
        }
        // switch (method) {
        //     case ALL_METHODS[0]:
        //         all2all_SCX = all2all_SC0;
        //         break;
        //     case ALL_METHODS[1]:
        //         all2all_SCX = all2all_SC1;
        //         break;
        //     default:
        //         printf("Wrong method: '%s' !!!", method.c_str());
        //         exit(- 1);
        // }
        for (int i = 0; i < SIZES_LEN - 2; ++ i) {
        // for (int i = 0; i < 7; ++ i) {
            LL SIZE = SIZES[i];
    #ifdef CHECK_RESULT
            SIZE = comm_size * comm_size;
    #endif 
            const LL SSIZE = SIZE / comm_size;
            const LL CHUNK_SIZE = SIZE / (comm_size * comm_size);
            int* send_buf_cpu = new int[SSIZE];
            int* recv_buf_cpu = new int[SSIZE];
            int* send_buf;
            int* recv_buf;
            int** input_list_cpu = new int*[comm_size];
            int** output_list_cpu = new int*[comm_size];
            int** input_list = new int*[comm_size];
            int** output_list = new int*[comm_size];
            CUDA_CHECK(cudaMalloc(&send_buf, SSIZE * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&recv_buf, SSIZE * sizeof(int)));
            for (int j = 0; j < comm_size; ++ j) {
                input_list_cpu[j] = send_buf_cpu + j * CHUNK_SIZE;
                output_list_cpu[j] = recv_buf_cpu + j * CHUNK_SIZE;
                input_list[j] = send_buf + j * CHUNK_SIZE;
                output_list[j] = recv_buf + j * CHUNK_SIZE;
            }
            // printf("rank %d: malloc && pointer done !!!\n", rank);
            // fflush(stdout);

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
                all2all_SCX(input_list, output_list, CHUNK_SIZE, comm_size, rank, comm, ncclFloat32, stream, true);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Barrier(MPI_COMM_WORLD);
            // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int _ = 0; _ < TIMES; ++ _) {
                all2all_SCX(input_list, output_list, CHUNK_SIZE, comm_size, rank, comm, ncclFloat32, stream, true);
                CUDA_CHECK(cudaDeviceSynchronize());    // [WHY]: 每一轮Sync一下会有性能提升！！！
            }
            // CUDA_CHECK(cudaEventRecord(stop_a2a, stream));
            // CUDA_CHECK(cudaEventSynchronize(stop_a2a));
            // still async !!!
            // CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Barrier(MPI_COMM_WORLD);
            auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT
            // CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start_a2a, stop_a2a));    // ms
            if (rank == 0) {
                // double t_d = (double)elapsedTime / 1000;    // s
                double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
                double calc = (double)CHUNK_SIZE * (comm_size - 1) * sizeof(int) * TIMES / pow(1024, 3);   // GB
                double avg_bd = calc / t_d;
                printf("%s: %lf s, REAL_BD %lf GB/s, SIZE %lf GB, comm_vol %lf GB\n", \
                        method.c_str(), t_d, avg_bd, (double)SIZE * sizeof(int) / pow(1024, 3), calc);
                // printf("SC0: %lf s, REAL_BD %lf GB/s, TOTAL_BD %lf GB/s, comm_vol %lf GB\n", \
                //         t_d, avg_bd, avg_bd, calc);
                root[method]["time"].append(Json::Value(t_d));
                root[method]["REAL_BD"].append(Json::Value(avg_bd));
                root[method]["SIZE"].append(Json::Value((double)SIZE * sizeof(int)));
                root[method]["comm_vol"].append(Json::Value(calc));
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

            CUDA_CHECK(cudaFree(recv_buf));
            CUDA_CHECK(cudaFree(send_buf));
            delete[] output_list;
            delete[] input_list;
            delete[] output_list_cpu;
            delete[] input_list_cpu;
            delete[] recv_buf_cpu;
            delete[] send_buf_cpu;
        }
    }

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
    
    MPI_Finalize();
    return 0;
}