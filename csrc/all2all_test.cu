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
#include <string>
#include <iostream>

// #define CHECK_RESULT
int TIMES = 20;

typedef long long LL;
const LL SIZES[8] = {
    1LL * 1024 * 1024 * 64,
    1LL * 1024 * 1024 * 128,
    1LL * 1024 * 1024 * 256,
    1LL * 1024 * 1024 * 512,
    1LL * 1024 * 1024 * 1024,
    1LL * 1024 * 1024 * 2048,
    1LL * 1024 * 1024 * 4096,
    1LL * 1024 * 1024 * 8192,   // OOM
};


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

    const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC0"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC1"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"SC4"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"BRUCK"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"RD"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"2DMESH"}};
    // const int METHOD_NUM = 1; std::string methods[METHOD_NUM] = {{"3DMESH"}};
    // const int METHOD_NUM = 2; std::string methods[METHOD_NUM] = {{"SC0"}, {"SC1"}};
    // const int ALL_METHOD_NUM = 2; const std::string ALL_METHODS[ALL_METHOD_NUM] = {{"SC0"}, {"SC1"}};

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
        for (int i = 0; i < 7; ++ i) {
            
        }
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
            
            MPI_Barrier(MPI_COMM_WORLD);
            // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int _ = 0; _ < TIMES; ++ _) {
                all2all_SCX(input_list, output_list, CHUNK_SIZE, comm_size, rank, comm, ncclFloat32, stream, true);
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
                printf("SIZE: %lf GB\n", (double)SIZE * sizeof(int) / pow(1024, 3));
                double calc = (double)CHUNK_SIZE * (comm_size - 1) * sizeof(int) * TIMES / pow(1024, 3);   // GB
                double avg_bd = calc / t_d;
                printf("%s: %lf s, REAL_BD %lf GB/s, comm_vol %lf GB\n", \
                        method.c_str(), t_d, avg_bd, calc);
                // printf("SC0: %lf s, REAL_BD %lf GB/s, TOTAL_BD %lf GB/s, comm_vol %lf GB\n", \
                //         t_d, avg_bd, avg_bd, calc);
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
    
    MPI_Finalize();
    return 0;
}