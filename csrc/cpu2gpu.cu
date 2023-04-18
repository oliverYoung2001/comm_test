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

// #define CHECK_RESULT
// #define PRINT_JSON
// #define RECORD_TABLE
int TIMES = 20;
int WARMUP = 10;
const int MAGIC_FACTOR = pow(2, 5) * pow(3, 3) * pow(5, 2) * 7;     // 151200, for tests on different number of GPUs
// 62792 B

const int SIZEIDX_START = 0;
const int SIZEIDX_END = 5;

const int SIZES_LEN = 10;
const LL SIZES[SIZES_LEN] = {   // int = 4B
    1LL * 1024 * 1024 * 1,      // 4MB      // 打不满带宽
    1LL * 1024 * 1024 * 32,     // 128MB
    1LL * 1024 * 1024 * 64,     // 256MB
    1LL * 1024 * 1024 * 128,    // 512MB
    1LL * 1024 * 1024 * 256,    // 1GB
    1LL * 1024 * 1024 * 512,    // 2GB
    1LL * 1024 * 1024 * 1024,   // 4GB      // 用cudaMemcpy，竟然有性能下降！！！
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
        CUDA_CHECK(cudaMemcpyAsync(recv_buf[pairs[k][1].asInt()], send_buf[pairs[k][0].asInt()], 
                                    SIZE * sizeof(int), cudaMemcpyHostToDevice, streams[k]));
        // CUDA_CHECK(cudaMemcpyPeerAsync(recv_buf[pairs[k][1].asInt()], pairs[k][1].asInt(), \
        //                            send_buf[pairs[k][0].asInt()], pairs[k][0].asInt(), \
        //                            SIZE * sizeof(int), streams[k]));                                // 两者性能相似
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

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Need at least 2 args: \"<command> <gpus> <backend> <cp_file>\"\n");
        return - 1;
    }
    //Get number of gpus in the node
    int N_GPUs, GPU_VISIBLE;
    CUDA_CHECK(cudaGetDeviceCount(&GPU_VISIBLE));
    N_GPUs = std::stoi(argv[1]);
    assert(N_GPUs <= GPU_VISIBLE);
    std::string BACKEND = argv[2];
    std::string cp_file = argv[3];

    void (*XXX_comm)(Json::Value& pairs, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request);
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
        enableP2P(N_GPUs);   // disable 后会有较大的性能下降，以为会走CPU memory
    }
    if (rank == 0) {
        printf("BACKEND: %s\n", BACKEND.c_str());
    }

#ifdef PRINT_JSON
    Json::Value root;
#endif
#ifdef RECORD_TABLE
    double result_table_si[N_GPUs][N_GPUs];
    double result_table_bi[N_GPUs][N_GPUs];
    memset(result_table_si, 0, sizeof(result_table_si));
    memset(result_table_bi, 0, sizeof(result_table_bi));
#endif
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
        // int tmp = 
        max_pair_num = std::max(max_pair_num, (int)root[cp].size());
    }
    int STREAM_NUM = std::max(N_GPUs, max_pair_num);
    cudaStream_t* streams = new cudaStream_t[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; ++ i) {
        cudaStreamCreate(&streams[i]);
    }

    // Init MPI_Request
    MPI_Request mpi_request[std::max(N_GPUs, max_pair_num)];

    // check_UVA(N_GPUs);        // 我理解，统一内存编址是为了方便，而不是性能

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


    // for (int m_id = 0; m_id < METHOD_NUM; ++ m_id) {
    // for (int src = 0; src < N_GPUs; ++ src) {
    // for (int dst = 0; dst < N_GPUs; ++ dst) {
        // if (src == dst) {
        //     continue;
        // }
        // if (true) {
        // // if (rank == 0) {
        //     printf("(%d, %d)\n", src, dst);
        // }
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
        // for (int i = 0; i < SIZES_LEN - 2; ++ i) {
        for (int i = SIZEIDX_START; i < SIZEIDX_END; ++ i) {
            LL SIZE = SIZES[i];
    #ifdef CHECK_RESULT
            SIZE = comm_size * comm_size;
    #endif 
            // const LL SSIZE = SIZE / comm_size;
            // const LL CHUNK_SIZE = SIZE / (comm_size * comm_size);
            int** send_buf_cpu = new int*[1];
            int** recv_buf_cpu = new int*[1];
            send_buf_cpu[0] = new int[SIZE];
            recv_buf_cpu[0] = new int[SIZE];
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
                // cudaMemcpyAsync(recv_buf[dst], send_buf[src], SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[0]);
                // cudaMemcpyAsync(recv_buf[src], send_buf[dst], SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[1]);
                // for (int k = 0; k < root[cp].size(); ++ k) {
                //     CUDA_CHECK(cudaMemcpyAsync(recv_buf[root[cp][k][1]], send_buf[root[cp][k][0]], 
                //                                SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[k]));
                // }
                XXX_comm(root[cp], send_buf_cpu, recv_buf, SIZE, streams, rank, comm, mpi_request);
                barrier(BACKEND, N_GPUs);
                // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
            }

            // CUDA_CHECK(cudaDeviceSynchronize());
            // MPI_Barrier(MPI_COMM_WORLD);
            // devicesSyncAll(N_GPUs);
            barrier(BACKEND, N_GPUs);

            // CUDA_CHECK(cudaEventRecord(start_a2a, stream));
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int _ = 0; _ < TIMES; ++ _) {
                // CUDA_CHECK(cudaMemcpy(send_buf[0], recv_buf[1], SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
                // cudaMemcpyAsync(recv_buf[dst], send_buf[src], SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[0]);
                // cudaMemcpyAsync(recv_buf[src], send_buf[dst], SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[1]);
                // for (int k = 0; k < root[cp].size(); ++ k) {
                //     CUDA_CHECK(cudaMemcpyAsync(recv_buf[root[cp][k][1].asInt()], send_buf[root[cp][k][0].asInt()], 
                //                                SIZE * sizeof(int), cudaMemcpyDeviceToDevice, streams[k]));
                // }
                XXX_comm(root[cp], send_buf_cpu, recv_buf, SIZE, streams, rank, comm, mpi_request);
                // CUDA_CHECK(cudaDeviceSynchronize());    // light-barrier, [WHY]: 会有性能提升！！！ 减少 comm contention ?
                // MPI_Barrier(MPI_COMM_WORLD);            // cpu-barrier, 没有意义
                // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
                barrier(BACKEND, N_GPUs);
            }
            // CUDA_CHECK(cudaEventRecord(stop_a2a, stream));
            // CUDA_CHECK(cudaEventSynchronize(stop_a2a));
            // still async !!!
            // CUDA_CHECK(cudaStreamSynchronize(stream));
            // CUDA_CHECK(cudaDeviceSynchronize());
            // MPI_Barrier(MPI_COMM_WORLD);
            // devicesSyncAll(N_GPUs);
            barrier(BACKEND, N_GPUs);

            auto t1 = std::chrono::high_resolution_clock::now();        // CORRECT
            // CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start_a2a, stop_a2a));    // ms
            // if (true) {
            if (rank == 0) {
                // double t_d = (double)elapsedTime / 1000;    // s
                double t_d = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / pow(1000, 2);  // s
                double calc = root[cp].size() * (double)SIZE * sizeof(int) * TIMES / pow(1024, 3);
                double avg_bd = calc / t_d;
                printf("time %lf s, REAL_BD %lf GB/s, SIZE %lf GB, comm_vol %lf GB\n", \
                        t_d, avg_bd, (double)SIZE * sizeof(int) / pow(1024, 3), calc);
#ifdef RECORD_TABLE
                // if (root[cp].size == )
                if (i + 1 == SIZEIDX_END) {
                    int src = root[cp][0][0].asInt();
                    int dst = root[cp][0][1].asInt();
                    if ((int)root[cp].size() == 1) {
                        result_table_si[src][dst] = avg_bd;
                    } else if ((int)root[cp].size() == 2 \
                            && src == (int)root[cp][1][1].asInt() && dst == (int)root[cp][1][0].asInt()) {
                        result_table_bi[src][dst] = avg_bd;
                    }
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
            // CUDA_CHECK(cudaFree(recv_buf));
            // CUDA_CHECK(cudaFree(send_buf));
            // delete[] output_list;
            // delete[] input_list;
            // delete[] output_list_cpu;
            // delete[] input_list_cpu;
            delete[] recv_buf_cpu[0];
            delete[] send_buf_cpu[0];
            delete[] recv_buf_cpu;
            delete[] send_buf_cpu;
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
    if (rank == 0) {
        Json::Value results;
        printf("P2P_SI: \n");
        for (int src = 0; src < N_GPUs; ++ src) {
            Json::Value list_;
            for (int dst = 0; dst < N_GPUs; ++ dst) {
                printf("%lf ", result_table_si[src][dst]);
                list_.append(Json::Value(result_table_si[src][dst]));
            }
            puts("");
            results["P2P_SI"].append(list_);
        }
        puts("");
        printf("P2P_BI: \n");
        for (int src = 0; src < N_GPUs; ++ src) {
            Json::Value list_;
            for (int dst = 0; dst < N_GPUs; ++ dst) {
                printf("%lf ", result_table_bi[src][dst]);
                list_.append(Json::Value(result_table_bi[src][dst]));
            }
            puts("");
            results["P2P_BI"].append(list_);
        }
        puts("");
        Json::StyledWriter sw;
        std::string output_file = "results/P2P_" + BACKEND + "_" + std::to_string(N_GPUs) + "_" + \
                                std::string(getenv("HOST")) + ".json";
        std::ofstream os(output_file.c_str(), std::ios::out);    // 覆盖写
        if (! os.is_open()) {
            std::cout << "error: can not find or create the file which named \" demo.json\"." << std::endl;
        }
        os << sw.write(results);
        os.close();
    }

#endif
    
    if (BACKEND.compare("cudaMemcpy") == 0) {
        disableP2P(N_GPUs);
    }
    for (int i = 0; i < STREAM_NUM; ++ i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    // MPI_Finalize();
    return 0;
}