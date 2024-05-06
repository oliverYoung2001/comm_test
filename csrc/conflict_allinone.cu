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

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Need at least 4 args: \"<command> <gpus> <backend> <cp_file>\"\n");
        return - 1;
    }
    setup_env(pp, argc, argv);
    std::string cp_file = argv[3];

    void (*XXX_comm)(PROC_PARAMS*& pp, Json::Value& pairs, int** send_buf, int** recv_buf, LL SIZE, \
               cudaStream_t* streams, int rank, ncclComm_t comm, MPI_Request* mpi_request);
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

#ifdef PRINT_JSON
    Json::Value root;
#endif

    double result_table_si[pp->N_GPUs][pp->N_GPUs];
    double result_table_bi[pp->N_GPUs][pp->N_GPUs];
    memset(result_table_si, 0, sizeof(result_table_si));
    memset(result_table_bi, 0, sizeof(result_table_bi));

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
        if (pp->BACKEND.find("cudaMemcpy") != std::string::npos && pp->ENABLE_GPU_P2P) {
            enableP2P(root[cp]);
        }
        for (int i = SIZEIDX_START; i < SIZEIDX_END; ++ i) {
            LL SIZE = SIZES[i];
    #ifdef CHECK_RESULT
            SIZE = comm_size * comm_size;
    #endif 
            // const LL SSIZE = SIZE / comm_size;
            // const LL CHUNK_SIZE = SIZE / (comm_size * comm_size);
            int** send_buf = new int*[pp->N_GPUs];
            int** recv_buf = new int*[pp->N_GPUs];
    
    #ifdef DIFF_BUF
            if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
                for (int gpuid = 0; gpuid < pp->N_GPUs; ++ gpuid) {
                    int local_send_buf_cnt = 0;
                    int local_recv_buf_cnt = 0;
                    for (int k = 0; k < root[cp].size(); ++ k) {
                        if (root[cp][k][0].asInt() == gpuid) {
                            ++ local_send_buf_cnt;
                        }
                        if (root[cp][k][1].asInt() == gpuid) {
                            ++ local_recv_buf_cnt;
                        }
                    }
                    CUDA_CHECK(cudaSetDevice(gpuid));
                    CUDA_CHECK(cudaMalloc(&send_buf[gpuid], local_send_buf_cnt * SIZE * sizeof(int)));
                    CUDA_CHECK(cudaMalloc(&recv_buf[gpuid], local_recv_buf_cnt * SIZE * sizeof(int)));
                }
            }
            if (pp->BACKEND.compare("NCCL") == 0 || pp->BACKEND.compare("MPI") == 0) {
                int local_send_buf_cnt = 0;
                int local_recv_buf_cnt = 0;
                for (int k = 0; k < root[cp].size(); ++ k) {
                    if (root[cp][k][0].asInt() == pp->rank) {
                        ++ local_send_buf_cnt;
                    }
                    if (root[cp][k][1].asInt() == pp->rank) {
                        ++ local_recv_buf_cnt;
                    }
                }
                CUDA_CHECK(cudaMalloc(&send_buf[pp->rank], local_send_buf_cnt * SIZE * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&recv_buf[pp->rank], local_recv_buf_cnt * SIZE * sizeof(int)));
            }
    #else
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
    #endif
            

    #ifdef CHECK_RESULT
            TIMES = 1;
            for (int j = 0; j < comm_size; ++ j) {
                int v = pp->rank * comm_size + j;
                for (int k = 0; k < CHUNK_SIZE; ++ k) {
                    input_list_cpu[j][k] = v;
                }
            }
            printf("pp->rank %d, send_buf_cpu:\n", pp->rank);
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
                XXX_comm(pp, root[cp], send_buf, recv_buf, SIZE, pp->streams, pp->rank, pp->comm, pp->mpi_requests);
                barrier(pp->BACKEND, pp->N_GPUs);
                // devicesSyncAll(N_GPUs);                 // barrier(= light-barrier + cpu-barrier)
            }

            // CUDA_CHECK(cudaDeviceSynchronize());
            // MPI_Barrier(MPI_COMM_WORLD);
            // devicesSyncAll(N_GPUs);
            barrier(pp->BACKEND, pp->N_GPUs);

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
                XXX_comm(pp, root[cp], send_buf, recv_buf, SIZE, pp->streams, pp->rank, pp->comm, pp->mpi_requests);
                barrier(pp->BACKEND, pp->N_GPUs);
                // CUDA_CHECK(cudaDeviceSynchronize());    // light-barrier, [WHY]: 会有性能提升！！！ 减少 comm contention ?
                // MPI_Barrier(MPI_COMM_WORLD);            // cpu-barrier, 没有意义
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
                double calc = root[cp].size() * (double)SIZE * sizeof(int) * TIMES;      // B
                double avg_bd = calc / t_d / pow(1024, 3);
                printf("time %lf s, REAL_BD %lf GB/s, SIZE %lf KB, comm_vol %lf KB\n", \
                        t_d, avg_bd, (double)SIZE * sizeof(int) / pow(1024, 1), calc / pow(1024, 1));
            if (pp->RECORD_P2P.size() > 0) {
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
            }
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
            printf("pp->rank %d, recv_buf_cpu:\n", pp->rank);
            for (int j = 0; j < SSIZE; ++ j) {
                printf("%d ", recv_buf_cpu[j]);
            }
            puts("");
            for (int j = 0; j < comm_size; ++ j) {
                int v = j * comm_size + pp->rank;
                for (int k = 0; k < CHUNK_SIZE; ++ k) {
                    if (output_list_cpu[j][k] != v) {
                        printf("Failed: Comm(All2All) error %s:%d '%d!=%d'\n",             \
                                __FILE__,__LINE__,output_list_cpu[j][k],v);   \
                        exit(EXIT_FAILURE); 
                    }
                    // assert(output_list[j][k] == v)
                }
            }
            printf("pp->rank %d: ALL2ALL CORRECT !!!\n", pp->rank);
            fflush(stdout);
    #endif
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
            disableP2P(root[cp]);
        }
    }

#ifdef PRINT_JSON
    if (pp->rank == 0) {
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
    if (pp->RECORD_P2P.size() > 0) {
        if (pp->rank == 0) {
            Json::Value results;
            printf("P2P_SI: \n");
            for (int src = 0; src < pp->N_GPUs; ++ src) {
                Json::Value list_;
                for (int dst = 0; dst < pp->N_GPUs; ++ dst) {
                    printf("%lf ", result_table_si[src][dst]);
                    list_.append(Json::Value(result_table_si[src][dst]));
                }
                puts("");
                results["P2P_SI"].append(list_);
            }
            puts("");
            printf("P2P_BI: \n");
            for (int src = 0; src < pp->N_GPUs; ++ src) {
                Json::Value list_;
                for (int dst = 0; dst < pp->N_GPUs; ++ dst) {
                    printf("%lf ", result_table_bi[src][dst]);
                    list_.append(Json::Value(result_table_bi[src][dst]));
                }
                puts("");
                results["P2P_BI"].append(list_);
            }
            puts("");
            Json::StyledWriter sw;
            std::string output_file = "results/P2P_" + pp->BACKEND + "_" + std::to_string(pp->N_GPUs) + "_" + \
                                    std::string(getenv("HOST")) + ".json";
            std::ofstream os(output_file.c_str(), std::ios::out);    // 覆盖写
            if (! os.is_open()) {
                std::cout << "error: can not find or create the file which named \" demo.json\"." << std::endl;
            }
            os << sw.write(results);
            os.close();
        }
    }

    delete pp;
    // MPI_Finalize();
    return 0;
}