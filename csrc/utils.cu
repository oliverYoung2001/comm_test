#include "utils.h"

void enableP2P(Json::Value& pairs) {
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

void enableP2P(int ngpus) {
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

void enableP2P(int i, int j) {
    if (i == j) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(i));
    int peer_access_available = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&peer_access_available, i, j));
    if (peer_access_available) {
        CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
    } else {
        printf("> GPU%d disabled direct access to GPU%d !!!\n", i, j);
        fflush(stdout);
    }  
}

void disableP2P(int i, int j) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(j));
}

void check_UVA(int ngpus) {
    for (int gpuid = 0; gpuid < ngpus; ++ gpuid) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuid));
        printf("GPU%d: %s unified addressing\n", gpuid, prop.unifiedAddressing ? "supports" : "does not support");
        fflush(stdout);
    }
}

// cudaMemcpy_comm: 不适用于多机
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
    MPI_Waitall(req_num, mpi_request , nullptr);

}

int parse_env(std::string key, std::string& value) {
    char* value0 = getenv(key.c_str());
    if (value0 == nullptr || strlen(value0) == 0) {
        return - 1;
    }
    value = std::string(value0);
    return 0;
}

int parse_env2int(std::string key, int& value) {
    std::string value_s;
    if (parse_env(key, value_s) != 0) {
        return - 1;
    }
    value = atoi(value_s.c_str());
    return 0;
}

void get_proc_params(PROC_PARAMS* pp) {
    // parse_env();
    parse_env("HOST", pp->host);
    parse_env2int("SLURM_PROCID", pp->rank);
    parse_env2int("SLURM_LOCALID", pp->local_rank);
    parse_env2int("SLURM_NTASKS", pp->comm_size);
    parse_env("SLURM_STEP_NODELIST", pp->ip);
    // hostname = socket.gethostname()
    // hostip = socket.gethostbyname(hostname)
    parse_env("SLURM_CLUSTER_NAME", pp->clustername);
    parse_env2int("SLURM_NODEID", pp->nodeid);
    parse_env("SLURMD_NODENAME", pp->nodename);
    parse_env2int("SLURM_TASKS_PER_NODE", pp->tasks_per_node);
    pp->nodes = pp->comm_size / pp->tasks_per_node;     // default = 1
}

void setup_env(PROC_PARAMS*& pp, int argc, char** argv) {
    assert(argc >= 3);

    //Get number of gpus in the node
    int N_GPUs = std::stoi(argv[1]);
    std::string BACKEND = argv[2];
    
    pp = new PROC_PARAMS(N_GPUs);
    pp->BACKEND = BACKEND;

    // Init MPI
    // int comm_size, rank;
    if (pp->BACKEND.compare("NCCL") == 0 || pp->BACKEND.compare("MPI") == 0) {
        MPI_Init(&argc, &argv);
        // MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &pp->comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &pp->rank);
        assert(pp->N_GPUs == pp->comm_size);
    }

    pp->local_rank = pp->rank;              // default
    pp->tasks_per_node = pp->comm_size;     // default

    get_proc_params(pp);
    if (pp->BACKEND.compare("NCCL") == 0 || pp->BACKEND.compare("MPI") == 0) {
        CUDA_CHECK(cudaSetDevice(pp->local_rank));      // 至关重要！！！
    }

    // Init NCCL
    if (pp->BACKEND.compare("NCCL") == 0) {
        ncclUniqueId id;
        if (pp->rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclCommInitRank(&pp->comm, pp->comm_size, id, pp->rank);
    }
    if (pp->BACKEND.find("cudaMemcpy") != std::string::npos) {
        pp->comm_size = 0;
        pp->rank = 0;
    }

    int GPU_VISIBLE;
    CUDA_CHECK(cudaGetDeviceCount(&GPU_VISIBLE));
    assert(pp->tasks_per_node <= GPU_VISIBLE);
    assert(pp->N_GPUs <= GPU_VISIBLE * pp->nodes);

    if (pp->rank == 0) {
        printf("BACKEND: %s\n", pp->BACKEND.c_str());
        fflush(stdout);
    }

    parse_env("RECORD_P2P", pp->RECORD_P2P);

    if (pp->BACKEND.find("-P") != std::string::npos) {      // found "-P", ENABLE
        pp->ENABLE_GPU_P2P = true;
    }
}
