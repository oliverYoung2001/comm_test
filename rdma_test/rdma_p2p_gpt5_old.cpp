// gdr_p2p.cpp
// Single-process GPUDirect RDMA demo:
// GPU0 (src) --[mlx5_3]--> [mlx5_4] -- GPU1 (dst) via RC QP and RDMA WRITE.
// Build: see build.sh
// Run:   see run.sh

#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#define CHECK_CUDA(cmd) do {                                 \
    cudaError_t e = (cmd);                                   \
    if (e != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(e));  \
        exit(1);                                             \
    }                                                        \
} while(0)

static void die(const char* msg) {
    perror(msg);
    exit(1);
}

struct Endpoint {
    std::string dev_name;
    int         port_num = 1;       // HCA port
    int         gid_index = 0;      // GID index (RoCE v2 usually 0 or as configured)

    ibv_context* ctx = nullptr;
    ibv_pd*      pd  = nullptr;
    ibv_cq*      cq  = nullptr;
    ibv_qp*      qp  = nullptr;

    ibv_port_attr port_attr{};
    ibv_gid       gid{};
    uint32_t      qp_num = 0;

    void*         gpu_buf = nullptr;
    size_t        buf_bytes = 0;
    ibv_mr*       mr = nullptr;

    // open device by name (e.g., "mlx5_3")
    void open_by_name(const char* want) {
        int num = 0;
        ibv_device** list = ibv_get_device_list(&num);
        if (!list || num == 0) die("ibv_get_device_list");

        for (int i = 0; i < num; ++i) {
            const char* name = ibv_get_device_name(list[i]);
            if (name && strcmp(name, want) == 0) {
                ctx = ibv_open_device(list[i]);
                if (!ctx) die("ibv_open_device");
                dev_name = want;
                break;
            }
        }
        ibv_free_device_list(list);
        if (!ctx) {
            fprintf(stderr, "RDMA device %s not found.\n", want);
            exit(1);
        }
    }

    void create_basic() {
        if (ibv_query_port(ctx, port_num, &port_attr))
            die("ibv_query_port");

        if (ibv_query_gid(ctx, port_num, gid_index, &gid))
            die("ibv_query_gid");

        pd = ibv_alloc_pd(ctx);
        if (!pd) die("ibv_alloc_pd");

        // Create CQ (both send & recv on one CQ)
        cq = ibv_create_cq(ctx, /*cqe*/16, nullptr, nullptr, 0);
        if (!cq) die("ibv_create_cq");

        // Create RC QP
        ibv_qp_init_attr qpia{};
        qpia.send_cq = cq;
        qpia.recv_cq = cq;
        qpia.qp_type = IBV_QPT_RC;
        qpia.cap.max_send_wr  = 16;
        qpia.cap.max_recv_wr  = 16;
        qpia.cap.max_send_sge = 1;
        qpia.cap.max_recv_sge = 1;

        qp = ibv_create_qp(pd, &qpia);
        if (!qp) die("ibv_create_qp");
        qp_num = qp->qp_num;

        // INIT
        ibv_qp_attr attr{};
        attr.qp_state        = IBV_QPS_INIT;
        attr.port_num        = port_num;
        attr.pkey_index      = 0;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;

        int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        if (ibv_modify_qp(qp, &attr, mask)) die("ibv_modify_qp INIT");
    }

    void alloc_and_reg_cuda(int cuda_dev, size_t bytes) {
        buf_bytes = bytes;
        CHECK_CUDA(cudaSetDevice(cuda_dev));
        CHECK_CUDA(cudaMalloc(&gpu_buf, buf_bytes));
        // For demo: fill pattern on source, clear on dest由调用方完成

        int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        mr = ibv_reg_mr(pd, gpu_buf, buf_bytes, acc);
        if (!mr) die("ibv_reg_mr (GPUDirect) failed. Is nvidia-peermem loaded?");
    }

    // connect this->qp to remote (RC)
    void connect_rc(const Endpoint& remote) {
        // RTR
        ibv_qp_attr attr{};
        memset(&attr, 0, sizeof(attr));
        attr.qp_state           = IBV_QPS_RTR;
        attr.path_mtu           = IBV_MTU_1024;
        attr.dest_qp_num        = remote.qp_num;
        attr.rq_psn             = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer      = 12;

        attr.ah_attr.is_global      = 1;
        attr.ah_attr.grh.hop_limit  = 64;
        attr.ah_attr.grh.dgid       = remote.gid;
        attr.ah_attr.grh.sgid_index = gid_index; // use our sgid index
        attr.ah_attr.sl             = 0;
        attr.ah_attr.src_path_bits  = 0;
        attr.ah_attr.port_num       = port_num;

        int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                   IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                   IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

        if (ibv_modify_qp(qp, &attr, mask)) die("ibv_modify_qp RTR");

        // RTS
        memset(&attr, 0, sizeof(attr));
        attr.qp_state      = IBV_QPS_RTS;
        attr.sq_psn        = 0;
        attr.timeout       = 14;
        attr.retry_cnt     = 7;
        attr.rnr_retry     = 7; // infinite
        attr.max_rd_atomic = 1;

        mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
               IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

        if (ibv_modify_qp(qp, &attr, mask)) die("ibv_modify_qp RTS");
    }

    void post_rdma_write(const Endpoint& remote, size_t bytes) {
        ibv_sge sge{};
        sge.addr   = (uintptr_t)gpu_buf;
        sge.length = (uint32_t)bytes;
        sge.lkey   = mr->lkey;

        ibv_send_wr wr{};
        wr.wr_id      = 0xBEEF;
        wr.opcode     = IBV_WR_RDMA_WRITE;
        wr.sg_list    = &sge;
        wr.num_sge    = 1;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = (uintptr_t)remote.gpu_buf;
        wr.wr.rdma.rkey        = remote.mr->rkey;

        ibv_send_wr* bad = nullptr;
        if (ibv_post_send(qp, &wr, &bad)) die("ibv_post_send");
    }

    void poll_one_wc() {
        ibv_wc wc{};
        while (true) {
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) die("ibv_poll_cq");
            if (n == 0) { usleep(1000); continue; }
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "WC error: status=%d, opcode=%d\n", wc.status, wc.opcode);
                exit(1);
            }
            break;
        }
    }
};

static uint64_t parse_size(const std::string& s) {
    // supports bytes, KB/MB/GB suffix
    char unit = 0;
    double val = 0.0;
    if (sscanf(s.c_str(), "%lf%c", &val, &unit) < 1) {
        throw std::runtime_error("Bad size");
    }
    uint64_t bytes = 0;
    switch (unit) {
        case 'G': case 'g': bytes = (uint64_t)(val * (1024ULL*1024ULL*1024ULL)); break;
        case 'M': case 'm': bytes = (uint64_t)(val * (1024ULL*1024ULL)); break;
        case 'K': case 'k': bytes = (uint64_t)(val * (1024ULL)); break;
        default:            bytes = (uint64_t)val; break; // assume bytes
    }
    return bytes;
}

int main(int argc, char** argv) {
    std::string dev_a = "mlx5_3";
    std::string dev_b = "mlx5_4";
    int port_a = 1, port_b = 1;
    int gid_idx_a = 0, gid_idx_b = 0;
    int gpu_src = 0, gpu_dst = 1;
    uint64_t bytes = parse_size("64M");

    // args
    for (int i=1; i<argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* err){ if (i+1>=argc) { fprintf(stderr, "Missing value for %s\n", err); exit(1);} return std::string(argv[++i]); };
        if (a == "--src-dev") dev_a = need("--src-dev");
        else if (a == "--dst-dev") dev_b = need("--dst-dev");
        else if (a == "--src-port") port_a = std::stoi(need("--src-port"));
        else if (a == "--dst-port") port_b = std::stoi(need("--dst-port"));
        else if (a == "--src-gid-idx") gid_idx_a = std::stoi(need("--src-gid-idx"));
        else if (a == "--dst-gid-idx") gid_idx_b = std::stoi(need("--dst-gid-idx"));
        else if (a == "--src-gpu") gpu_src = std::stoi(need("--src-gpu"));
        else if (a == "--dst-gpu") gpu_dst = std::stoi(need("--dst-gpu"));
        else if (a == "--size") bytes = parse_size(need("--size"));
        else if (a == "--help" || a == "-h") {
            printf(
                "Usage: %s [--src-dev mlx5_3] [--dst-dev mlx5_4] [--src-port 1] [--dst-port 1]\n"
                "          [--src-gid-idx 0] [--dst-gid-idx 0]\n"
                "          [--src-gpu 0] [--dst-gpu 1] [--size 64M]\n", argv[0]);
            return 0;
        }
    }

    printf("Config:\n");
    printf("  SRC dev=%s port=%d gid_idx=%d gpu=%d\n", dev_a.c_str(), port_a, gid_idx_a, gpu_src);
    printf("  DST dev=%s port=%d gid_idx=%d gpu=%d\n", dev_b.c_str(), port_b, gid_idx_b, gpu_dst);
    printf("  size=%lu bytes\n", (unsigned long)bytes);

    Endpoint A, B;
    A.port_num = port_a; A.gid_index = gid_idx_a;
    B.port_num = port_b; B.gid_index = gid_idx_b;

    // Open devices
    A.open_by_name(dev_a.c_str());
    B.open_by_name(dev_b.c_str());

    // Create resources
    A.create_basic();
    B.create_basic();

    // Allocate CUDA buffers and register MRs
    A.alloc_and_reg_cuda(gpu_src, bytes);
    B.alloc_and_reg_cuda(gpu_dst, bytes);

    // Initialize data on GPU0, clear on GPU1
    CHECK_CUDA(cudaSetDevice(gpu_src));
    CHECK_CUDA(cudaMemset(A.gpu_buf, 0xAB, bytes));
    CHECK_CUDA(cudaSetDevice(gpu_dst));
    CHECK_CUDA(cudaMemset(B.gpu_buf, 0x00, bytes));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Bring both QPs up (A<->B)
    A.connect_rc(B);
    B.connect_rc(A);

    // Post RDMA WRITE on A -> B
    A.post_rdma_write(B, bytes);
    A.poll_one_wc();

    // Validate by copying a small slice back to host from GPU1
    const size_t chk = std::min<uint64_t>(bytes, 4096);
    std::vector<unsigned char> host(chk);
    CHECK_CUDA(cudaSetDevice(gpu_dst));
    CHECK_CUDA(cudaMemcpy(host.data(), B.gpu_buf, chk, cudaMemcpyDeviceToHost));

    size_t bad = 0;
    for (size_t i=0;i<chk;i++) if (host[i] != 0xAB) { bad=i; break; }
    if (bad==0) {
        std::cout << "RDMA WRITE OK. First " << chk << " bytes match 0xAB." << std::endl;
    } else {
        std::cerr << "Data mismatch at offset " << bad << ": got 0x"
                  << std::hex << (int)host[bad] << " expected 0xAB\n";
        return 2;
    }

    std::cout << "Done." << std::endl;
    return 0;
}