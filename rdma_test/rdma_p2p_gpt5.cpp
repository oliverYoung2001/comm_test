// gdr_p2p_auto.cpp
// Auto-detect IB (LID) vs RoCE (v1/v2) and pick addressing automatically.
// Single-process demo: GPU0 --[mlx5_3]--> [mlx5_4] -- GPU1 via RC QP + RDMA WRITE.

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <unistd.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
// === Benchmark utils ===
#include <chrono>
using clk = std::chrono::steady_clock;

struct BenchRes {
    double gbps = 0.0;
    double avg_us = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
};

static inline uint64_t ns_since(const clk::time_point& t0, const clk::time_point& t1){
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

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

static std::string read_text_file(const std::string& p) {
    std::ifstream f(p);
    if (!f.is_open()) return "";
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

static void print_gid(const char* tag, const ibv_gid& g) {
    char buf[64] = {0};
    inet_ntop(AF_INET6, g.raw, buf, sizeof(buf));
    printf("%s GID=%s\n", tag, buf[0]?buf:"<none>");
}

static void assert_port_active(ibv_context* ctx, int port) {
    ibv_port_attr pa{};
    if (ibv_query_port(ctx, port, &pa)) die("ibv_query_port");
    printf("  port %d state=%u (4=ACTIVE), LID=0x%x\n", port, pa.state, pa.lid);
    if (pa.state != IBV_PORT_ACTIVE) {
        fprintf(stderr, "Port not ACTIVE.\n");
        exit(1);
    }
}

enum class AddrMode {
    IB_LID,      // InfiniBand with LID routing (non-global AV)
    ROCE_V1,     // Ethernet RoCE v1 (global AV with GID)
    ROCE_V2      // Ethernet RoCE v2 (global AV with GID)
};

struct Endpoint {
    std::string dev_name = "mlx5_3";
    int         port_num = 1;

    // If user wants to override gid idx; <0 means auto
    int         gid_index_override = -1;

    ibv_context* ctx = nullptr;
    ibv_pd*      pd  = nullptr;
    ibv_cq*      cq  = nullptr;
    ibv_qp*      qp  = nullptr;

    ibv_port_attr port_attr{};
    ibv_gid       gid{};       // only used for RoCE
    int           gid_index = -1;

    AddrMode      mode = AddrMode::IB_LID;

    uint32_t      qp_num = 0;
    void*         gpu_buf = nullptr;
    size_t        buf_bytes = 0;
    ibv_mr*       mr = nullptr;

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

    static std::string link_layer_sysfs(const std::string& dev, int port) {
        char p[256];
        snprintf(p, sizeof(p), "/sys/class/infiniband/%s/ports/%d/link_layer", dev.c_str(), port);
        std::string s = read_text_file(p);
        // Typically "InfiniBand\n" or "Ethernet\n"
        s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
        return s;
    }

    static std::string gid_type_sysfs(const std::string& dev, int port, int idx) {
        char p[300];
        snprintf(p, sizeof(p), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", dev.c_str(), port, idx);
        std::string s = read_text_file(p);
        s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
        return s; // "", "ib", "roce v1", "roce v2"
    }

    void detect_mode_and_pick_gid() {
        // Query basic port
        if (ibv_query_port(ctx, port_num, &port_attr)) die("ibv_query_port");
        assert_port_active(ctx, port_num);

        const std::string ll = link_layer_sysfs(dev_name, port_num);
        printf("[%s] link_layer=%s\n", dev_name.c_str(), ll.c_str());   // InfiniBand

        if (gid_index_override >= 0) {  // False
            // User overrides -> just use RoCE with that gid
            if (ibv_query_gid(ctx, port_num, gid_index_override, &gid)) die("ibv_query_gid");
            gid_index = gid_index_override;
            // try to detect type for printing
            std::string typ = gid_type_sysfs(dev_name, port_num, gid_index);
            if (typ.find("v1") != std::string::npos) mode = AddrMode::ROCE_V1;
            else if (typ.find("v2") != std::string::npos) mode = AddrMode::ROCE_V2;
            else if (ll == "InfiniBand") mode = AddrMode::IB_LID; // override may be meaningless on IB
            else mode = AddrMode::ROCE_V2;
            print_gid((dev_name + " (override)").c_str(), gid);
            return;
        }

        if (ll == "InfiniBand") {   // True
            // Pure IB: prefer LID routing (not global)
            mode = AddrMode::IB_LID;
            gid_index = -1; // not used
            memset(&gid, 0, sizeof(gid));
            printf("[%s] Using IB LID addressing (no GID needed)\n", dev_name.c_str());
            return;
        }

        // Ethernet: RoCE. Prefer v1 if available (per你的网络环境偏好)
        int picked = -1;
        ibv_gid tmp{};
        auto try_pick = [&](const char* want) -> bool {
            for (int i = 0; i < 128; ++i) {
                std::string typ = gid_type_sysfs(dev_name, port_num, i);
                if (typ.empty()) break; // end of list for most systems
                if (want && typ != want) continue;
                if (ibv_query_gid(ctx, port_num, i, &tmp)) continue;
                picked = i;
                gid = tmp;
                return true;
            }
            return false;
        };

        if (try_pick("roce v1")) { mode = AddrMode::ROCE_V1; gid_index = picked; }
        else if (try_pick("roce v2")) { mode = AddrMode::ROCE_V2; gid_index = picked; }
        else {
            // Any first valid GID as last resort
            for (int i = 0; i < 128; ++i) {
                if (ibv_query_gid(ctx, port_num, i, &tmp)) break;
                gid = tmp; picked = i; mode = AddrMode::ROCE_V2; break;
            }
            gid_index = picked;
        }

        if (gid_index < 0) {
            fprintf(stderr, "[%s] No valid GID found on port %d.\n", dev_name.c_str(), port_num);
            exit(1);
        }
        std::string typ = gid_type_sysfs(dev_name, port_num, gid_index);
        printf("[%s] Picked GID index=%d, type=%s\n", dev_name.c_str(), gid_index, typ.c_str());
        print_gid(dev_name.c_str(), gid);
    }

    void create_qp_pd_cq() {    // queue_pair, protection domain, completion queue
        pd = ibv_alloc_pd(ctx);
        if (!pd) die("ibv_alloc_pd");

        int want_cqe = 8192;  // 够用的上限
        cq = ibv_create_cq(ctx, /*cqe*/want_cqe, nullptr, nullptr, 0);
        if (!cq) die("ibv_create_cq");

        ibv_qp_init_attr qpia{};
        qpia.send_cq = cq;
        qpia.recv_cq = cq;
        qpia.qp_type = IBV_QPT_RC;
        qpia.cap.max_send_wr  = 64;
        qpia.cap.max_recv_wr  = 64;
        qpia.cap.max_send_sge = 1;
        qpia.cap.max_recv_sge = 1;

        qp = ibv_create_qp(pd, &qpia);
        if (!qp) die("ibv_create_qp");
        qp_num = qp->qp_num;

        ibv_qp_attr attr{};
        memset(&attr, 0, sizeof(attr));
        attr.qp_state        = IBV_QPS_INIT;
        attr.port_num        = port_num;
        attr.pkey_index      = 0;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

        int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        if (ibv_modify_qp(qp, &attr, mask)) die("ibv_modify_qp INIT");
    }

    void alloc_and_reg_cuda(int cuda_dev, size_t bytes) {
        buf_bytes = bytes;
        CHECK_CUDA(cudaSetDevice(cuda_dev));
        CHECK_CUDA(cudaMalloc(&gpu_buf, buf_bytes));

        int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        mr = ibv_reg_mr(pd, gpu_buf, buf_bytes, acc);   // mr: memory region
        if (!mr) die("ibv_reg_mr (GPUDirect) failed. Is nvidia-peermem loaded?");
    }

    void connect_rc(const Endpoint& remote) {   // rc: reliable connection
        // RTR
        ibv_qp_attr attr{};
        memset(&attr, 0, sizeof(attr));
        attr.qp_state           = IBV_QPS_RTR;
        attr.path_mtu           = IBV_MTU_512; // 保守；必要时可调 1024/2048
        attr.dest_qp_num        = remote.qp_num;
        attr.rq_psn             = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer      = 12;

        attr.ah_attr.port_num   = port_num;
        attr.ah_attr.sl         = 0;
        attr.ah_attr.src_path_bits = 0;

        if (mode == AddrMode::IB_LID) { // True
            // IB LID addressing：非全局，使用对端 LID
            attr.ah_attr.is_global = 0;
            attr.ah_attr.dlid      = remote.port_attr.lid;
        } else {
            // RoCE v1/v2：全局 AV + GID
            attr.ah_attr.is_global      = 1;
            attr.ah_attr.grh.hop_limit  = 64;
            attr.ah_attr.grh.dgid       = remote.gid;
            attr.ah_attr.grh.sgid_index = gid_index;
            // 可选: traffic_class / flow_label 按需设置
        }

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

        ibv_send_wr wr{};   // wr: work request
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
                fprintf(stderr, "WC error: status=%d(%s), opcode=%d, vendor_err=0x%x\n",
                        wc.status, ibv_wc_status_str(wc.status), wc.opcode, wc.vendor_err);
                exit(1);
            }
            break;
        }
    }
};

static uint64_t parse_size(const std::string& s) {
    char unit = 0;
    double val = 0.0;
    if (sscanf(s.c_str(), "%lf%c", &val, &unit) < 1) {
        fprintf(stderr, "Bad size string: %s\n", s.c_str());
        exit(1);
    }
    uint64_t bytes = 0;
    switch (unit) {
        case 'G': case 'g': bytes = (uint64_t)(val * (1024ULL*1024ULL*1024ULL)); break;
        case 'M': case 'm': bytes = (uint64_t)(val * (1024ULL*1024ULL)); break;
        case 'K': case 'k': bytes = (uint64_t)(val * (1024ULL)); break;
        default:            bytes = (uint64_t)val; break;
    }
    return bytes;
}


// 直接贴在你的文件里（与 Endpoint 同级或下方皆可）
static inline int post_rdma_write_bytes(ibv_qp* qp, ibv_mr* lmr, void* laddr,   // A.qp, A.mr(useless), A.gpu_buf
                                    uint32_t lkey, uintptr_t raddr, uint32_t rkey,  // A.mr->lkey, (uintptr_t)B.gpu_buf, B.mr->rkey,
                                    size_t bytes, bool signaled, uint64_t wrid) // bytes, signaled, 0x2000+i
{
    ibv_sge sge{};
    sge.addr   = (uintptr_t)laddr;
    sge.length = (uint32_t)bytes;
    sge.lkey   = lkey;

    ibv_send_wr wr{};
    wr.wr_id      = wrid;
    wr.opcode     = IBV_WR_RDMA_WRITE;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey        = rkey;

    ibv_send_wr* bad = nullptr;
    return ibv_post_send(qp, &wr, &bad);
}

// 参数说明：
//   bytes         本次 RDMA WRITE 的字节数
//   iters         统计迭代次数（不含预热）
//   warmup        预热次数（不计入统计）
//   inflight      允许未完成的最大 WR 数（软件 pipeline）
//   signal_every  每多少个 WR 打一次 CQE（例如 32）
// 返回：吞吐 GB/s、平均/50分位/99分位 单次写入时延（微秒）
static BenchRes bench_one_size(Endpoint& A, const Endpoint& B,
                               size_t bytes, int iters, int warmup,
                               int inflight, int signal_every)
{
    // 预热
    {
        int outstanding = 0;
        for (int i=0;i<warmup;i++){
            bool signaled = (++outstanding % signal_every) == 0;
            if (post_rdma_write_bytes(A.qp, A.mr, A.gpu_buf, A.mr->lkey,
                                      (uintptr_t)B.gpu_buf, B.mr->rkey,
                                      bytes, signaled, 0x1000+i))
                die("ibv_post_send warmup");
            if (signaled) {
                // poll 1 CQE
                ibv_wc wc{};
                while (ibv_poll_cq(A.cq, 1, &wc)==0) { /* spin */ }
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "Warmup WC error: %d(%s)\n",
                            wc.status, ibv_wc_status_str(wc.status));
                    exit(1);
                }
                outstanding = 0;
            } else if (outstanding >= inflight) {
                // 强制打一条有信号的以回收 CQ
                if (post_rdma_write_bytes(A.qp, A.mr, A.gpu_buf, A.mr->lkey,
                                          (uintptr_t)B.gpu_buf, B.mr->rkey,
                                          1, true, 0x1ABC))
                    die("ibv_post_send warmup-drain");
                ibv_wc wc{};
                while (ibv_poll_cq(A.cq, 1, &wc)==0) { /* spin */ }
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "Warmup drain WC error: %d(%s)\n",
                            wc.status, ibv_wc_status_str(wc.status));
                    exit(1);
                }
                outstanding = 0;
            }
        }
        // 清尾
        if (post_rdma_write_bytes(A.qp, A.mr, A.gpu_buf, A.mr->lkey,
                                  (uintptr_t)B.gpu_buf, B.mr->rkey,
                                  1, true, 0x1DEF))
            die("ibv_post_send warmup-final");
        ibv_wc wc{};
        while (ibv_poll_cq(A.cq, 1, &wc)==0) {}
        if (wc.status != IBV_WC_SUCCESS) { fprintf(stderr, "Warmup final WC error\n"); exit(1); }
    }

    // 正式测试：收集每次“提交→对应 CQE”的往返时间（软件层）
    std::vector<double> us_each;
    us_each.reserve(iters);

    int outstanding = 0;
    int need_cqe = 0;
    auto t_begin = clk::now();

    for (int i=0;i<iters;i++) {
        auto t0 = clk::now();
        bool signaled = (++outstanding % signal_every) == 0;
        if (post_rdma_write_bytes(A.qp, A.mr, A.gpu_buf, A.mr->lkey,
                                  (uintptr_t)B.gpu_buf, B.mr->rkey,
                                  bytes, signaled, 0x2000+i)) {
            fprintf(stderr, "ibv_post_send bench failed: errno=%d (%s)\n",
                errno, strerror(errno));
            die("ibv_post_send bench");
        }

        if (signaled) need_cqe++;

        // 如果 outstanding 达到上限，先回收一个 CQE（打断点控深度）
        if (outstanding >= inflight) {
            ibv_wc wc{};
            while (ibv_poll_cq(A.cq, 1, &wc)==0) { /* spin */ }
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "Bench WC error: %d(%s)\n",
                        wc.status, ibv_wc_status_str(wc.status));
                exit(1);
            }
            auto t1 = clk::now();
            us_each.push_back(ns_since(t0, t1) / 1000.0);
            outstanding = 0;
            need_cqe--; // 刚收了一个
        }
    }

    // 把剩余 CQE 收完，同时记录时间
    while (need_cqe > 0) {
        ibv_wc wc{};
        clk::time_point t0 = clk::now(); // 这里没有精确对应到哪一个 WR 的 t0
        while (ibv_poll_cq(A.cq, 1, &wc)==0) {}
        clk::time_point t1 = clk::now();
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "Tail WC error: %d(%s)\n", wc.status, ibv_wc_status_str(wc.status));
            exit(1);
        }
        us_each.push_back(ns_since(t0, t1) / 1000.0); // 作为估算值
        need_cqe--;
    }

    auto t_end = clk::now();
    double total_s = ns_since(t_begin, t_end) / 1e9;
    double total_bytes = (double)bytes * iters;
    BenchRes r{};
    r.gbps = (total_bytes / total_s) / 1e9;

    // 统计延迟
    if (!us_each.empty()) {
        std::sort(us_each.begin(), us_each.end());
        double sum = 0.0; for (double x : us_each) sum += x;
        r.avg_us = sum / us_each.size();
        r.p50_us = us_each[us_each.size()/2];
        r.p99_us = us_each[(size_t)std::max<size_t>(0, (int)(us_each.size()*0.99)-1)];
    }
    return r;
}

// 扫描一组 message size 并打印 CSV（size_bytes, GB/s, avg_us, p50_us, p99_us）
static void sweep_and_print_csv(Endpoint& A, const Endpoint& B,
                                const std::vector<size_t>& sizes,
                                int iters, int warmup, int inflight, int signal_every)
{
    printf("size_bytes,gbps,avg_us,p50_us,p99_us\n");
    for (size_t sz : sizes) {
        auto res = bench_one_size(A, B, sz, iters, warmup, inflight, signal_every);
        printf("%zu,%.6f,%.3f,%.3f,%.3f\n",
               sz, res.gbps, res.avg_us, res.p50_us, res.p99_us);
        fflush(stdout);
    }
}

int main(int argc, char** argv) {
    std::string dev_a = "mlx5_3";
    std::string dev_b = "mlx5_4";
    int port_a = 1, port_b = 1;
    int gpu_src = 0, gpu_dst = 1;
    int override_gid_a = -1, override_gid_b = -1;
    uint64_t bytes = parse_size("64M");

    // [NOTE]: physical port number on HCA. One RDMA NIC has 1/2 ports, numbered from 1.
    // On bingxing, seen by command `ibv_devinfo`, each HCA only has one port, numbered as 1.
    for (int i=1; i<argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* err){ if (i+1>=argc) { fprintf(stderr, "Missing value for %s\n", err); exit(1);} return std::string(argv[++i]); };
        if (a == "--src-dev") dev_a = need("--src-dev");
        else if (a == "--dst-dev") dev_b = need("--dst-dev");
        else if (a == "--src-port") port_a = std::stoi(need("--src-port"));
        else if (a == "--dst-port") port_b = std::stoi(need("--dst-port"));
        else if (a == "--src-gpu") gpu_src = std::stoi(need("--src-gpu"));
        else if (a == "--dst-gpu") gpu_dst = std::stoi(need("--dst-gpu"));
        else if (a == "--size") bytes = parse_size(need("--size"));
        else if (a == "--src-gid-idx") override_gid_a = std::stoi(need("--src-gid-idx"));
        else if (a == "--dst-gid-idx") override_gid_b = std::stoi(need("--dst-gid-idx"));
        else if (a == "--help" || a == "-h") {
            printf(
                "Usage: %s [--src-dev mlx5_3] [--dst-dev mlx5_4] [--src-port 1] [--dst-port 1]\n"
                "          [--src-gpu 0] [--dst-gpu 1] [--size 64M]\n"
                "          [--src-gid-idx X] [--dst-gid-idx Y]  # optional override\n", argv[0]);
            return 0;
        }
    }

    printf("Config:\n");
    printf("  SRC dev=%s port=%d gpu=%d\n", dev_a.c_str(), port_a, gpu_src);
    printf("  DST dev=%s port=%d gpu=%d\n", dev_b.c_str(), port_b, gpu_dst);
    printf("  size=%lu bytes\n", (unsigned long)bytes);

    Endpoint A, B;
    A.dev_name = dev_a; A.port_num = port_a; A.gid_index_override = override_gid_a;
    B.dev_name = dev_b; B.port_num = port_b; B.gid_index_override = override_gid_b;

    // Open + detect addressing
    A.open_by_name(dev_a.c_str());  // Find RDMA device
    B.open_by_name(dev_b.c_str());
    A.detect_mode_and_pick_gid();
    B.detect_mode_and_pick_gid();

    // 容错：如果一端是 IB_LID，另一端也必须 IB_LID；如果不一致，优先选择 IB_LID（同机常见是同类型）
    if (A.mode != B.mode) { // False
        printf("Addressing mode mismatch: SRC=%d, DST=%d. Trying to reconcile...\n",
               (int)A.mode, (int)B.mode);
        // 简单策略：若任意端是 IB_LID，两端都按 IB_LID 处理（同一 fabric 的 IB 最常见）
        if (A.mode == AddrMode::IB_LID || B.mode == AddrMode::IB_LID) {
            A.mode = B.mode = AddrMode::IB_LID;
            printf("Forcing both endpoints to IB_LID.\n");
        }
    }

    // Resources
    A.create_qp_pd_cq();
    B.create_qp_pd_cq();

    // CUDA buffers
    A.alloc_and_reg_cuda(gpu_src, bytes);
    B.alloc_and_reg_cuda(gpu_dst, bytes);

    // init patterns
    CHECK_CUDA(cudaSetDevice(gpu_src));
    CHECK_CUDA(cudaMemset(A.gpu_buf, 0xAB, bytes));
    CHECK_CUDA(cudaSetDevice(gpu_dst));
    CHECK_CUDA(cudaMemset(B.gpu_buf, 0x00, bytes));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Connect both directions
    A.connect_rc(B);
    B.connect_rc(A);

    // RDMA write A->B
    A.post_rdma_write(B, bytes);
    A.poll_one_wc();

    // Verify
    const size_t chk = std::min<uint64_t>(bytes, 4096);
    std::vector<unsigned char> host(chk);
    CHECK_CUDA(cudaSetDevice(gpu_dst));
    CHECK_CUDA(cudaMemcpy(host.data(), B.gpu_buf, chk, cudaMemcpyDeviceToHost));

    size_t badpos = chk; // assume ok
    for (size_t i=0; i<chk; ++i) if (host[i] != 0xAB) { badpos = i; break; }
    if (badpos == chk) {
        std::cout << "RDMA WRITE OK. First " << chk << " bytes match 0xAB." << std::endl;
    } else {
        std::cerr << "Data mismatch at " << badpos << "\n";
        return 2;
    }

    std::cout << "Done." << std::endl;

    // 例：单点测试（64MB，预热 100，统计 1000 次，并发 64，CQE 每 32 个打一次）
{
    int warmup = 100;
    int iters  = 1000;
    // int inflight = 64;
    int inflight = 32;
    int signal_every = 32;
    size_t bytes = /* 例如 */ 64ULL<<20;    // 64MB

    auto r = bench_one_size(A, B, bytes, iters, warmup, inflight, signal_every);
    std::cout << "[single] size=" << bytes
              << " bytes  GB/s=" << std::fixed << std::setprecision(6) << r.gbps
              << "  avg=" << r.avg_us << "us  p50=" << r.p50_us
              << "us  p99=" << r.p99_us << "us\n";
}
    // 例：尺寸扫描（从 256B 到 64MB，按 2x 递增）
{
    std::vector<size_t> sizes;
    // for (size_t s = 256; s <= (64ULL<<20); s <<= 1) sizes.push_back(s);
    for (size_t s = 256; s <= (4ULL<<20); s <<= 1) sizes.push_back(s);  // 256B -> 4MB

    int warmup = 64;
    int iters  = 2048;    // 小包建议更多次
    // int inflight = 128;   // 大包可适当减小，视 HCA & CQ 深度
    int inflight = 32;
    int signal_every = 32;

    sweep_and_print_csv(A, B, sizes, iters, warmup, inflight, signal_every);
}

    return 0;
}
