#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netinet/in.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cassert>

#define PORT 18515
#define SIZE (1024)

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); if (e != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(e) << "\n"; exit(1);} } while(0)

// Used to exchange QP connection data
struct QPConn {
    uint32_t qp_num;
    uint16_t lid;
    uint32_t rkey;
    uint64_t addr;
};

void die(const char *reason) {
    fprintf(stderr, "%s\n", reason);
    exit(EXIT_FAILURE);
}

int tcp_sync(int sockfd, QPConn& local, QPConn& remote) {
    write(sockfd, &local, sizeof(local));
    read(sockfd, &remote, sizeof(remote));
    return 0;
}

int main(int argc, char** argv) {
    bool is_server = (argc == 1);
    const char* peer_ip = is_server ? nullptr : argv[1];

    // 1. TCP socket for parameter exchange
    int sockfd;
    if (is_server) {
        int listenfd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in serv{AF_INET, htons(PORT), INADDR_ANY};
        bind(listenfd, (sockaddr*)&serv, sizeof(serv));
        listen(listenfd, 1);
        sockfd = accept(listenfd, nullptr, nullptr);
    } else {
        sockaddr_in serv{AF_INET, htons(PORT), inet_addr(peer_ip)};
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        while (connect(sockfd, (sockaddr*)&serv, sizeof(serv)) != 0) sleep(1);
    }

    // 2. CUDA setup
    CUDA_CHECK(cudaSetDevice(0));
    void* buf0; CUDA_CHECK(cudaMalloc(&buf0, SIZE));
    CUDA_CHECK(cudaMemset(buf0, 0xAB, SIZE));
    CUDA_CHECK(cudaSetDevice(1));
    void* buf1; CUDA_CHECK(cudaMalloc(&buf1, SIZE));
    CUDA_CHECK(cudaMemset(buf1, 0, SIZE));
    uint8_t host_buf[SIZE];

    // 3. RDMA setup
    ibv_device** dev_list = ibv_get_device_list(nullptr);
    // ibv_context* ctx = ibv_open_device(dev_list[0]);  // Adjust for mlx5_3/NIC3
    ibv_context* ctx = nullptr;
    struct ibv_context *selected_ctx = NULL;
    for (int i = 0; dev_list[i]; i++) {
        if (strcmp(ibv_get_device_name(dev_list[i]), is_server ? "mlx5_4" : "mlx5_3") == 0) {
            ctx = ibv_open_device(dev_list[i]);
            break;
        }
    }
    if (! ctx) die("Failed to find NIC");
    ibv_pd* pd = ibv_alloc_pd(ctx); // pd: Protection Domain
    ibv_cq* cq = ibv_create_cq(ctx, 16, nullptr, nullptr, 0);   // cq: completion queue
    ibv_mr* mr0 = ibv_reg_mr(pd, buf0, SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE); // mr: Memory Region
    ibv_mr* mr1 = ibv_reg_mr(pd, buf1, SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ibv_qp_init_attr init{.send_cq=cq, .recv_cq=cq, .cap={.max_send_wr=1, .max_recv_wr=1, .max_send_sge=1, .max_recv_sge=1}, .qp_type=IBV_QPT_RC};
    ibv_qp* qp = ibv_create_qp(pd, &init);  // qp: queue pair

    // 4. Query port to get LID
    ibv_port_attr port_attr;
    ibv_query_port(ctx, 1, &port_attr);

    // 5. Build local connection data
    QPConn local{qp->qp_num, port_attr.lid, mr1->rkey, (uint64_t)buf1};
    QPConn remote{};
    tcp_sync(sockfd, local, remote);

    // 6. Modify QP to INIT, RTR, RTS
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0; attr.port_num = 1; attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

    attr = {}; attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_256; attr.dest_qp_num = remote.qp_num;
    attr.rq_psn = 0; attr.max_dest_rd_atomic = 1; attr.min_rnr_timer = 0;
    ibv_ah_attr ah_attr{};
    ah_attr.is_global = 0; ah_attr.dlid = remote.lid; ah_attr.sl = 0;
    ah_attr.src_path_bits = 0; ah_attr.port_num = 1;
    attr.ah_attr = ah_attr;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER);

    attr = {}; attr.qp_state = IBV_QPS_RTS; attr.timeout = 14;
    attr.retry_cnt = 7; attr.rnr_retry = 7; attr.sq_psn = 0; attr.max_rd_atomic = 1;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC);

    // 7. RDMA Write op
    //      7.1 prepare wr
    //      7.2 call ibv_post_send
    //      7.3 wait for wc(work completion)
    ibv_sge sge{(uint64_t)buf0, SIZE, mr0->lkey};   // sge: scatter gather elements
    // ibv_send_wr wr{.opcode=IBV_WR_RDMA_WRITE, .sg_list=&sge, .num_sge=1, .send_flags=IBV_SEND_SIGNALED};
    ibv_send_wr wr;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    // wr.wr.rdma.remote_addr = ctx->remote_addr;
    // wr.wr.rdma.rkey = ctx->remote_rkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    wr.wr.rdma.remote_addr = remote.addr; wr.wr.rdma.rkey = remote.rkey;
    ibv_send_wr* bad_wr;

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(host_buf, buf1, SIZE, cudaMemcpyDeviceToHost));
    std::cout<<"Received byte before p2p: 0x"<< std::hex<<(int)host_buf[0] << " on " << (is_server ? "server" : "client") <<std::endl;

    ibv_post_send(qp, &wr, &bad_wr);

    ibv_wc wc;  // wc: work completion
    while (ibv_poll_cq(cq, 1, &wc) == 0);
    if (wc.status != IBV_WC_SUCCESS) { std::cerr<<"RDMA write failed\n"; return 1; }

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(host_buf, buf1, SIZE, cudaMemcpyDeviceToHost));
    std::cout<<"Received byte: 0x"<< std::hex<<(int)host_buf[0] << " on " << (is_server ? "server" : "client") <<std::endl;

    return 0;
}