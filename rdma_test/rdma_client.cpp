#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <cuda_runtime.h>
#include <arpa/inet.h>
#include "magic_enum.hpp"

struct context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    void *gpu_buffer;
    char *host_buffer;
    uint64_t remote_addr;
    uint32_t remote_rkey;
    struct rdma_cm_id *cm_id;
    struct rdma_event_channel *ec;
};

struct memory_info {
    uint64_t addr;
    uint32_t rkey;
};

void die(const char *reason) {
    fprintf(stderr, "%s\n", reason);
    exit(EXIT_FAILURE);
}

void post_receive(struct context *ctx) {
    struct ibv_recv_wr wr = {};
    struct ibv_sge sge = {};
    struct ibv_recv_wr *bad_wr;

    sge.addr = (uintptr_t)ctx->host_buffer;
    sge.length = sizeof(struct memory_info);
    sge.lkey = ctx->mr->lkey;

    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(ctx->qp, &wr, &bad_wr)) {
        die("ibv_post_recv failed");
    }
}

void send_gpu_data(struct context *ctx) {
    struct ibv_send_wr wr = {};
    struct ibv_sge sge = {};
    struct ibv_send_wr *bad_wr;

    const char *data = "Hello from GPU0!";
    cudaMemcpy(ctx->gpu_buffer, data, strlen(data) + 1, cudaMemcpyHostToDevice);

    sge.addr = (uintptr_t)ctx->gpu_buffer;
    sge.length = strlen(data) + 1;
    sge.lkey = ctx->mr->lkey;

    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = ctx->remote_addr;
    wr.wr.rdma.rkey = ctx->remote_rkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_send(ctx->qp, &wr, &bad_wr)) {
        die("ibv_post_send failed");
    }
}

void on_event(struct rdma_cm_event *event, struct context *ctx) {
    std::cout << "[INFO] event of client: " << magic_enum::enum_name(event->event) << std::endl << std::flush;
    if (event->event == RDMA_CM_EVENT_ADDR_RESOLVED) {
        ctx->ctx = ctx->cm_id->verbs;
        ctx->pd = ibv_alloc_pd(ctx->ctx);
        ctx->cq = ibv_create_cq(ctx->ctx, 10, NULL, NULL, 0);
        struct ibv_qp_init_attr qp_attr = {};
        qp_attr.send_cq = ctx->cq;
        qp_attr.recv_cq = ctx->cq;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = 10;
        qp_attr.cap.max_recv_wr = 10;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        if (rdma_create_qp(ctx->cm_id, ctx->pd, &qp_attr)) {
            die("rdma_create_qp failed");
        }
        ctx->qp = ctx->cm_id->qp;

        // 设置 GPU0
        if (cudaSetDevice(0) != cudaSuccess) {
            die("cudaSetDevice GPU0 failed");
        }

        // 分配 GPU0 内存
        if (cudaMalloc(&ctx->gpu_buffer, 1024) != cudaSuccess) {
            die("cudaMalloc failed");
        }

        // 注册 GPU 内存
        ctx->mr = ibv_reg_mr(ctx->pd, ctx->gpu_buffer, 1024, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!ctx->mr) {
            die("ibv_reg_mr failed");
        }

        // 分配主机缓冲区用于接收 rkey/addr
        ctx->host_buffer = (char *)calloc(1024, 1);
        ctx->mr = ibv_reg_mr(ctx->pd, ctx->host_buffer, 1024, IBV_ACCESS_LOCAL_WRITE);

        post_receive(ctx);

        if (rdma_resolve_route(ctx->cm_id, 2000)) {
            die("rdma_resolve_route failed");
        }
    } else if (event->event == RDMA_CM_EVENT_ROUTE_RESOLVED) {
        struct rdma_conn_param cm_params = {};
        if (rdma_connect(ctx->cm_id, &cm_params)) {
            die("rdma_connect failed");
        }
    } else if (event->event == RDMA_CM_EVENT_ESTABLISHED) {
        printf("[INFO] Connection established on client\n");
        fflush(stdout);
    } else if (event->event == RDMA_CM_EVENT_DISCONNECTED) {
        printf("Disconnected\n");
        exit(0);
    }
    rdma_ack_cm_event(event);
}

int main(int argc, char *argv[]) {
    printf("[INFO] In rdma_client!\n");
    fflush(stdout);
    // return 0;
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <server_ip>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // 选择 NIC3/mlx5_3
    struct ibv_device **device_list = ibv_get_device_list(NULL);
    struct ibv_context *selected_ctx = NULL;
    for (int i = 0; device_list[i]; i++) {
        if (strcmp(ibv_get_device_name(device_list[i]), "mlx5_3") == 0) {
            selected_ctx = ibv_open_device(device_list[i]);
            break;
        }
    }
    if (!selected_ctx) die("Failed to find mlx5_3");
    ibv_free_device_list(device_list);

    // 创建事件通道
    struct rdma_event_channel *ec = rdma_create_event_channel();
    if (!ec) die("rdma_create_event_channel failed");

    // 创建 RDMA 连接 ID
    struct rdma_cm_id *cm_id;
    if (rdma_create_id(ec, &cm_id, NULL, RDMA_PS_TCP)) {
        die("rdma_create_id failed");
    }
    cm_id->verbs = selected_ctx;

    // 解析服务器地址 (NIC4/mlx5_4: 14.14.15.5)
    struct addrinfo *addr;
    if (getaddrinfo(argv[1], "10121", NULL, &addr)) {
        die("getaddrinfo failed");
    }
    struct sockaddr_in local_addr = {};
    local_addr.sin_family = AF_INET;
    local_addr.sin_addr.s_addr = inet_addr("14.14.15.4"); // NIC3/mlx5_3 IP
    if (rdma_resolve_addr(cm_id, (struct sockaddr *)&local_addr, addr->ai_addr, 2000)) {
        die("rdma_resolve_addr failed");
    }
    freeaddrinfo(addr);

    struct context ctx = {};
    ctx.ec = ec;
    ctx.cm_id = cm_id;

    // 主事件循环
    while (1) {
        printf("[INFO] In main_loop of client.\n");
        fflush(stdout);
        struct rdma_cm_event *event;
        
        auto get_event_ret = rdma_get_cm_event(ec, &event);
        std::cout << "[INFO] get_event_ret=" << get_event_ret << " on client." << std::endl;
        if (get_event_ret) {
            die("rdma_get_cm_event failed");
        }
        on_event(event, &ctx);

        // 轮询完成队列
        struct ibv_wc wc;
        int num_completions = ibv_poll_cq(ctx.cq, 1, &wc);
        std::cout << "[INFO] num_completions=" << num_completions << " on client." << std::endl;
        while (num_completions > 0) {
            std::cout << "[INFO] wc.status of client: " << magic_enum::enum_name(wc.status) << std::endl << std::flush;
            if (wc.status == IBV_WC_SUCCESS) {
                if (wc.opcode == IBV_WC_RECV) {
                    struct memory_info *info = (struct memory_info *)ctx.host_buffer;
                    ctx.remote_addr = info->addr;
                    ctx.remote_rkey = info->rkey;
                    printf("Received remote memory info: addr=0x%lx, rkey=0x%x\n", ctx.remote_addr, ctx.remote_rkey);
                    fflush(stdout);
                    send_gpu_data(&ctx); // 发送 GPU0 数据
                } else if (wc.opcode == IBV_WC_RDMA_WRITE) {
                    printf("RDMA Write completed\n");
                    fflush(stdout);
                }
            }
            num_completions = ibv_poll_cq(ctx.cq, 1, &wc);
        }
    }

    ibv_dereg_mr(ctx.mr);
    cudaFree(ctx.gpu_buffer);
    free(ctx.host_buffer);
    rdma_destroy_qp(ctx.cm_id);
    ibv_destroy_cq(ctx.cq);
    ibv_dealloc_pd(ctx.pd);
    rdma_destroy_id(cm_id);
    rdma_destroy_event_channel(ec);
    ibv_close_device(selected_ctx);
    return 0;
}