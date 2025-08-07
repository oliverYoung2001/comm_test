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
    struct ibv_context *ctx;           // RDMA 设备上下文
    struct ibv_pd *pd;                 // 保护域
    struct ibv_cq *cq;                 // 完成队列
    struct ibv_qp *qp;                 // 队列对
    struct ibv_mr *mr;                 // 内存区域
    void *gpu_buffer;                  // GPU 内存
    char *host_buffer;                 // 用于验证的 CPU 缓冲区
    struct rdma_cm_id *cm_id;          // RDMA 连接 ID
    struct rdma_event_channel *ec;     // 事件通道
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

void send_memory_info(struct context *ctx) {
    struct memory_info {
        uint64_t addr;
        uint32_t rkey;
    } info = {(uint64_t)ctx->gpu_buffer, ctx->mr->rkey};

    struct ibv_send_wr wr = {};
    struct ibv_sge sge = {};
    struct ibv_send_wr *bad_wr;

    sge.addr = (uintptr_t)&info;
    sge.length = sizeof(info);
    sge.lkey = ctx->mr->lkey;

    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_send(ctx->qp, &wr, &bad_wr)) {
        die("ibv_post_send failed");
    }
}

void on_event(struct rdma_cm_event *event, struct context *ctx) {
    std::cout << "[INFO] event of server: " << magic_enum::enum_name(event->event) << std::endl << std::flush;
    if (event->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
        ctx->cm_id = event->id;
        ctx->ctx = ctx->cm_id->verbs;

        // 设置 GPU1
        if (cudaSetDevice(1) != cudaSuccess) {
            die("cudaSetDevice GPU1 failed");
        }

        // 创建保护域、完成队列和队列对
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

        // 分配 GPU1 内存
        if (cudaMalloc(&ctx->gpu_buffer, 1024) != cudaSuccess) {
            die("cudaMalloc failed");
        }

        // 注册 GPU 内存为 RDMA 内存区域
        ctx->mr = ibv_reg_mr(ctx->pd, ctx->gpu_buffer, 1024, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!ctx->mr) {
            die("ibv_reg_mr failed");
        }

        // 分配主机缓冲区用于接收 rkey/addr
        ctx->host_buffer = (char *)calloc(1024, 1);
        ctx->mr = ibv_reg_mr(ctx->pd, ctx->host_buffer, 1024, IBV_ACCESS_LOCAL_WRITE);

        // 发布接收请求
        post_receive(ctx);

        // 接受连接
        struct rdma_conn_param cm_params = {};
        if (rdma_accept(ctx->cm_id, &cm_params)) {
            die("rdma_accept failed");
        }
    } else if (event->event == RDMA_CM_EVENT_ESTABLISHED) {
        printf("[INFO] Connection established on server\n");
        fflush(stdout);
        send_memory_info(ctx); // 发送 GPU1 内存的 rkey 和 addr
    } else if (event->event == RDMA_CM_EVENT_DISCONNECTED) {
        printf("Disconnected\n");
        exit(0);
    }
    rdma_ack_cm_event(event);
}

int main() {
    printf("[INFO] In rdma_server!\n");
    fflush(stdout);
    // return 0;
    // 选择 NIC4/mlx5_3
    struct ibv_device **device_list = ibv_get_device_list(NULL);
    struct ibv_context *selected_ctx = NULL;
    for (int i = 0; device_list[i]; i++) {
        if (strcmp(ibv_get_device_name(device_list[i]), "mlx5_4") == 0) {
            selected_ctx = ibv_open_device(device_list[i]);
            break;
        }
    }
    if (!selected_ctx) die("Failed to find mlx5_4");
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

    // 绑定到 NIC4/mlx5_4 (IP: 14.14.15.5)
    struct sockaddr_in sin = {};
    sin.sin_family = AF_INET;
    sin.sin_port = htons(10121);
    sin.sin_addr.s_addr = inet_addr("14.14.15.5");
    if (rdma_bind_addr(cm_id, (struct sockaddr *)&sin)) {
        die("rdma_bind_addr failed");
    }

    // 开始监听
    if (rdma_listen(cm_id, 10)) {
        die("rdma_listen failed");
    }

    struct context ctx = {};
    ctx.ec = ec;
    ctx.cm_id = cm_id;

    // 主事件循环
    while (1) {
        printf("[INFO] In main_loop of server.\n");
        fflush(stdout);
        struct rdma_cm_event *event;
        
        if (rdma_get_cm_event(ec, &event)) {
            die("rdma_get_cm_event failed");
        }
        on_event(event, &ctx);

        // 轮询完成队列，检查 RDMA WRITE 完成
        struct ibv_wc wc;
        int num_completions = ibv_poll_cq(ctx.cq, 1, &wc);
        std::cout << "[INFO] num_completions=" << num_completions << " on server." << std::endl;
        while (num_completions > 0) {
            std::cout << "[INFO] wc.status of server: " << magic_enum::enum_name(wc.status) << std::endl << std::flush;
            if (wc.status == IBV_WC_SUCCESS) {
                if (wc.opcode == IBV_WC_RECV) {
                    printf("Received client memory info\n");
                    fflush(stdout);
                }
            }
            num_completions = ibv_poll_cq(ctx.cq, 1, &wc);
        }
    }

    // 清理资源
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