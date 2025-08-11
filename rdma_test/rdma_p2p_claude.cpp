#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <vector>
#include <memory>
#include <arpa/inet.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_IBV(call) do { \
    if (call) { \
        std::cerr << "IBV error at " << __FILE__ << ":" << __LINE__ << " - " << strerror(errno) << std::endl; \
        exit(1); \
    } \
} while(0)

class RDMAConnection {
private:
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr_send, *mr_recv;
    struct rdma_cm_id *cm_id;
    struct rdma_event_channel *event_channel;
    
    void *gpu_buffer;
    size_t buffer_size;
    int gpu_id;
    
public:
    RDMAConnection(int gpu_id, size_t buf_size) : gpu_id(gpu_id), buffer_size(buf_size) {
        // 设置GPU设备
        CHECK_CUDA(cudaSetDevice(gpu_id));
        
        // 分配GPU内存
        CHECK_CUDA(cudaMalloc(&gpu_buffer, buffer_size));
        
        // 创建RDMA事件通道
        event_channel = rdma_create_event_channel();
        if (!event_channel) {
            std::cerr << "Failed to create RDMA event channel" << std::endl;
            exit(1);
        }
        
        // 创建RDMA CM ID
        if (rdma_create_id(event_channel, &cm_id, nullptr, RDMA_PS_TCP)) {
            std::cerr << "Failed to create RDMA CM ID" << std::endl;
            exit(1);
        }
    }
    
    ~RDMAConnection() {
        cleanup();
    }
    
    void cleanup() {
        if (mr_send) ibv_dereg_mr(mr_send);
        if (mr_recv) ibv_dereg_mr(mr_recv);
        if (qp) ibv_destroy_qp(qp);
        if (cq) ibv_destroy_cq(cq);
        if (pd) ibv_dealloc_pd(pd);
        if (cm_id) rdma_destroy_id(cm_id);
        if (event_channel) rdma_destroy_event_channel(event_channel);
        if (gpu_buffer) cudaFree(gpu_buffer);
    }
    
    void init_device(const std::string& device_name) {
        struct ibv_device **device_list;
        int num_devices;
        
        device_list = ibv_get_device_list(&num_devices);
        if (!device_list) {
            std::cerr << "Failed to get IB device list" << std::endl;
            exit(1);
        }
        
        struct ibv_device *ib_dev = nullptr;
        for (int i = 0; i < num_devices; i++) {
            if (std::string(ibv_get_device_name(device_list[i])) == device_name) {
                ib_dev = device_list[i];
                break;
            }
        }
        
        if (!ib_dev) {
            std::cerr << "IB device " << device_name << " not found" << std::endl;
            ibv_free_device_list(device_list);
            exit(1);
        }
        
        context = ibv_open_device(ib_dev);
        ibv_free_device_list(device_list);
        
        if (!context) {
            std::cerr << "Failed to open IB device" << std::endl;
            exit(1);
        }
        
        // 分配保护域
        pd = ibv_alloc_pd(context);
        if (!pd) {
            std::cerr << "Failed to allocate protection domain" << std::endl;
            exit(1);
        }
        
        // 创建完成队列
        cq = ibv_create_cq(context, 16, nullptr, nullptr, 0);
        if (!cq) {
            std::cerr << "Failed to create completion queue" << std::endl;
            exit(1);
        }
        
        // 注册GPU内存
        mr_send = ibv_reg_mr(pd, gpu_buffer, buffer_size, 
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_send) {
            std::cerr << "Failed to register GPU memory for send" << std::endl;
            exit(1);
        }
        
        mr_recv = mr_send; // 使用同一块内存
    }
    
    void create_qp() {
        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = cq;
        qp_init_attr.recv_cq = cq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.cap.max_send_wr = 16;
        qp_init_attr.cap.max_recv_wr = 16;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        
        qp = ibv_create_qp(pd, &qp_init_attr);
        if (!qp) {
            std::cerr << "Failed to create queue pair" << std::endl;
            exit(1);
        }
    }
    
    void connect_as_server(const std::string& ip, int port) {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
        
        if (rdma_bind_addr(cm_id, (struct sockaddr*)&addr)) {
            std::cerr << "Failed to bind RDMA address" << std::endl;
            exit(1);
        }
        
        if (rdma_listen(cm_id, 1)) {
            std::cerr << "Failed to listen on RDMA" << std::endl;
            exit(1);
        }
        
        std::cout << "Listening for connections on " << ip << ":" << port << std::endl;
        
        // 等待连接请求
        struct rdma_cm_event *event;
        if (rdma_get_cm_event(event_channel, &event)) {
            std::cerr << "Failed to get CM event" << std::endl;
            exit(1);
        }
        
        if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
            std::cerr << "Unexpected CM event: " << event->event << std::endl;
            rdma_ack_cm_event(event);
            exit(1);
        }
        
        // 获取连接信息
        struct rdma_cm_id *conn_id = event->id;
        context = conn_id->verbs;
        rdma_ack_cm_event(event);
        
        // 初始化设备资源
        pd = ibv_alloc_pd(context);
        cq = ibv_create_cq(context, 16, nullptr, nullptr, 0);
        
        // 注册内存
        mr_recv = ibv_reg_mr(pd, gpu_buffer, buffer_size,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        
        // 创建QP
        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = cq;
        qp_init_attr.recv_cq = cq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.cap.max_send_wr = 16;
        qp_init_attr.cap.max_recv_wr = 16;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        
        if (rdma_create_qp(conn_id, pd, &qp_init_attr)) {
            std::cerr << "Failed to create QP" << std::endl;
            exit(1);
        }
        
        qp = conn_id->qp;
        cm_id = conn_id;
        
        // 接受连接
        if (rdma_accept(cm_id, nullptr)) {
            std::cerr << "Failed to accept connection" << std::endl;
            exit(1);
        }
        
        std::cout << "Connection accepted" << std::endl;
    }
    
    void connect_as_client(const std::string& server_ip, int port) {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, server_ip.c_str(), &addr.sin_addr);
        
        if (rdma_resolve_addr(cm_id, nullptr, (struct sockaddr*)&addr, 2000)) {
            std::cerr << "Failed to resolve address" << std::endl;
            exit(1);
        }
        
        // 等待地址解析完成
        struct rdma_cm_event *event;
        if (rdma_get_cm_event(event_channel, &event)) {
            std::cerr << "Failed to get CM event" << std::endl;
            exit(1);
        }
        
        if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
            std::cerr << "Address resolution failed" << std::endl;
            rdma_ack_cm_event(event);
            exit(1);
        }
        
        context = cm_id->verbs;
        rdma_ack_cm_event(event);
        
        // 解析路由
        if (rdma_resolve_route(cm_id, 2000)) {
            std::cerr << "Failed to resolve route" << std::endl;
            exit(1);
        }
        
        // 等待路由解析完成
        if (rdma_get_cm_event(event_channel, &event)) {
            std::cerr << "Failed to get CM event" << std::endl;
            exit(1);
        }
        
        if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
            std::cerr << "Route resolution failed" << std::endl;
            rdma_ack_cm_event(event);
            exit(1);
        }
        
        rdma_ack_cm_event(event);
        
        // 初始化设备资源
        pd = ibv_alloc_pd(context);
        cq = ibv_create_cq(context, 16, nullptr, nullptr, 0);
        
        // 注册内存
        mr_send = ibv_reg_mr(pd, gpu_buffer, buffer_size,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        
        // 创建QP
        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = cq;
        qp_init_attr.recv_cq = cq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.cap.max_send_wr = 16;
        qp_init_attr.cap.max_recv_wr = 16;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        
        if (rdma_create_qp(cm_id, pd, &qp_init_attr)) {
            std::cerr << "Failed to create QP" << std::endl;
            exit(1);
        }
        
        qp = cm_id->qp;
        
        // 连接到服务器
        if (rdma_connect(cm_id, nullptr)) {
            std::cerr << "Failed to connect" << std::endl;
            exit(1);
        }
        
        // 等待连接建立
        if (rdma_get_cm_event(event_channel, &event)) {
            std::cerr << "Failed to get CM event" << std::endl;
            exit(1);
        }
        
        if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
            std::cerr << "Connection failed" << std::endl;
            rdma_ack_cm_event(event);
            exit(1);
        }
        
        rdma_ack_cm_event(event);
        std::cout << "Connected to server" << std::endl;
    }
    
    void send_data(const void* data, size_t size) {
        // 将数据复制到GPU内存
        CHECK_CUDA(cudaMemcpy(gpu_buffer, data, size, cudaMemcpyHostToDevice));
        
        // 准备发送请求
        struct ibv_sge sge = {};
        sge.addr = (uintptr_t)gpu_buffer;
        sge.length = size;
        sge.lkey = mr_send->lkey;
        
        struct ibv_send_wr send_wr = {};
        send_wr.wr_id = 1;
        send_wr.sg_list = &sge;
        send_wr.num_sge = 1;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        
        struct ibv_send_wr *bad_wr;
        if (ibv_post_send(qp, &send_wr, &bad_wr)) {
            std::cerr << "Failed to post send" << std::endl;
            exit(1);
        }
        
        // 等待完成
        wait_for_completion();
        std::cout << "Data sent successfully" << std::endl;
    }
    
    void receive_data(void* data, size_t size) {
        // 准备接收请求
        struct ibv_sge sge = {};
        sge.addr = (uintptr_t)gpu_buffer;
        sge.length = size;
        sge.lkey = mr_recv->lkey;
        
        struct ibv_recv_wr recv_wr = {};
        recv_wr.wr_id = 2;
        recv_wr.sg_list = &sge;
        recv_wr.num_sge = 1;
        
        struct ibv_recv_wr *bad_wr;
        if (ibv_post_recv(qp, &recv_wr, &bad_wr)) {
            std::cerr << "Failed to post receive" << std::endl;
            exit(1);
        }
        
        // 等待完成
        wait_for_completion();
        
        // 将数据从GPU内存复制到主机
        CHECK_CUDA(cudaMemcpy(data, gpu_buffer, size, cudaMemcpyDeviceToHost));
        std::cout << "Data received successfully" << std::endl;
    }
    
private:
    void wait_for_completion() {
        struct ibv_wc wc;
        int completed = 0;
        
        while (!completed) {
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) {
                std::cerr << "Failed to poll CQ" << std::endl;
                exit(1);
            }
            
            if (n > 0) {
                if (wc.status != IBV_WC_SUCCESS) {
                    std::cerr << "Work completion failed with status: " << wc.status << std::endl;
                    exit(1);
                }
                completed = 1;
            }
        }
    }
};

void server_process() {
    std::cout << "Starting server (GPU1 receiver)..." << std::endl;
    
    RDMAConnection conn(1, 1024 * 1024); // GPU1, 1MB buffer
    conn.init_device("mlx5_4");
    
    // 监听连接
    // conn.connect_as_server("14.14.15.4", 12345);
    conn.connect_as_server("127.0.0.1", 12345);
    
    // 接收数据
    std::vector<char> received_data(1024);
    conn.receive_data(received_data.data(), received_data.size());
    
    std::cout << "Received data: ";
    for (size_t i = 0; i < 10 && i < received_data.size(); i++) {
        std::cout << (int)received_data[i] << " ";
    }
    std::cout << std::endl;
}

void client_process() {
    std::cout << "Starting client (GPU0 sender)..." << std::endl;
    
    // 等待服务器启动
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    RDMAConnection conn(0, 1024 * 1024); // GPU0, 1MB buffer
    conn.init_device("mlx5_3");
    
    // 连接到服务器
    // conn.connect_as_client("14.14.15.4", 12345);
    conn.connect_as_client("127.0.0.1", 12345);
    
    // 准备发送数据
    std::vector<char> send_data(1024);
    for (size_t i = 0; i < send_data.size(); i++) {
        send_data[i] = i % 256;
    }
    
    std::cout << "Sending data..." << std::endl;
    conn.send_data(send_data.data(), send_data.size());
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <server|client>" << std::endl;
        return 1;
    }
    
    std::string mode(argv[1]);
    
    try {
        if (mode == "server") {
            server_process();
        } else if (mode == "client") {
            client_process();
        } else {
            std::cout << "Invalid mode. Use 'server' or 'client'" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}