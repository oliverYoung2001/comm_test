// #include "nccl.h"
#include "comm.h"
#include "utils.h"
#include <cmath>
#include "cuda_runtime.h"
#include <curand_kernel.h>

#define RANK2D(y, x) ((y) * BLOCK_X + (x))
#define RANK3D(z, y, x) (((z) * BLOCK_Y + (y)) * BLOCK_X + (x))

void swap_chunk(void* a, void* b, void* buf, LL SIZE) {
    CUDA_CHECK(cudaMemcpy(buf, a, SIZE, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(a, b, SIZE, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(b, buf, SIZE, cudaMemcpyDefault));
}

int get_factor_2D(int z) {
    for (int x = floor(sqrt(z)); x >= 1; -- x) {
        if ((z % x) == 0) {
            return x;
        }
    }
    return 1;
}

int get_factor_3D(int z) {
    for (int x = floor(pow(z, 1.0 / 3)); x >= 1; -- x) {
        if ((z % x) == 0) {
            return x;
        }
    }
    return 1;
}

void all2all_SC0(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    /*
        RING
        1	1	2	3
        3	1	1	2
        2	3	1	1
        1	2	3	1
        12.0GB/s
    */
   int dst, src;
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    // overlapped ???
    CUDA_CHECK(cudaMemcpyAsync(output_list[rank], input_list[rank], CHUNK_SIZE * sizeof(ncclDataType), cudaMemcpyDefault, stream1));
    
    // int dst = (rank + 1) % comm_size;
    // int src = (rank + comm_size - 1) % comm_size;
    // NCCL_CHECK(ncclGroupStart());
    // NCCL_CHECK(ncclSend(input_list[dst], CHUNK_SIZE, ncclDataType, dst, comm, stream));
    // NCCL_CHECK(ncclRecv(output_list[src], CHUNK_SIZE, ncclDataType, src, comm, stream));
    // NCCL_CHECK(ncclGroupEnd());
    // CUDA_CHECK(cudaDeviceSynchronize());

    for (int r = 1; r < comm_size; ++ r) {
        dst = (rank + r) % comm_size;
        src = (rank + comm_size - r) % comm_size;
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(input_list[dst], CHUNK_SIZE, ncclDataType, dst, comm, stream));
        NCCL_CHECK(ncclRecv(output_list[src], CHUNK_SIZE, ncclDataType, src, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaDeviceSynchronize());        // [WHY] 加上Sync能提高性能！！！
    }
    // NCCL_CHECK(ncclGroupEnd());
    if (async_op == false) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void all2all_SC1(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    /*
        1	1	1	1
        1	1	1	1
        1	1	1	1
        1	1	1	1
        1.6 GB/s
        random_shuffle all send/recv: 1.6GB/s
    */
    // [WHY]: 使用cudaMemcpyAsync有轻微性能下降！！！
    // cudaStream_t stream1;
    // cudaStreamCreate(&stream1);
    // // overlapped ???
    // CUDA_CHECK(cudaMemcpyAsync(output_list[rank], input_list[rank], CHUNK_SIZE * sizeof(ncclDataType), cudaMemcpyDefault, stream1));
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < comm_size; ++ i) {
        // if (i != rank) {
        if (true) {
            NCCL_CHECK(ncclSend(input_list[i], CHUNK_SIZE, ncclDataType, i, comm, stream));
            NCCL_CHECK(ncclRecv(output_list[i], CHUNK_SIZE, ncclDataType, i, comm, stream));
            // NCCL_CHECK(ncclRecv(output_list[i], CHUNK_SIZE, ncclDataType, i, comm, stream));
            // NCCL_CHECK(ncclSend(input_list[i], CHUNK_SIZE, ncclDataType, i, comm, stream));
        }
    }

    // for (int _ = 0; _ < 10; ++ _) {
    //     // NCCL_CHECK(ncclSend(input_list[rank], CHUNK_SIZE, ncclDataType, rank, comm, stream));
    //     // NCCL_CHECK(ncclRecv(output_list[rank], CHUNK_SIZE, ncclDataType, rank, comm, stream));
    //     // cudaMemcpy远快于send/recv
    //     // CUDA_CHECK(cudaMemcpy(output_list[rank], input_list[rank], CHUNK_SIZE * sizeof(ncclDataType), cudaMemcpyDefault)); 
    // }
    NCCL_CHECK(ncclGroupEnd());
    // CUDA_CHECK(cudaDeviceSynchronize());    // [WHY]: 这里添加Sync能有性能提升！！！
    if (async_op == false) {
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void all2all_SC4(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    /*
        PAIR (just for comm_size = 2 ^ n, temporarily)
        12.1GB/s
    */
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    // overlapped ???
    CUDA_CHECK(cudaMemcpyAsync(output_list[rank], input_list[rank], CHUNK_SIZE * sizeof(ncclDataType), cudaMemcpyDefault, stream1));
    
    int dst = rank ^ 1;
    int src = rank ^ 1;
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(input_list[dst], CHUNK_SIZE, ncclDataType, dst, comm, stream));
    NCCL_CHECK(ncclRecv(output_list[src], CHUNK_SIZE, ncclDataType, src, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int r = 2; r < comm_size; ++ r) {
        dst = src = rank ^ r;
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(input_list[dst], CHUNK_SIZE, ncclDataType, dst, comm, stream));
        NCCL_CHECK(ncclRecv(output_list[src], CHUNK_SIZE, ncclDataType, src, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // NCCL_CHECK(ncclGroupEnd());
    if (async_op == false) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void all2all_SC5(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    /*
        1	2	3	4
        3	1	4	2
        2	4	1	3
        4	3	2	1
    */
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    // overlapped ???
    CUDA_CHECK(cudaMemcpyAsync(output_list[rank], input_list[rank], CHUNK_SIZE * sizeof(ncclDataType), cudaMemcpyDefault, stream1));
    
    int dst, src;
    for (int r = 1; r < comm_size; ++ r) {
        dst = (rank + r) % comm_size;
        src = (rank + comm_size - r) % comm_size;
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(input_list[dst], CHUNK_SIZE, ncclDataType, dst, comm, stream));
        NCCL_CHECK(ncclRecv(output_list[src], CHUNK_SIZE, ncclDataType, src, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaDeviceSynchronize());        // [WHY] 加上Sync能提高性能！！！
    }
    // NCCL_CHECK(ncclGroupEnd());
    if (async_op == false) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void all2all_BRUCK(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    /*
        BRUCK
        comm_size = 4
        6.5GB/s (unreal BD, smaller than real BD !!!)
        8.8GB/s (use `output_list[0]` as buffer)
        comm_size = 6
        7.2GB/s (use `output_list[0]` as buffer)
    */

    // Phase1: Adjustment && Comm

    int dst;
    int src;
    int* buf = output_list[0];   // ISSUE: 可以使用output_list[0]作为buffer
    // int* buf_bak;
    // CUDA_CHECK(cudaMalloc(&buf, CHUNK_SIZE * sizeof(int)));
    // buf_bak = buf;  

    int ub_k = ceil(log2(comm_size));
    for (int k = 0; k < ub_k; ++ k) {
        dst = (rank + (1 << k)) % comm_size;
        src = (rank + comm_size - (1 << k)) % comm_size;
        for (int l = 0; l < comm_size; ++ l) {
            if (l >> k & 1) {
                NCCL_CHECK(ncclGroupStart());
                NCCL_CHECK(ncclSend(input_list[(l + rank) % comm_size], CHUNK_SIZE, ncclDataType, dst, comm, stream));
                NCCL_CHECK(ncclRecv(buf, CHUNK_SIZE, ncclDataType, src, comm, stream));
                NCCL_CHECK(ncclGroupEnd());
                CUDA_CHECK(cudaDeviceSynchronize());
                std::swap(input_list[(l + rank) % comm_size], buf);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phase2: Adjustment
    // for (int i = 0; i < comm_size; ++ i) {
    //     CUDA_CHECK(cudaMemcpy(output_list[i], input_list[i], CHUNK_SIZE * sizeof(int), cudaMemcpyDefault));
    // }
    for (int i = rank + 1; i < comm_size; ++ i) {
        CUDA_CHECK(cudaMemcpy(output_list[comm_size - (i - rank)], input_list[(i + rank) % comm_size], CHUNK_SIZE * sizeof(int), cudaMemcpyDefault));
    }
    for (int i = 0; i <= rank; ++ i) {
        CUDA_CHECK(cudaMemcpy(output_list[rank - i], input_list[(i + rank) % comm_size], CHUNK_SIZE * sizeof(int), cudaMemcpyDefault));
    }

    // int* buf;
    // CUDA_CHECK(cudaMalloc(&buf, CHUNK_SIZE * sizeof(int)));
    // for (int i = 0; (i << 1) < rank; ++ i) {
    //     swap_chunk(output_list[i], output_list[rank - i], buf, CHUNK_SIZE * sizeof(int));
    // }
    // for (int i = 1; (i << 1) <= (comm_size - rank - 1); ++ i) {
    //     swap_chunk(output_list[rank + i], output_list[comm_size - i], buf, CHUNK_SIZE * sizeof(int));
    // }
    // CUDA_CHECK(cudaFree(buf));

    // CUDA_CHECK(cudaFree(buf_bak));

    if (async_op == false) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void all2all_RD(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    /*
        Recursive Doubling (just for comm_size = 2 ^ n, temporarily)
        7.0GB/s (unreal BD, smaller than real BD !!!)
    */
    
    // Phase1: Adjustment && Comm

    int dst;
    int src;
    int* buf;
    int* buf_bak;
    CUDA_CHECK(cudaMalloc(&buf, CHUNK_SIZE * sizeof(int)));
    buf_bak = buf;

    int ub_k = ceil(log2(comm_size));
    for (int k = 0; k < ub_k; ++ k) {
        dst = (rank ^ (1 << k));
        src = (rank ^ (1 << k));
        for (int l = 0; l < comm_size; ++ l) {
            if (l >> k & 1) {
                NCCL_CHECK(ncclGroupStart());
                NCCL_CHECK(ncclSend(input_list[l ^ rank], CHUNK_SIZE, ncclDataType, dst, comm, stream));
                NCCL_CHECK(ncclRecv(buf, CHUNK_SIZE, ncclDataType, src, comm, stream));
                NCCL_CHECK(ncclGroupEnd());
                CUDA_CHECK(cudaDeviceSynchronize());
                std::swap(input_list[l ^ rank], buf);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phase2: Adjustment
    for (int i = 0; i < comm_size; ++ i) {
        CUDA_CHECK(cudaMemcpy(output_list[i], input_list[i], CHUNK_SIZE * sizeof(int), cudaMemcpyDefault));
    }
    CUDA_CHECK(cudaFree(buf_bak));

    if (async_op == false) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void all2all_2DMESH(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    /*
    8.9GB/s (unreal BD, smaller than real BD !!!)
    */
    int BLOCK_Y = get_factor_2D(comm_size);        // \le than `sqrt(comm_size)`
    int BLOCK_X = comm_size / BLOCK_Y;          // \ge than `sqrt(comm_size)`
    // printf("BLOCK: (%d, %d)\n", BLOCK_Y, BLOCK_X);

    int X = rank % BLOCK_X;
    int Y = rank / BLOCK_X;
    int dst_X, dst_Y;
    int src_X, src_Y;
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // Phase1: X-axis
    for (int r = 0; r < BLOCK_X; ++ r) {        // RING, can be PAIR
        dst_X = (X + r) % BLOCK_X;
        dst_Y = Y;
        src_X = (X + BLOCK_X - r) % BLOCK_X;
        src_Y = Y;
        for (int y = 0; y < BLOCK_Y; ++ y) {
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclSend(input_list[y * BLOCK_X + dst_X], CHUNK_SIZE, ncclDataType, RANK2D(dst_Y, dst_X), comm, stream));
            NCCL_CHECK(ncclRecv(output_list[y * BLOCK_X + src_X], CHUNK_SIZE, ncclDataType, RANK2D(src_Y, src_X), comm, stream));
            NCCL_CHECK(ncclGroupEnd());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phase2: Y-axis
    for (int r = 1; r < BLOCK_Y; ++ r) {        // RING, can be PAIR
        dst_X = X;
        dst_Y = (Y + r) % BLOCK_Y;
        src_X = X;
        src_Y = (Y + BLOCK_Y - r) % BLOCK_Y;
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(output_list[dst_Y * BLOCK_X], CHUNK_SIZE * BLOCK_X, ncclDataType, RANK2D(dst_Y, dst_X), comm, stream));
        NCCL_CHECK(ncclRecv(input_list[src_Y * BLOCK_X], CHUNK_SIZE * BLOCK_X, ncclDataType, RANK2D(src_Y, src_X), comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaDeviceSynchronize());

    }
    // overlapped ???
    CUDA_CHECK(cudaMemcpyAsync(input_list[Y * BLOCK_X], output_list[Y * BLOCK_X], 
                               CHUNK_SIZE * BLOCK_X * sizeof(int), cudaMemcpyDefault, stream1));

    CUDA_CHECK(cudaDeviceSynchronize());
    // results are in `input_list`
    for (int i = 0; i < comm_size; ++ i) {
        output_list[i] = input_list[i];
    }

    if (async_op == false) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void all2all_3DMESH(int** input_list, int** output_list, LL CHUNK_SIZE, int comm_size, int rank, ncclComm_t comm, ncclDataType_t ncclDataType, cudaStream_t stream, bool async_op) {
    /*
    5.2GB/s (unreal BD, smaller than real BD !!!)
    */
    int BLOCK_Y = get_factor_3D(comm_size);             // BLOCK_X >= BLOCK_Y >= BLOCK_Z
    int BLOCK_Z = get_factor_2D(comm_size / BLOCK_Y);
    int BLOCK_X = comm_size / BLOCK_Y / BLOCK_Z;
    if (BLOCK_Y < BLOCK_Z) {
        std::swap(BLOCK_Y, BLOCK_Z);
    }
    // BLOCK_X = 3;
    // BLOCK_Y = 2;
    // BLOCK_Z = 1;
    // printf("BLOCK: (%d, %d, %d)\n", BLOCK_Z, BLOCK_Y, BLOCK_X);

    int X = rank % BLOCK_X;
    int Y = rank / BLOCK_X % BLOCK_Y;
    int Z = rank / (BLOCK_X * BLOCK_Y);
    int dst_X, dst_Y, dst_Z;
    int src_X, src_Y, src_Z;

    // Phase1: X, Y_d * Z_d
    // input -> output
    dst_Y = src_Y = Y;
    dst_Z = src_Z = Z;
    
    for (int r = 0; r < BLOCK_X; ++ r) {        // RING, can be PAIR
        dst_X = (X + r) % BLOCK_X;
        src_X = (X + BLOCK_X - r) % BLOCK_X;
        for (int z = 0; z < BLOCK_Z; ++ z) {
            for (int y = 0; y < BLOCK_Y; ++ y) {
                NCCL_CHECK(ncclGroupStart());
                NCCL_CHECK(ncclSend(input_list[RANK3D(z, y, dst_X)], CHUNK_SIZE, ncclDataType, RANK3D(dst_Z, dst_Y, dst_X), comm, stream));
                NCCL_CHECK(ncclRecv(output_list[RANK3D(z, y, src_X)], CHUNK_SIZE, ncclDataType, RANK3D(src_Z, src_Y, src_X), comm, stream));
                NCCL_CHECK(ncclGroupEnd());
                CUDA_CHECK(cudaDeviceSynchronize());        // [WHY] 加上Sync能提高性能！！！
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phase2: Y, X_s * Z_d
    // output -> input
    dst_X = src_X = X;
    dst_Z = src_Z = Z;
    
    for (int r = 0; r < BLOCK_Y; ++ r) {        // RING, can be PAIR
        dst_Y = (Y + r) % BLOCK_Y;
        src_Y = (Y + BLOCK_Y - r) % BLOCK_Y;
        for (int z = 0; z < BLOCK_Z; ++ z) {
            for (int x = 0; x < BLOCK_X; ++ x) {
                NCCL_CHECK(ncclGroupStart());
                NCCL_CHECK(ncclSend(output_list[RANK3D(z, dst_Y, x)], CHUNK_SIZE, ncclDataType, RANK3D(dst_Z, dst_Y, dst_X), comm, stream));
                NCCL_CHECK(ncclRecv(input_list[RANK3D(z, src_Y, x)], CHUNK_SIZE, ncclDataType, RANK3D(src_Z, src_Y, src_X), comm, stream));
                NCCL_CHECK(ncclGroupEnd());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phase3: Z, X_s * Y_s
    // intput -> output
    dst_X = src_X = X;
    dst_Y = src_Y = Y;
    
    for (int r = 0; r < BLOCK_Z; ++ r) {        // RING, can be PAIR
        dst_Z = (Z + r) % BLOCK_Z;
        src_Z = (Z + BLOCK_Z - r) % BLOCK_Z;
        for (int y = 0; y < BLOCK_Y; ++ y) {
            for (int x = 0; x < BLOCK_X; ++ x) {
                NCCL_CHECK(ncclGroupStart());
                NCCL_CHECK(ncclSend(input_list[RANK3D(dst_Z, y, x)], CHUNK_SIZE, ncclDataType, RANK3D(dst_Z, dst_Y, dst_X), comm, stream));
                NCCL_CHECK(ncclRecv(output_list[RANK3D(src_Z, y, x)], CHUNK_SIZE, ncclDataType, RANK3D(src_Z, src_Y, src_X), comm, stream));
                NCCL_CHECK(ncclGroupEnd());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (async_op == false) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
