// ibgda_p2p_mpi_nvshmem.cu
// NVSHMEM 3.3.9 + MPI bootstrap + IBGDA (GPU-initiated)
// Rank0: GPU0 + mlx5_3:1  →  Rank1: GPU1 + mlx5_4:1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// 建议块大小（可调）
#ifndef GDA_BLOCK_THREADS
// #define GDA_BLOCK_THREADS 256
#define GDA_BLOCK_THREADS 512
#endif

#define CUDA_CHECK(stmt) do {                                      \
  cudaError_t _e = (stmt);                                         \
  if (_e != cudaSuccess) {                                         \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,      \
            cudaGetErrorString(_e));                                \
    MPI_Abort(MPI_COMM_WORLD, 1);                                   \
  }                                                                 \
} while(0)

static size_t parse_size(const char* s) {
  char u=0; double v=0.0;
  if (sscanf(s, "%lf%c", &v, &u) < 1) return strtoull(s,nullptr,10);
  switch(u){
    case 'G': case 'g': return (size_t)(v * (1024ULL*1024ULL*1024ULL));
    case 'M': case 'm': return (size_t)(v * (1024ULL*1024ULL));
    case 'K': case 'k': return (size_t)(v * 1024ULL);
    default: return (size_t)v;
  }
}

// 设备端：把 nbytes 切成多块，每块由整个 block 协作发起，最后一次 quiet
__global__ void gda_put_kernel_pipelined(char* symm_buf, size_t nbytes, int peer,
                                         size_t chunk_bytes, int chunks_inflight)
{
    // 对齐 chunk 到 64B（PCIe/doorbell 友好）
    chunk_bytes = (chunk_bytes + 63) & ~size_t(63);
    if (chunk_bytes == 0) return;

    // 每次最多 pipeline 发起 chunks_inflight 个 chunk，再小小歇一口（确保 WQE 不被过早回收）
    size_t offset = 0;
    while (offset < nbytes) {
        int posted = 0;
        #pragma unroll 1
        for (; posted < chunks_inflight && offset < nbytes; ++posted) {
            size_t len = min(chunk_bytes, nbytes - offset);
        
        nvshmemx_putmem_nbi_block(symm_buf + offset, symm_buf + offset, len, peer);
        // #if defined(NVSHMEMX_API_VERSION)
        //     // block 协作接口（效率更高）
        //     nvshmemx_putmem_nbi_block(symm_buf + offset, symm_buf + offset, len, peer);
        // #else
        //     // 回退：让 block 内线程分段并发发起
        //     size_t t = threadIdx.x + blockIdx.x * blockDim.x;
        //     size_t stride = blockDim.x * gridDim.x;
        //     // 每个线程发自己的一段（避免大量小 put，按 1KB 对齐）
        //     const size_t slice = 1024;
        //     for (size_t off = offset + t * slice; off < offset + len; off += stride * slice) {
        //         size_t l = min(slice, offset + len - off);
        //         if (l) nvshmem_putmem_nbi(symm_buf + off, symm_buf + off, l, peer);
        //     }
        //     __syncthreads();
        // #endif
            offset += len;
        }
        // 可选：轻量让网络喘口气（很多平台可以直接连发，保留空循环更稳）
        __syncthreads();
    }

    // 整个大消息发完再做一次设备端 quiet（只做一次 fence）
    if (threadIdx.x == 0) nvshmem_quiet();
    __syncthreads();
}

__global__ void gda_put_kernel(char* symm_buf, size_t nbytes, int peer) {
  // 设备侧发起一次非阻塞 put，然后在设备侧 quiet；兼容 NVSHMEM 3.3.9（无 nvshmemx_quiet）
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // 把本 PE 的 symm_buf 写到 peer 的同名对称地址
    nvshmem_putmem_nbi(symm_buf, symm_buf, nbytes, peer);
    nvshmem_quiet(); // 等待前述 nbi 操作网络侧完成
  }
}

__global__ void signal_kernel(uint32_t* flag, uint32_t value, int peer) {
  if (threadIdx.x == 0) {
    // 向对端写一个值，通知本条大消息完成
    nvshmem_uint_p(flag, value, peer);
    nvshmem_quiet();
  }
}

static void sizes_from_cmd(int argc, char** argv,
                           std::vector<size_t>& sizes,
                           size_t& iters, size_t& warmup) {
  // 默认参数：从 256B 到 64MB，x2 扫描；iters=100，warmup=20
  size_t minb = 256, maxb = (64ULL<<20), factor = 2;
  iters = 100; warmup = 20;

  for (int i=1;i<argc;i++){
    if (!strcmp(argv[i],"--min") && i+1<argc) minb = parse_size(argv[++i]);
    else if (!strcmp(argv[i],"--max") && i+1<argc) maxb = parse_size(argv[++i]);
    else if (!strcmp(argv[i],"--factor") && i+1<argc) factor = std::max<size_t>(2, strtoull(argv[++i],nullptr,10));
    else if (!strcmp(argv[i],"--iters") && i+1<argc) iters = strtoull(argv[++i],nullptr,10);
    else if (!strcmp(argv[i],"--warmup") && i+1<argc) warmup = strtoull(argv[++i],nullptr,10);
  }
  if (minb > maxb) std::swap(minb, maxb);
  size_t s = minb;
  while (s <= maxb) {
    sizes.push_back(s);
    if (s > maxb / factor) break; // 防溢出
    s *= factor;
  }
}

int main(int argc, char** argv)
{
  // --- MPI 初始化 ---
  MPI_Init(&argc, &argv);
  int rank=0, nranks=1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks != 2) {
    if (rank==0) fprintf(stderr, "This demo requires exactly 2 MPI ranks.\n");
    MPI_Finalize();
    return 1;
  }

  // --- NVSHMEM 用 MPI 引导 ---
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm world = MPI_COMM_WORLD;
  attr.mpi_comm = &world;
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
    if (rank==0) fprintf(stderr, "nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM) failed\n");
    MPI_Finalize();
    return 1;
  }

  // node 内本地 PE 用于选择 GPU（要求外部已设置 CUDA_VISIBLE_DEVICES per-rank）
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("Rank%d, nranks=%d, mype=%d, npes=%d, mype_node=%d\n", rank, nranks, mype, npes, mype_node);
  CUDA_CHECK(cudaSetDevice(mype_node));
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // --- 组装 sizes/iters/warmup ---
  std::vector<size_t> sizes; size_t iters=100, warmup=20;
  sizes_from_cmd(argc, argv, sizes, iters, warmup);

  // 对称堆分配：一个缓冲区即可（两端地址相同）
  size_t max_size = *std::max_element(sizes.begin(), sizes.end());
  char* symm = (char*)nvshmem_malloc(max_size);
  // completion signal
  uint32_t* done_flag = (uint32_t*)nvshmem_malloc(sizeof(uint32_t));
  CUDA_CHECK(cudaMemsetAsync(done_flag, 0, sizeof(uint32_t), stream));
  // static uint32_t token = 1;

  if (!symm) {
    fprintf(stderr, "PE%d: nvshmem_malloc(%zu) failed\n", mype, max_size);
    nvshmem_global_exit(2);
  }

  // 初始化不同内容，便于简单校验
  CUDA_CHECK(cudaMemsetAsync(symm, (mype==0)?0xAB:0x00, max_size, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 对齐起跑线
  nvshmemx_barrier_all_on_stream(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const int peer = (mype + 1) % 2; // 只支持 2 PE：0↔1
  if (mype==0) {
    printf("# IBGDA NVSHMEM P2P benchmark (GPU-initiated)\n");
    printf("# sizes from %zu to %zu bytes, iters=%zu warmup=%zu\n",
           sizes.front(), sizes.back(), iters, warmup);
    printf("chunk,inflight,size_bytes,GBps\n"); fflush(stdout);
  }

  // ---- 尺寸扫描 ----
  for (size_t nbytes : sizes) {
    // // 根据消息大小调整管线参数（经验值，可再微调）
    // size_t chunk = (nbytes >= (8ULL<<20)) ? (2ULL<<20) :      // ≥8MB 用 2MB chunk
    //               (nbytes >= (1ULL<<20)) ? (512ULL<<10) :    // ≥1MB 用 512KB
    //               (64ULL<<10);                               // 小包：64KB
    // int inflight = (nbytes >= (8ULL<<20)) ? 8 : 4;            // in-kernel pipeline 深度

    // Tuning:
    for (size_t chunk = (1ULL<<20); chunk <= (8ULL<<20); chunk *= 2) {
      for (int inflight = 4; inflight <= 16; inflight *= 2) {

    // warmup
    for (size_t w=0; w<warmup; ++w) {
      if (mype==0) {
        // gda_put_kernel<<<1,1,0,stream>>>(symm, nbytes, peer);
        gda_put_kernel_pipelined<<<1,GDA_BLOCK_THREADS,0,stream>>>(symm, nbytes, peer, chunk, inflight);

        // signal_kernel<<<1,32,0,stream>>>(done_flag, token, peer);
      }
      // nvshmemx_barrier_all_on_stream(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // 正式计时：用 CUDA events 记录 GPU 视角时间
    float ms_total = 0.0f;
    cudaEvent_t evs, eve; CUDA_CHECK(cudaEventCreate(&evs)); CUDA_CHECK(cudaEventCreate(&eve));

    for (size_t it=0; it<iters; ++it) {
      if (mype==0) {
        CUDA_CHECK(cudaEventRecord(evs, stream));
        // gda_put_kernel<<<1,1,0,stream>>>(symm, nbytes, peer);
        gda_put_kernel_pipelined<<<1,GDA_BLOCK_THREADS,0,stream>>>(symm, nbytes, peer, chunk, inflight);
      }
      // NVSHMEM stream barrier 协步
      nvshmemx_barrier_all_on_stream(stream);
      if (mype==0) {
        CUDA_CHECK(cudaEventRecord(eve, stream));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (mype==0) {
        float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, evs, eve));
        ms_total += ms;
      }
    }
    if (mype==0) {
      // 平均每次的 GB/s（只算数据传输时间，不含 warmup）
      double s_per_iter = (ms_total / iters) / 1e3;
      double gbps = (double)nbytes / s_per_iter / 1024 / 1024 / 1024;
      printf("%zuMB, %d, %zu, %.6f\n", chunk>>20, inflight, nbytes, gbps); fflush(stdout);
    }
    CUDA_CHECK(cudaEventDestroy(evs)); CUDA_CHECK(cudaEventDestroy(eve));
  }
  }
  }

  // 简单校验：末尾让 PE1 回读前 4KB 检查是否 0xAB
  if (mype==1) {
    const size_t chk = std::min(max_size, (size_t)4096);
    std::vector<unsigned char> host(chk);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), symm, chk, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t bad = chk; for (size_t i=0;i<chk;i++) if (host[i] != 0xAB) { bad = i; break; }
    if (bad == chk) printf("PE1 verify OK. First %zu bytes == 0xAB.\n", chk);
    else            printf("PE1 verify FAIL at %zu: got 0x%02x\n", bad, (int)host[bad]);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  nvshmem_free(symm);
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
