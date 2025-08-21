// ibgda_p2p_nvshmem_srun.cu
// NVSHMEM 3.3.9 + IBGDA (GPU-initiated). Launcher: srun (PMIx/PMI2).
// 路径：PE0(GPU0, mlx5_3:1)  --[IB]-->  PE1(GPU1, mlx5_4:1)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// 某些构建里这两个宏不会在头文件里导出，做个兜底（值按官方定义）
#ifndef NVSHMEMX_INIT_WITH_PMI
#define NVSHMEMX_INIT_WITH_PMI   0x2
#endif
#ifndef NVSHMEMX_INIT_WITH_PMIX
#define NVSHMEMX_INIT_WITH_PMIX  0x4
#endif

#define CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__, cudaGetErrorString(e)); \
  nvshmem_global_exit(1); }}while(0)

__forceinline__ size_t min_sz(size_t a, size_t b){ return a<b?a:b; }

// 设备端：单线程发起 put + quiet（3.3.9 兼容，无 nvshmemx_quiet）
__global__ void gda_put_kernel(char* symm_buf, size_t nbytes, int peer)
{
  if (blockIdx.x==0 && threadIdx.x==0) {
    // 非阻塞 put 到 peer 的对称地址（地址在两端相同）
    nvshmem_putmem_nbi(symm_buf, symm_buf, nbytes, peer);
    // 等待之前 nbi 操作在网络侧完成（设备端 fence/flush）
    nvshmem_quiet();
  }
}

// 简单解析 64M/1G 等
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

static int nvshmem_bootstrap_auto_init() {
  nvshmemx_init_attr_t attr; std::memset(&attr, 0, sizeof(attr));

  int flags = 0;
  const char* force = std::getenv("NVSHMEM_BOOTSTRAP");
  if (force) {
    if (!strcasecmp(force, "PMI2") || !strcasecmp(force, "PMI")) flags = NVSHMEMX_INIT_WITH_PMI;
    else if (!strcasecmp(force, "PMIX"))                         flags = NVSHMEMX_INIT_WITH_PMIX;
  }
  if (!flags) {
    if (std::getenv("PMIX_RANK")) flags = NVSHMEMX_INIT_WITH_PMIX;
    else if (std::getenv("PMI_RANK") || std::getenv("PMI_SIZE")) flags = NVSHMEMX_INIT_WITH_PMI;
  }

  if (!flags) {
    std::fprintf(stderr, "ERROR: no PMIx/PMI env; run with srun --mpi=pmix|pmi2 or use nvshmemrun\n");
    std::exit(1);
  }
  printf("flags: %d\n", flags);
  int rc = nvshmemx_init_attr(flags, &attr);
  if (rc) {
    std::fprintf(stderr, "ERROR: nvshmemx_init_attr(flags=0x%x) failed (rc=%d). "
                         "Your NVSHMEM may lack the requested bootstrap (+pmi/+pmix).\n", flags, rc);
    std::exit(1);
  }
  return 0;

  // nvshmemx_init_attr_t attr;
  // memset(&attr, 0, sizeof(attr));

  // int flags = 0;
  // if (getenv("PMIX_RANK")) {
  //   flags = NVSHMEMX_INIT_WITH_PMIX;
  // } else if (getenv("PMI_RANK") || getenv("PMI_SIZE")) {
  //   flags = NVSHMEMX_INIT_WITH_PMI;
  // }
  // printf("flags: %d\n", flags);
  // if (flags) {
  //   int rc = nvshmemx_init_attr(flags, &attr);
  //   if (rc == 0) return 0;
  //   fprintf(stderr, "nvshmemx_init_attr(flags=0x%x) failed (rc=%d), trying nvshmem_init()\n",
  //           flags, rc);
  // } else {
  //   fprintf(stderr, "No PMIx/PMI env detected; trying nvshmem_init()\n");
  // }

  // // 兜底：老式自动引导（部分构建允许）
  // nvshmem_init();
  // return 0;
}

int main(int argc, char** argv)
{
  if (nvshmem_bootstrap_auto_init() != 0) {
    fprintf(stderr, "NVSHMEM bootstrap init failed.\n");
    return 1;
  }

  int pe   = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  if (npes != 2) {
    if (pe==0) fprintf(stderr, "This demo needs exactly 2 PEs (srun -n 2). But %d PEs got\n", npes);
    nvshmem_finalize();
    return 1;
  }

  // // ---- NVSHMEM 3.3.9：显式 bootstrap with PMIx/PMI2 ----
  // nvshmemx_init_attr_t attr;
  // memset(&attr, 0, sizeof(attr));

  // int flags = 0;
  // if (getenv("PMIX_RANK")) {
  //     // srun 使用 PMIx（现代 Slurm 默认）
  //     flags = NVSHMEMX_INIT_WITH_PMIX;
  // } else if (getenv("PMI_RANK") || getenv("PMI_SIZE")) {
  //     // 一些集群仍是 PMI2
  //     flags = NVSHMEMX_INIT_WITH_PMI;
  // } else {
  //     fprintf(stderr,
  //       "NVSHMEM bootstrap not found (no PMIX_RANK/PMI_RANK). "
  //       "Are you running under srun -n 2 ?\n");
  //     return 1;
  // }

  // if (nvshmemx_init_attr(flags, &attr) != 0) {
  //     fprintf(stderr, "nvshmemx_init_attr failed (flags=0x%x)\n", flags);
  //     return 1;
  // }

  // int pe   = nvshmem_my_pe();
  // int npes = nvshmem_n_pes();
  // if (npes != 2) {
  //     if (pe == 0) fprintf(stderr, "This demo needs exactly 2 PEs (use srun -n 2).\n");
  //     nvshmem_finalize();
  //     return 1;
  // }
  // // 直接用 nvshmem_init()，由 srun (PMIx/PMI2) 引导
  // nvshmem_init();

  // int pe   = nvshmem_my_pe();
  // int npes = nvshmem_n_pes();
  // if (npes != 2) {
  //   if (pe==0) fprintf(stderr, "This demo needs exactly 2 PEs (use srun -n 2).\n");
  //   nvshmem_finalize(); return 1;
  // }

  // srun wrapper 会设置每个任务的 CUDA_VISIBLE_DEVICES，因此这里都选 0
  CHECK_CUDA(cudaSetDevice(0));

  const size_t nbytes = (argc>=2) ? parse_size(argv[1]) : (64ULL<<20);

  // 对称显存分配（两端地址一致，便于设备端 put 直接用本地指针作为远端指针）
  char* symm = (char*)nvshmem_malloc(nbytes);
  if (!symm) {
    fprintf(stderr, "PE%d: nvshmem_malloc %zu bytes failed\n", pe, nbytes);
    nvshmem_global_exit(2);
  }

  // 初始化：PE0 填 0xAB，PE1 清 0x00
  CHECK_CUDA(cudaMemset(symm, pe==0 ? 0xAB : 0x00, nbytes));
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaStream_t st; CHECK_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

  // 用 stream 上的 barrier 对齐两端（3.3.9 可用）
  nvshmemx_barrier_all_on_stream(st);

  if (pe == 0) {
    // 在 GPU 上发起一次跨节点 put（IBGDA 路径由 NVSHMEM/UCX/MLX5 实现）
    gda_put_kernel<<<1, 1, 0, st>>>(symm, nbytes, /*peer=*/1);
    CHECK_CUDA(cudaGetLastError());
  }

  // 等待网络完成并对齐（host 侧等待 GPU stream）
  nvshmemx_barrier_all_on_stream(st);
  CHECK_CUDA(cudaStreamSynchronize(st));

  // 校验：PE1 读取前 4KB，预期 0xAB
  if (pe == 1) {
    const size_t chk = min_sz((size_t)4096, nbytes);
    std::vector<unsigned char> h(chk);
    CHECK_CUDA(cudaMemcpy(h.data(), symm, chk, cudaMemcpyDeviceToHost));
    size_t bad = chk;
    for (size_t i=0;i<chk;i++) if (h[i] != 0xAB) { bad = i; break; }
    if (bad == chk) printf("PE1 verify OK. First %zu bytes == 0xAB.\n", chk);
    else            printf("PE1 verify FAIL at %zu: got 0x%02x\n", bad, (int)h[bad]);
    fflush(stdout);
  }

  CHECK_CUDA(cudaStreamDestroy(st));
  nvshmem_free(symm);
  nvshmem_finalize();
  return 0;
}
