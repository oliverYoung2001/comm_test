# comm_test
Aim to test communication atop GPU cluster.

Support 4 kinds of p2p comm at the cuda level:
1. NCCL 
2. MPI 
3. cudaMemcpy-P 
4. cudaMemcpy-nP
(cudaMemcpy can enable or disable P2P mode)

Support nearly all kinds of comm in torch.distributed package.

## Download Sources
```shell
git clone <git_address> --recurse-submodules
```

## Setup Environments

```shell
source ./scripts/env<_hostname>.sh
./scripts/install.sh    # build nccl
```

complete your own config in `./common.mk`, this is a template:
```makefile
COMM_TEST_PATH = <path to this repo>
NCCL_PATH = $(COMM_TEST_PATH)/third_party/nccl
NCCL_BUILD_PATH = $(NCCL_PATH)/build
CUDART_LIB = $(shell dirname `which nvcc`)/../lib64
MPI_CXX = "`which mpicxx`"
GENCODE = "-gencode=arch=compute_80,code=sm_80"
```

## Tools & Usage

### 1 conflict_allinone

This tool can benchmark **together bandwidth** when multiple P2P communications execute simultaneously. (cuda level)

#### scripts
```shell
./scripts/conflict_allinone.sh
```

#### externel inputs
Need a **conflict pattern file** as input config which descripts how P2P comms execute simultaneously.
e.g. (ring algorithm for alltoall on a 4-GPU cluster)
```
[
    [[0, 1], [1, 2], [2, 3], [3, 0]],   // Ring4 + 1
    [[0, 2], [1, 3], [2, 0], [3, 1]],   // Ring4 + 2
    [[0, 3], [1, 0], [2, 1], [3, 2]]    // Ring4 + 3
]
```
There are more examples in `scripts/configs`.


### 2 conflict_bench

This tool can benchmark **together bandwidth** when multiple P2P communications execute simultaneously. (pytorch level)

#### scripts
```shell
./scripts/conflict_bench.sh
```

#### externel inputs
Need a **conflict pattern file** as input config which descripts how P2P comms execute simultaneously.
e.g. (ring algorithm for alltoall on a 4-GPU cluster)
```
[
    [[0, 1], [1, 2], [2, 3], [3, 0]],   // Ring4 + 1
    [[0, 2], [1, 3], [2, 0], [3, 1]],   // Ring4 + 2
    [[0, 3], [1, 0], [2, 1], [3, 2]]    // Ring4 + 3
]
```
There are more examples in `scripts/configs`.


### 3 coll_comm_bench

This tool can benchmark **together bandwidth** of nearly all kinds of comm in torch.distributed package. (pytorch level)

#### scripts
```shell
./scripts/coll_comm_bench.sh
```

#### externel inputs
