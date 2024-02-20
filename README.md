# comm_test
Aim to test communication atop GPU cluster.

Support 4 kinds of p2p comm:
1. NCCL 
2. MPI 
3. cudaMemcpy-P 
4. cudaMemcpy-nP

(cudaMemcpy can enable or disable P2P mode)

## Setup Environments

```shell
./scripts/env<_hostname>.sh
```

## Tools & Usage

### conflict_allinone

This tool can benchmark **together bandwidth** when multiple P2P communications execute simultaneously.

#### scripts
```shell
./scripts/<host_name>/conflict_allinone.sh
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
There are more examples in `csrc/configs`.
