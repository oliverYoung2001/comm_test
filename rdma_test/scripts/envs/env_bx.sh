#!/bin/bash
# Spack
source /data/yanghy/.local/spack/share/spack/setup-env.sh
#   CUDA 12.8
spack load cuda@12.8.1
# #   htop
# spack load htop
#   hwloc
spack load hwloc +libxml2
#   MPI
# spack install openmpi@5.0.8 && spack install openmpi@4.1.8
# spack install openmpi@4.1.7 +pmi schedulers=slurm
# spack load openmpi@4.1.7
# spack install mpich@4.3.1
spack load mpich@4.3.1
#   NVSHMEM
# spack install nvshmem@3.3.9 cuda_arch=90 ^mpich@4.3.1 ^cuda@12.8.1     # set `cuda_arch` is necesssary !!!
spack load nvshmem@3.3.9

# GCC 12.3.0 (default)
