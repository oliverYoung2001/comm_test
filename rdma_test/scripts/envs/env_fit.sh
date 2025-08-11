#!/bin/bash
# Spack
source /home/fit/zhaijdyhy/.local/spack/share/spack/setup-env.sh
#   CUDA 12.8
spack load cuda@12.8.1
#   htop
spack load htop
#   hwloc
spack install hwloc +libxml2

# GCC 12.3.0 (default)


# Proxy
export http_proxy=127.0.0.1:18901
export https_proxy=127.0.0.1:18901
