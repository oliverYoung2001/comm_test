# add nccl to submodules
# git submodule add -b v2.18 https://github.com/NVIDIA/nccl.git third_party/nccl
# git submodule add -b v2.18.6-1 https://github.com/NVIDIA/nccl.git third_party/nccl
# git submodule add -b v2.18.6-1 https://github.com/NVIDIA/nccl.git third_party/nccl
# git submodule add https://github.com/NVIDIA/nccl-tests.git third_party/nccl-tests

# update submodules
git submodule update --init --recursive

ROOT_PATH=<path to comm_test>

# build nccl
NCCL_PATH=./third_party/nccl
NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90"  # both for A100 and H100
pushd $NCCL_PATH
# rm -r build
# git checkout master   # for debug
# git checkout v2.17.1-1
# git checkout v2.10.3-1      # 性能弱于latest
make -j src.build NVCC_GENCODE="${NVCC_GENCODE}"
popd

# build nccl-tests
NCCL_TESTS_PATH=./third_party/nccl-tests
pushd $NCCL_TESTS_PATH
make -j MPI=1 MPI_HOME=$(dirname `which mpicxx`)/.. NCCL_HOME=${ROOT_PATH}/third_party/nccl/build CUDA_HOME=$(dirname `which nvcc`)/..  # aim to find .so
export LD_LIBRARY_PATH=${ROOT_PATH}/third_party/nccl/build/lib:$(dirname `which mpicxx`)/../lib:$(dirname `which nvcc`)/../targets/x86_64-linux/lib:$LD_LIBRARY_PATH
popd

# Q: how to build openmpi with cuda support? 
# A: https://www.open-mpi.org/faq/?category=buildcuda (openmpi, cuda, ucx, gdrcopy)

# install gdrcopy (v2.4.1 -> v1.3)
GDRCOPY_HOME=<position to install>  #  /home/zhaijidong/yhy/.local/gdrcopy or /public/home/qinghuatest/yhy/.local/gdrcopy
git clone https://github.com/NVIDIA/gdrcopy.git # in Sotware
cd gdrcopy && git checkout v2.4.1
# git checkout v1.3
# [NOTE]: need build on node with cuda driver !!!
make prefix=$GDRCOPY_HOME CUDA=$(dirname `which nvcc`)/../ all install    # for v2.4.1
# make PREFIX=$GDRCOPY_HOME CUDA=$(dirname `which nvcc`)/../ all install      # for v1.3
# sudo ./insmod.sh   # need sudo
mv $GDRCOPY_HOME/lib $GDRCOPY_HOME/lib64    # [NOTE]: fit for ucx !!!
# export LD_LIBRARY_PATH="$GDRCOPY_HOME/lib64:$LD_LIBRARY_PATH"


# install ucx (v1.16.0)
UCX_HOME=<position to install>  # /home/zhaijidong/yhy/.local/ucx or /public/home/qinghuatest/yhy/.local/ucx
# wget https://github.com/openucx/ucx/releases/download/v1.4.0/ucx-1.4.0.tar.gz
# tar -zxvf ucx-1.4.0.tar.gz
git clone https://github.com/openucx/ucx.git
cd ucx && git checkout v1.16.0
./autogen.sh
./configure --prefix=$UCX_HOME --with-cuda=$(dirname `which nvcc`)/../ --with-gdrcopy=$GDRCOPY_HOME \
&& make -j install


# install openmpi
# Ref: https://yuhldr.github.io/posts/bfa79f01.html
OPENMPI_HOME=<position to install>  # /home/zhaijidong/yhy/.local/openmpi or /public/home/qinghuatest/yhy/.local/openmpi
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2
tar jxvf openmpi-4.1.6.tar.bz2
cd openmpi-4.1.6
./configure --prefix=$OPENMPI_HOME \
--with-slurm --with-pmix \
--enable-orterun-prefix-by-default --enable-mpirun-prefix-by-default \
--with-cuda=$(dirname `which nvcc`)/../ --with-ucx=$UCX_HOME \    # for cuda-aware mpi
# for -levent_core and -levent_pthreads
source ~/yhy/.local/spack/share/spack/setup-env.sh
spack load libevent
export LIBRARY_PATH="$(dirname `which event_rpcgen.py`)/../lib:$LIBRARY_PATH"   # for libevent
make -j && make install # on nico1
# necessary to install openmpi with slurm support

# ENV SETUP:
export OPENMPI_HOME=/home/zhaijidong/yhy/.local/openmpi or /home/yhy/.local/openmpi
export PATH="$OPENMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"
cd ..

