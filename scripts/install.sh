# add nccl to submodules
# git submodule add -b v2.18 https://github.com/NVIDIA/nccl.git third_party/nccl
# git submodule add -b v2.18.6-1 https://github.com/NVIDIA/nccl.git third_party/nccl

# update submodules
git submodule update --init --recursive

# build nccl
NCCL_PATH=./third_party/nccl
NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
pushd $NCCL_PATH
# rm -r build
# git checkout master   # for debug
# git checkout v2.17.1-1
# git checkout v2.10.3-1      # 性能弱于latest
make -j src.build NVCC_GENCODE=$NVCC_GENCODE
popd

# Q: how to build openmpi with cuda support? 
# A: https://www.open-mpi.org/faq/?category=buildcuda (openmpi, cuda, ucx, gdrcopy)

# install gdrcopy (v2.4.1 -> v1.3)
GDRCOPY_HOME=<position to install>  #  /home/zhaijidong/yhy/.local/gdrcopy
git clone git@github.com:NVIDIA/gdrcopy.git
git checkout v2.4.1
# git checkout v1.3
make prefix=$GDRCOPY_HOME CUDA=$(dirname `which nvcc`)/../ all install    # for v2.4.1
make PREFIX=$GDRCOPY_HOME CUDA=$(dirname `which nvcc`)/../ all install      # for v1.3
# sudo ./insmod.sh   # need sudo
mv $GDRCOPY_HOME/lib $GDRCOPY_HOME/lib64    # [NOTE]: fit for ucx !!!
# export LD_LIBRARY_PATH="$GDRCOPY_HOME/lib64:$LD_LIBRARY_PATH"


# install ucx (v1.16.0)
UCX_HOME=<position to install>  # /home/zhaijidong/yhy/.local/ucx
# wget https://github.com/openucx/ucx/releases/download/v1.4.0/ucx-1.4.0.tar.gz
# tar -zxvf ucx-1.4.0.tar.gz
git clone git@github.com:openucx/ucx.git
git checkout v1.16.0
./autogen.sh
./configure --prefix=$UCX_HOME --with-cuda=$(dirname `which nvcc`)/../ --with-gdrcopy=$GDRCOPY_HOME \
&& make -j install


# install openmpi
# Ref: https://yuhldr.github.io/posts/bfa79f01.html
OPENMPI_HOME=<position to install>
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2
tar jxvf openmpi-4.1.6.tar.bz2
cd openmpi-4.1.6
./configure --prefix=$OPENMPI_HOME \
--with-slurm --with-pmix \
--enable-orterun-prefix-by-default --enable-mpirun-prefix-by-default \
--with-cuda=$(dirname `which nvcc`)/../ --with-ucx=$UCX_HOME \    # for cuda-aware mpi
&& make -j && make install # on nico1
# necessary to install openmpi with slurm support

# ENV SETUP:
export OPENMPI_HOME=/home/zhaijidong/yhy/.local/openmpi or /home/yhy/.local/openmpi
export PATH="$OPENMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"
cd ..

