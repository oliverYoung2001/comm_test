# add nccl to submodules
# git submodule add -b v2.18 https://github.com/NVIDIA/nccl.git third_party/nccl

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

# install openmpi
# Ref: https://yuhldr.github.io/posts/bfa79f01.html
OPENMPI_HOME=<position to install>
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2
tar jxvf openmpi-4.1.6
cd openmpi-4.1.6
./configure --prefix=$OPENMPI_HOME --with-slurm --with-pmix # necessary to install openmpi with slurm support
make -j
make install
export OPENMPI_HOME=/home/zhaijidong/yhy/.local/openmpi
export PATH="$OPENMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"
cd ..

