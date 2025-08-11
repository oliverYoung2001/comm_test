#!/bin/bash
source ~/env.sh
spack load cuda@12.0.1%gcc@=12.2.0
# spack load openmpi@4.1.5%gcc@=12.2.0
spack unload python

# # Openmpi
export OPENMPI_HOME=/home/yhy/.local/openmpi-4.1.6

# use defaule Openmpi
# export OPENMPI_HOME=/home/yhy/.local/anaconda3/bin/mpirun


export PATH="$OPENMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$C_INCLUDE_PATH"  # for #include <mpi.h>
export CPLUS_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$CPLUS_INCLUDE_PATH"  # for #include <mpi.h>

conda activate mg
