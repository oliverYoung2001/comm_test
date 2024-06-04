#!/bin/bash
source ~/env.sh
# spack load openmpi@4.1.5%gcc@12.2.0
spack load gcc@11.3.0
spack load openmpi@4.1.5%gcc@11.3.0
spack load cuda@12.2.1%gcc@12.2.0
conda activate mg
