#!/bin/bash
source ~/env.sh
spack load cuda@12.0.1%gcc@=12.2.0
spack load openmpi@4.1.5%gcc@=12.2.0
