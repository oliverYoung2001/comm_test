#!/bin/bash

NNODESs="3 4 5 6"
NNODESs="2"
for NNODES in $NNODESs; do
    export NNODES
    ./scripts/run/bx/nccl-tests/all_reduce.sh 2>&1 | tee ./logs/nt_ar_G$((NNODES*8)).log
done