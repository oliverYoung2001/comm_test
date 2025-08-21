#!/bin/bash
# nccl-tests
./scripts/run/bx/nccl-tests/all_reduce.sh 2>&1 | tee ./logs/nt_ar.log
