#!/bin/bash
./scripts/wrapper_conflict_bench.sh 2>&1 | tee ./results_wq/cb_g11-14.log
./scripts/wrapper_coll_comm_bench.sh 2>&1 | tee ./results_wq/ccb_g11-14.log
