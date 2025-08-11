#!/bin/bash

HOST="g0274"

# run
srun -w $HOST ./scripts/executions/bx/ib_write_bw_executor.sh
