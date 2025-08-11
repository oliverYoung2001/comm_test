#!/bin/bash

PARTITION=h01
HOST="g46"

# run
srun -p $PARTITION -w $HOST ./scripts/executions/fit/ib_write_bw_executor.sh
