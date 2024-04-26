#!/bin/bash

export OPENMPI_HOME=/home/zhaijidong/yhy/.local/openmpi
export PATH="$OPENMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"

hostname