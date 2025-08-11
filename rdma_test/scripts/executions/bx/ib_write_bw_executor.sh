#!/bin/bash

# On server node (14.14.15.5, ibs112)
ib_write_bw -d mlx5_4 -p 10121 & pid0=$!
sleep 1
# On client node (14.14.15.4, ibs111)
ib_write_bw -d mlx5_3 -p 10121 127.0.0.1 & pid1=$!

echo "Started process0 with PID $pid0"
echo "Started process1 with PID $pid1"

wait           # 可选：等待所有后台进程完成
