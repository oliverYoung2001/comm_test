#!/bin/bash

./build/rdma_p2p_claude server 2>&1 | tee ./logs/bx/rdma_server_claude.log & pid0=$!
sleep 1
./build/rdma_p2p_claude client 2>&1 | tee ./logs/bx/rdma_client_claude.log & pid1=$!

echo "Started process0 with PID $pid0"
echo "Started process1 with PID $pid1"

wait           # 可选：等待所有后台进程完成
