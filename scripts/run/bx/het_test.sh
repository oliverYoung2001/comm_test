#!/bin/bash
# 用法示例：
#   build_het_groups "node[01-03],node05"
#   build_het_groups "$SLURM_NODELIST"
#
# 依赖：scontrol, sinfo（均为 Slurm 自带）
build_het_groups() {
  local HOST_INPUT="$1"
  if [[ -z "$HOST_INPUT" ]]; then
    echo "Usage: build_het_groups <HOSTLIST>" >&2
    return 2
  fi

  # 1) 展开 hostlist 为节点数组
  mapfile -t ALL_NODES < <(scontrol show hostnames "$HOST_INPUT")
  if [[ ${#ALL_NODES[@]} -eq 0 ]]; then
    echo "No nodes resolved from: $HOST_INPUT" >&2
    return 3
  fi

  # 2) 获取每个节点的 CPU 数量（优先 scontrol 的 CPUTot，失败再用 sinfo）
  declare -A CPU_BY_NODE=()
  for n in "${ALL_NODES[@]}"; do
    local cpu=""
    cpu=$(scontrol show node -o "$n" 2>/dev/null \
            | awk '{
                for (i=1;i<=NF;i++){
                  if ($i ~ /^CPUTot=/){split($i,a,"="); print a[2]; exit}
                  if ($i ~ /^CPUs=/){split($i,a,"="); print a[2]; exit}
                }
              }')
    if [[ -z "$cpu" ]]; then
      cpu=$(sinfo -n "$n" -h -o "%c" 2>/dev/null)
    fi
    if [[ -z "$cpu" ]]; then
      echo "Warning: cannot determine CPU count for $n" >&2
      cpu=1
    fi
    CPU_BY_NODE["$n"]="$cpu"
  done

  # 3) 打印每节点 CPU 表
  printf "\n%-30s %8s\n" "NODE" "CPUs"
  printf "%-30s %8s\n" "------------------------------" "--------"
  for n in "${ALL_NODES[@]}"; do
    printf "%-30s %8s\n" "$n" "${CPU_BY_NODE[$n]}"
  done
  echo

  # 4) 按 CPU 数分组（相同 CPU 的节点合并为一个 het-group）
  declare -A GROUPS=()        # key=cpus, value=逗号分隔的nodelist
  declare -A GROUP_COUNTS=()  # key=cpus, value=节点数量
  for n in "${ALL_NODES[@]}"; do
    local c="${CPU_BY_NODE[$n]}"
    if [[ -z "${GROUPS[$c]}" ]]; then
      GROUPS[$c]="$n"
      GROUP_COUNTS[$c]=1
    else
      GROUPS[$c]+=",$n"
      GROUP_COUNTS[$c]=$(( GROUP_COUNTS[$c] + 1 ))
    fi
  done

  # 5) 对 CPU 组做升序排序，稳定生成 het-group 索引
  mapfile -t CPU_KEYS < <(
    for k in "${!GROUPS[@]}"; do echo "$k"; done | sort -n
  )

  # 6) 生成 --het-group 参数串（每组 --nodes=组内节点数，--cpus-per-task=该组CPU数）
  local out="--exact"
  local idx=0
  for c in "${CPU_KEYS[@]}"; do
    local nodes="${GROUPS[$c]}"
    local nn="${GROUP_COUNTS[$c]}"
    out+=" : --het-group=${idx} --nodes=${nn} --nodelist=${nodes} --ntasks-per-node=1 --cpus-per-task=${c}"
    ((idx++))
  done

  # 7) 输出最终参数串
  echo "$out"
}

build_het_groups "g[0297,0278],g[0290-0291]"
exit 0
EXECUTABLE=hostname

PARTITION=H100
NNODES=2
NPROC_PER_NODE=8
GPUS_PER_NODE=8

#   --het-group=0 --nodes=1 --nodelist="${NODES[0]}" --ntasks-per-node=1 --cpus-per-task="$CPUS0" \
# : --het-group=1 --nodes=1 --nodelist="${NODES[1]}" --ntasks-per-node=1 --cpus-per-task="$CPUS1" \
# 0297, 0278
CPUS0=128
CPUS1=176
CPU_PER_TASK0=$((CPUS0 / NPROC_PER_NODE ))
CPU_PER_TASK1=$((CPUS1 / NPROC_PER_NODE ))

SLURM_ARGS="
-p $PARTITION \
-K \
--exact \
--het-group=0 --nodes=1 --nodelist="g0297" --ntasks-per-node=$NPROC_PER_NODE \
    --cpus-per-task="$CPU_PER_TASK0" --gres=gpu:$GPUS_PER_NODE --cpu-bind=none \
: --het-group=1 --nodes=1 --nodelist="g0310" --ntasks-per-node=$NPROC_PER_NODE \
    --cpus-per-task="$CPU_PER_TASK1" --gres=gpu:$GPUS_PER_NODE --cpu-bind=none \
"
# # --exclusive \
# # --cpu-bind=none \
# if [ "$HOST" != "None" ]; then
#     SLURM_ARGS="$SLURM_ARGS \
#         -w $HOST \
#     "
# fi

# # Run with Slurm
# export SLURM_CPU_BIND=verbose
# export SLURM_CPU_BIND_VERBOSE=1
RUNNER_CMD="srun $SLURM_ARGS"

set -x
$RUNNER_CMD \
./scripts/executor.sh \
$EXECUTABLE
set +x
