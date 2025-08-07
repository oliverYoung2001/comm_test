# Cluster profiling
#   1. GPU-GPU
nvidia-smi topo -m
#   2. ib
ibstatus
#   3. cpu & gpu topo
lstopo
#       visualize
lstopo --of svg > lstopo.svg    # can see how cpu & gpu are connected
#   4. pcie attributes   # [NOTE]: no cpu !!! 
#       [NOTE]: root complexes are the highest level nodes in a pcie tree
lspci -tvv  # pcie tree
#   5. cpu
lscpu
#   6. show ips of NICs
ip addr show
ip -br addr show    # for short

# Install htop&lstopo
spack install htop
spack install hwloc +libxml2

# Execution scripts
#   1. topo_discover
srun -p h01 -w g42 --gres=gpu:8 ./build/topo_discover 2>&1 | tee ./logs/topo_discover.log
