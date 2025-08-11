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
#   7. PCIe bandwidth
cat /sys/bus/pci/devices/0000:19:00.0/current_link_speed
cat /sys/bus/pci/devices/0000:19:00.0/current_link_width
cat /sys/bus/pci/devices/0000:19:00.0/max_link_speed
cat /sys/bus/pci/devices/0000:19:00.0/max_link_width

# Install htop&lstopo
spack install htop
spack install hwloc +libxml2

# Execution scripts
#   1. topo_discover
srun -p h01 -w g42 --gres=gpu:8 ./build/topo_discover 2>&1 | tee ./cluster_profiler/fit/topo_discover.log
srun -w g0274 --gres=gpu:8 ./build/topo_discover 2>&1 | tee ./cluster_profiler/bx/topo_discover.log
