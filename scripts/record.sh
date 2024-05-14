#!/bin/bash

# GPU-GPU
nvidia-smi topo -m

# ib
ibstatus

# cpu & gpu topo
lstopo
# visualize
lstopo --of png > output.png    # can see how cpu & gpu are connected

# pcie attributes   # [NOTE]: no cpu !!! 
# [NOTE]: root complexes are the highest level nodes in a pcie tree
lspci -tvv  # pcie tree
/sys/devices/...    # pcie bus detailed info
0000:47:02.0:pcie001
0000:47:02.0:pcie010
0000:48:00.0
aer_dev_correctable
aer_dev_fatal
aer_dev_nonfatal
aer_rootport_total_err_cor
aer_rootport_total_err_fatal
aer_rootport_total_err_nonfatal
ari_enabled
broken_parity_status
class
config
consistent_dma_mask_bits
current_link_speed
current_link_width
d3cold_allowed
device
dma_mask_bits
driver
driver_override
enable
firmware_node
irq
link
local_cpulist
local_cpus
max_link_speed  # 16.0 GT/s -> PCIe4.0
max_link_width  # 16x -> 16 lanes, total BW = 16 * 16.0 GT/s / 8 * (128 / 130) = 31.508GB/s
modalias
msi_bus
msi_irqs
numa_node
pci_bus
power
remove
rescan
resource
resource0
revision
secondary_bus_number
subordinate_bus_number
subsystem
subsystem_device
subsystem_vendor
uevent
vendor
wakeup

NVIDIA Corporation Device ....      # GPU product
Mellanox Technologies MT28908 Family [ConnectX-6]   # NIC

# cpu
lscpu
