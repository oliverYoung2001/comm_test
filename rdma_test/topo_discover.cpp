// #include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <infiniband/verbs.h>
#include <inttypes.h>
#include <limits.h>
#include <algorithm>
#include <dirent.h>
#include <cuda_runtime.h>

struct TopologyEntry {
    std::string name;
    std::vector<std::string> preferred_hca;
    std::vector<std::string> avail_hca;

    // Json::Value toJson() const {
    //     Json::Value matrix(Json::arrayValue);
    //     Json::Value hca_list(Json::arrayValue);
    //     for (auto &hca : preferred_hca) {
    //         hca_list.append(hca);
    //     }
    //     matrix.append(hca_list);
    //     hca_list.clear();
    //     for (auto &hca : avail_hca) {
    //         hca_list.append(hca);
    //     }
    //     matrix.append(hca_list);
    //     return matrix;
    // }
};

using TopologyMatrix =
    std::unordered_map<std::string /* storage type */, TopologyEntry>;

struct InfinibandDevice {
    std::string name;
    std::string pci_bus_id;
    int numa_node;
};

void print_device_info(struct ibv_device **device_list, int num_devices) {
    printf("Found %d RDMA device(s)\n", num_devices);

    for (int i = 0; i < num_devices; i++) {
        struct ibv_device *dev = device_list[i];
        printf("\nDevice %d:\n", i);

        // 设备名称
        const char *dev_name = ibv_get_device_name(dev);
        printf("  Name: %s\n", dev_name ? dev_name : "Unknown");

        // 设备 GUID
        uint64_t guid = ibv_get_device_guid(dev);
        printf("  GUID: 0x%" PRIx64 "\n", guid);

        // 打开设备上下文
        struct ibv_context *ctx = ibv_open_device(dev);
        if (!ctx) {
            fprintf(stderr, "Failed to open device %s\n", dev_name);
            continue;
        }

        // 查询设备属性
        struct ibv_device_attr dev_attr;
        if (ibv_query_device(ctx, &dev_attr)) {
            fprintf(stderr, "Failed to query device %s\n", dev_name);
            ibv_close_device(ctx);
            continue;
        }
        printf("  Firmware Version: %s\n", dev_attr.fw_ver);
        printf("  Node GUID: 0x%" PRIx64 "\n", dev_attr.node_guid);
        printf("  Max MR Size: 0x%" PRIx64 "\n", dev_attr.max_mr_size);
        printf("  Max QPs: %d\n", dev_attr.max_qp);
        printf("  Max CQs: %d\n", dev_attr.max_cq);

        // 查询端口属性
        for (int port = 1; port <= dev_attr.phys_port_cnt; port++) {
            struct ibv_port_attr port_attr;
            if (ibv_query_port(ctx, port, &port_attr)) {
                fprintf(stderr, "Failed to query port %d on device %s\n", port, dev_name);
                continue;
            }
            printf("  Port %d:\n", port);
            printf("    State: %s\n", port_attr.state == IBV_PORT_ACTIVE ? "Active" : 
                                  port_attr.state == IBV_PORT_DOWN ? "Down" : "Other");
            printf("    LID: %d\n", port_attr.lid);
            printf("    MTU: %d bytes\n", port_attr.active_mtu);

            // 查询 GID
            union ibv_gid gid;
            if (ibv_query_gid(ctx, port, 0, &gid)) {
                fprintf(stderr, "Failed to query GID for port %d on device %s\n", port, dev_name);
                continue;
            }
            printf("    GID[0]: %02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:"
                  "%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x\n",
                  gid.raw[0], gid.raw[1], gid.raw[2], gid.raw[3],
                  gid.raw[4], gid.raw[5], gid.raw[6], gid.raw[7],
                  gid.raw[8], gid.raw[9], gid.raw[10], gid.raw[11],
                  gid.raw[12], gid.raw[13], gid.raw[14], gid.raw[15]);
        }

        ibv_close_device(ctx);
    }
}

void print_infiniband_device_info(const std::vector<InfinibandDevice> &all_hca) {
    printf("\nFound %ld Infiniband device(s)\n", all_hca.size());
    for (int i = 0; i < all_hca.size(); ++ i) {
        printf("\nInfiniband Device %d:\n", i);
        printf("    Name: %s\n", all_hca[i].name.c_str());
        printf("    Pci_bus_id: %s\n", all_hca[i].pci_bus_id.c_str());
        printf("    Numa_node: %d\n", all_hca[i].numa_node);
    }
}

void print_topology_entry(const TopologyEntry &entry) {
    printf("Entry Name: %s\n", entry.name.c_str());
    printf("    preferred_hca: ");
    for (auto& preferred_hca: entry.preferred_hca) {
        printf(" %s", preferred_hca.c_str());
    }
    printf("\n    avail_hca: ");
    for (auto& avail_hca: entry.avail_hca) {
        printf(" %s", avail_hca.c_str());
    }
    printf("\n");
}

void print_topology_matrix(const TopologyMatrix &matrix_) {
    printf("\nFound %ld storage device(s)\n", matrix_.size());
    for (const auto& dev_pair : matrix_) {
        printf("\nStorage device name: %s\n", dev_pair.first.c_str());
        print_topology_entry(dev_pair.second);
    }
}

std::vector<InfinibandDevice> listInfiniBandDevices(
    const std::vector<std::string> &filter) {
    int num_devices = 0;
    std::vector<InfinibandDevice> devices;

    struct ibv_device **device_list = ibv_get_device_list(&num_devices);
    print_device_info(device_list, num_devices);
    // return devices;
    if (!device_list) {
        // LOG(WARNING) << "No RDMA devices found, check your device installation";
        printf("[WARN] No RDMA devices found, check your device installation");
        return {};
    }
    if (device_list && num_devices <= 0) {
        // LOG(WARNING) << "No RDMA devices found, check your device installation";
        printf("[WARN] No RDMA devices found, check your device installation");
        ibv_free_device_list(device_list);
        return {};
    }

    for (int i = 0; i < num_devices; ++i) {
        std::string device_name = ibv_get_device_name(device_list[i]);
        if (!filter.empty() && std::find(filter.begin(), filter.end(),
                                         device_name) == filter.end())
            continue;
        char path[PATH_MAX + 32];
        char resolved_path[PATH_MAX];
        // Get the PCI bus id for the infiniband device. Note that
        // "/sys/class/infiniband/mlx5_X/" is a symlink to
        // "/sys/devices/pciXXXX:XX/XXXX:XX:XX.X/infiniband/mlx5_X/".
        snprintf(path, sizeof(path), "/sys/class/infiniband/%s/../..",
                 device_name.c_str());
        if (realpath(path, resolved_path) == NULL) {
            // PLOG(ERROR) << "listInfiniBandDevices: realpath " << path
            //             << " failed";
            std::cout << "[ERROR] listInfiniBandDevices: realpath " << path << " failed\n";
            continue;
        }
        std::string pci_bus_id = basename(resolved_path);

        int numa_node = -1;
        snprintf(path, sizeof(path), "%s/numa_node", resolved_path);
        std::ifstream(path) >> numa_node;

        devices.push_back(InfinibandDevice{.name = std::move(device_name),
                                           .pci_bus_id = std::move(pci_bus_id),
                                           .numa_node = numa_node});
    }
    ibv_free_device_list(device_list);
    return devices;
}

static std::vector<TopologyEntry> discoverCpuTopology(
    const std::vector<InfinibandDevice> &all_hca) {
    DIR *dir = opendir("/sys/devices/system/node");
    struct dirent *entry;
    std::vector<TopologyEntry> topology;

    if (dir == NULL) {
        // PLOG(WARNING)
        //     << "discoverCpuTopology: open /sys/devices/system/node failed";
        std::cout << "discoverCpuTopology: open /sys/devices/system/node failed";
        return {};
    }
    while ((entry = readdir(dir))) {
        const char *prefix = "node";
        if (entry->d_type != DT_DIR ||
            strncmp(entry->d_name, prefix, strlen(prefix)) != 0) {
            continue;
        }
        int node_id = atoi(entry->d_name + strlen(prefix));
        std::vector<std::string> preferred_hca;
        std::vector<std::string> avail_hca;
        // an HCA connected to the same cpu NUMA node is preferred
        for (const auto &hca : all_hca) {
            if (hca.numa_node == node_id) {
                preferred_hca.push_back(hca.name);
            } else {
                avail_hca.push_back(hca.name);
            }
        }
        topology.push_back(
            TopologyEntry{.name = "cpu:" + std::to_string(node_id),
                          .preferred_hca = std::move(preferred_hca),
                          .avail_hca = std::move(avail_hca)});
    }
    (void)closedir(dir);
    return topology;
}

static int getPciDistance(const char *bus1, const char *bus2) {
    char buf[PATH_MAX];
    char path1[PATH_MAX];
    char path2[PATH_MAX];
    snprintf(buf, sizeof(buf), "/sys/bus/pci/devices/%s", bus1);
    if (realpath(buf, path1) == NULL) {
        return -1;
    }
    snprintf(buf, sizeof(buf), "/sys/bus/pci/devices/%s", bus2);
    if (realpath(buf, path2) == NULL) {
        return -1;
    }

    char *ptr1 = path1;
    char *ptr2 = path2;
    while (*ptr1 && *ptr1 == *ptr2) {
        ptr1++;
        ptr2++;
    }
    int distance = 0;
    for (; *ptr1; ptr1++) {
        distance += (*ptr1 == '/');
    }
    for (; *ptr2; ptr2++) {
        distance += (*ptr2 == '/');
    }

    return distance;
}

static std::vector<TopologyEntry> discoverCudaTopology(
    const std::vector<InfinibandDevice> &all_hca) {
    std::vector<TopologyEntry> topology;
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        device_count = 0;
    }
    for (int i = 0; i < device_count; i++) {
        char pci_bus_id[20];
        if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), i) !=
            cudaSuccess) {
            continue;
        }
        for (char *ch = pci_bus_id; (*ch = tolower(*ch)); ch++);

        std::vector<std::string> preferred_hca;
        std::vector<std::string> avail_hca;

        // Find HCAs with minimum distance in one pass
        int min_distance = INT_MAX;
        std::vector<std::string> min_distance_hcas;

        for (const auto &hca : all_hca) {
            int distance = getPciDistance(hca.pci_bus_id.c_str(), pci_bus_id);
            if (distance >= 0) {
                if (distance < min_distance) {
                    min_distance = distance;
                    min_distance_hcas.clear();
                    min_distance_hcas.push_back(hca.name);
                } else if (distance == min_distance) {
                    min_distance_hcas.push_back(hca.name);
                }
            }
        }

        // Add HCAs with minimum distance to preferred_hca, others to avail_hca
        for (const auto &hca : all_hca) {
            if (std::find(min_distance_hcas.begin(), min_distance_hcas.end(),
                          hca.name) != min_distance_hcas.end()) {
                preferred_hca.push_back(hca.name);
            } else {
                avail_hca.push_back(hca.name);
            }
        }
        topology.push_back(
            TopologyEntry{.name = "cuda:" + std::to_string(i),
                          .preferred_hca = std::move(preferred_hca),
                          .avail_hca = std::move(avail_hca)});
    }
    return topology;
}

// int resolve() {
//     resolved_matrix_.clear();
//     hca_list_.clear();
//     std::map<std::string, int> hca_id_map;
//     int next_hca_map_index = 0;
//     for (auto &entry : matrix_) {
//         for (auto &hca : entry.second.preferred_hca) {
//             if (!hca_id_map.count(hca)) {
//                 hca_list_.push_back(hca);
//                 hca_id_map[hca] = next_hca_map_index++;

//                 resolved_matrix_[kWildcardLocation].preferred_hca.push_back(
//                     hca_id_map[hca]);
//                 resolved_matrix_[kWildcardLocation]
//                     .preferred_hca_name_to_index_map_[hca] = hca_id_map[hca];
//             }
//             resolved_matrix_[entry.first].preferred_hca.push_back(
//                 hca_id_map[hca]);
//             resolved_matrix_[entry.first]
//                 .preferred_hca_name_to_index_map_[hca] = hca_id_map[hca];
//         }
//         for (auto &hca : entry.second.avail_hca) {
//             if (!hca_id_map.count(hca)) {
//                 hca_list_.push_back(hca);
//                 hca_id_map[hca] = next_hca_map_index++;

//                 resolved_matrix_[kWildcardLocation].preferred_hca.push_back(
//                     hca_id_map[hca]);
//                 resolved_matrix_[kWildcardLocation]
//                     .preferred_hca_name_to_index_map_[hca] = hca_id_map[hca];
//             }
//             resolved_matrix_[entry.first].avail_hca.push_back(hca_id_map[hca]);
//             resolved_matrix_[entry.first].avail_hca_name_to_index_map_[hca] =
//                 hca_id_map[hca];
//         }
//     }
//     return 0;
// }

int discover(const std::vector<std::string> &filter) {
    TopologyMatrix matrix_;
    matrix_.clear();
    auto all_hca = listInfiniBandDevices(filter);
    print_infiniband_device_info(all_hca);
    for (auto &ent : discoverCpuTopology(all_hca)) {
        matrix_[ent.name] = ent;
    }
// // #ifdef USE_CUDA
    for (auto &ent : discoverCudaTopology(all_hca)) {
        matrix_[ent.name] = ent;
    }
    print_topology_matrix(matrix_);
// // #endif
//     return resolve();
}

int main(int argc, char **argv) {
    std::vector<std::string> filter_;
    discover(filter_);
    return 0;
}