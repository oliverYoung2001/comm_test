import torch

def calc_comm_vol_a2a(t: torch.Tensor, world_size): # -> GB
    return t.nelement() * t.element_size() * (world_size - 1) / world_size / pow(1024, 3)

def calc_comm_vol_ag(t: torch.Tensor, world_size):  # -> GB
    return t.nelement() * t.element_size() * (world_size - 1) / pow(1024, 3)
