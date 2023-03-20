import torch.distributed as dist

class GlobalParallelContext():
    def __init__(self, world_size = 0, local_rank = 0) -> None:
        self.world_size = world_size
        self.local_rank = local_rank
        self.group = dist.GroupMember.WORLD
    
    def set_world_size(self, world_size):
        self.world_size = world_size
    
    def get_world_size(self):
        return self.world_size
    
    def set_local_rank(self, local_rank):
        self.local_rank = local_rank
    
    def get_local_rank(self):
        return self.local_rank
    
    def set_group(self, group: dist.ProcessGroup):
        self.group = group
    
    def get_group(self):
        return self.group
        

gpc = GlobalParallelContext()