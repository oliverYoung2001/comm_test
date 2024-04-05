import torch
from torch import distributed

def broadcast(tensor, src, group, async_op):
    distributed.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)
    pass

def reduce(tensor, dst, group, async_op):
    distributed.reduce(tensor=tensor, dst=dst, group=group, async_op=async_op)
    pass

def gather(tensor, gather_list, dst, group, async_op):
    distributed.gather(tensor=tensor, gather_list=gather_list, dst=dst, group=group, async_op=async_op)
    pass

def scatter(tensor, scatter_list, src, group, async_op):
    distributed.scatter(tensor=tensor, scatter_list=scatter_list, src=src, group=group, async_op=async_op)
    pass

def allgather(tensor_list, tensor, group, async_op):
    distributed.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)
    pass

def reducescatter(tensor, tensor_list, group, async_op):
    distributed.reduce_scatter(output=tensor, input_list=tensor_list, group=group, async_op=async_op)
    pass

def allreduce(tensor, group, async_op):
    distributed.all_reduce(tensor=tensor, group=group, async_op=async_op)
    pass

def alltoall(output_list, input_list, group, async_op):
    distributed.all_to_all(output_tensor_list=output_list, input_tensor_list=input_list, group=group, async_op=async_op)
    pass
