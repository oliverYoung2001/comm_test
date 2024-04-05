from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from utils.parallel_context import gpc
import time
# from colossalai.context.parallel_mode import ParallelMode
# from colossalai.core import global_context as gpc

from utils.comm_async import gather_async, gather_async_opp
from utils.common import calc_comm_vol_a2a, calc_comm_vol_ag
import random
import sys

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)

def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _reduce(tensor: Tensor) -> Tensor:
    if gpc.get_world_size() == 1:
        return tensor

    dist.all_reduce(tensor,
                    op=dist.ReduceOp.SUM,
                    group=gpc.get_group(),
                    async_op=False)

    return tensor


def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    if gpc.get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[dim], gpc.get_world_size())
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[gpc.get_local_rank()].contiguous()

    return output


def copy(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Copy.apply(input)
    return input


class Copy(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Copy", input: Tensor) -> Tensor:
        return input

    @staticmethod
    def backward(ctx: "Copy", grad_output: Tensor) -> Tensor:
        return _reduce(grad_output)


def scatter(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Scatter.apply(input, dim)
    else:
        input = _split(input, dim=dim)
    return input


class Scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Scatter", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx: "Scatter", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _gather(grad_output, dim=int(dim)), None


def reduce(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Reduce.apply(input)
    else:
        input = _reduce(input)
    return input


class Reduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Reduce", input: Tensor) -> Tensor:
        return _reduce(input)

    @staticmethod
    def backward(ctx: "Reduce", grad_output: Tensor) -> Tensor:
        return grad_output


def gather(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Gather.apply(input, dim)
    else:
        input = _gather(input, dim=dim)
    return input


class Gather(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Gather", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx: "Gather", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _split(grad_output, dim=int(dim)), None

def _gather(tensor: Tensor, dim: int = -1, method=None) -> Tensor:
    if gpc.get_world_size() == 1:
        return tensor
    bindwidth = 0
    if dim == 1 and list(tensor.shape)[0] == 1:
        output_shape = list(tensor.shape)
        output_shape[1] *= gpc.get_world_size()
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = output.chunk(gpc.get_world_size(), dim=1)

        comm_vol = calc_comm_vol_ag(tensor, gpc.get_world_size())
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        dist.all_gather(list(tensor_list),
                        tensor,
                        group=gpc.get_group(),
                        async_op=False)        
        torch.cuda.synchronize()
        dist.barrier()
        t_d = time.time() - t0
        bandwidth = comm_vol / t_d
        # print(f'{method}: {t_d}, BD: {bandwidth} GB/s')
        
    else:
        tensor_list = [
            torch.empty_like(tensor) for _ in range(gpc.get_world_size())
        ]
        dist.all_gather(tensor_list,
                        tensor,
                        group=gpc.get_group(),
                        async_op=False)
        output = torch.cat(tensor_list, dim=dim)

    return output, bandwidth if method == 'AGD' else output

def _all_to_all_allgather_del(tensor: Tensor, in_dim: int = -1, out_dim: int = -1, method='AGD') -> Tensor:
    # print(f'in all_to_all_AG_del !!!')
    world_size = gpc.get_world_size()
    if world_size == 1:
        return tensor
    # allgather
    # t0 = time.time()
    gathered, bandwidth = _gather(tensor.contiguous(), dim=out_dim, method=method)
    # torch.cuda.synchronize()
    # t1 = time.time()
    # print(f'AG time: {t1 - t0}')
    # gathered, work = gather_async(tensor.contiguous(), dim=out_dim)
    # gathered = gather_async_opp(gathered, work, dim=out_dim)
    # del
    local_rank = gpc.get_local_rank()
    # chunk_size = gathered.shape[in_dim] // world_size
    # if in_dim == 1:
    #     output = gathered[:,  chunk_size * local_rank: chunk_size * (local_rank + 1), ...]
    # else:
    #     output = gathered[:, :, chunk_size * local_rank: chunk_size * (local_rank + 1), ...]
    # output = torch.split(gathered, gathered.shape[in_dim] // world_size, dim=in_dim)[local_rank]
    # tmp = gathered.chunk(world_size, dim=in_dim)[local_rank]
    # output = tmp.clone()
    
    output = gathered.chunk(world_size, dim=in_dim)[local_rank].clone()
    # method 3
    # tmp = output
    # output = torch.empty_like(tmp)
    # output.data.copy_(tmp)
    
    # print(f'local_rank: {local_rank}')
    # print(f'tmp.data_ptr(): {hex(tmp.data_ptr())}')
    # print(f'output.data_ptr(): {hex(output.data_ptr())}')
    # print(f'rank: {local_rank}, tensor.shape: {tensor.shape}, gathered.shape: {gathered.shape}, output.shape: {output.shape}')
    del gathered
    return output, bandwidth
    

def a2a_p2p_SC0(output_tensor_list, input_tensor_list, async_op=True):
    """
    17GB/s, octave
        1	1	2	3
        3	1	1	2
        2	3	1	1
        1	2	3	1
        1	1	2	3	4	5	6	7
        7	1	1	2	3	4	5	6
        6	7	1	1	2	3	4	5
        5	6	7	1	1	2	3	4
        4	5	6	7	1	1	2	3
        3	4	5	6	7	1	1	2
        2	3	4	5	6	7	1	1
        1	2	3	4	5	6	7	1
    """
    world_size = gpc.get_world_size()
    local_rank = gpc.get_local_rank()
    assert world_size == len(output_tensor_list) == len(input_tensor_list)
    assert output_tensor_list[0].shape == input_tensor_list[0].shape
    group = gpc.get_group()
    if async_op:
        ops = []
        last = (local_rank + world_size - 1) % world_size
        next = (local_rank + 1) % world_size
        ops.append(dist.P2POp(dist.irecv, output_tensor_list[last], last))
        ops.append(dist.P2POp(dist.isend, input_tensor_list[next], next))
        works = dist.batch_isend_irecv(ops)
        output_tensor_list[local_rank].data.copy_(input_tensor_list[local_rank])    # overlapped with comm, 远小于一轮的通信时间！！！
        for work in works:
            work.wait()
            
        for r in range(2, world_size):
            ops = []
            src = (local_rank + world_size - r) % world_size
            dst = (local_rank + r) % world_size
            ops.append(dist.P2POp(dist.irecv, output_tensor_list[src], src))
            ops.append(dist.P2POp(dist.isend, input_tensor_list[dst], dst))
            works = dist.batch_isend_irecv(ops)
            for work in works:
                work.wait()
    else:
        # [TODO]
        pass

def a2a_p2p_SC1(output_tensor_list, input_tensor_list, async_op=True):
    """
        1	1	1	1
        1	1	1	1
        1	1	1	1
        1	1	1	1
        1	1	1	1	1	1	1	1
        1	1	1	1	1	1	1	1
        1	1	1	1	1	1	1	1
        1	1	1	1	1	1	1	1
        1	1	1	1	1	1	1	1
        1	1	1	1	1	1	1	1
        1	1	1	1	1	1	1	1
        1	1	1	1	1	1	1	1
    """
    world_size = gpc.get_world_size()
    local_rank = gpc.get_local_rank()
    assert world_size == len(output_tensor_list) == len(input_tensor_list)
    assert output_tensor_list[0].shape == input_tensor_list[0].shape
    # print(f'local_rank: {local_rank}, in async p2p for a2a')
    group = gpc.get_group()
    if async_op:
        # works = []
        ops = []
        # for i in range(1, world_size):
        #     src = (local_rank + i) % world_size
        #     ops.append(dist.P2POp(dist.irecv, output_tensor_list[src], src))
        # for i in range(1, world_size):
        #     dst = (local_rank + i) % world_size
        #     ops.append(dist.P2POp(dist.isend, input_tensor_list[dst], dst))
        for i in range(1, world_size):          # scheduling like 3 rounds, but no effect !!!
            src = (local_rank + world_size - i) % world_size
            ops.append(dist.P2POp(dist.irecv, output_tensor_list[src], src))
            dst = (local_rank + i) % world_size
            ops.append(dist.P2POp(dist.isend, input_tensor_list[dst], dst))
            
        # random.shuffle(ops) # aim to check whether the order of `ops` affects the comm performance
        # conclusion: no influence !!!
        # print(f'local_rank: {local_rank}, ops: {ops[0].peer}, {ops[0].op}')  
        works = dist.batch_isend_irecv(ops)
        # output_tensor_list[local_rank] = input_tensor_list[local_rank]            # copy reference
        output_tensor_list[local_rank].data.copy_(input_tensor_list[local_rank])    # copy value
        # next = (local_rank + 1) % world_size
        # last = (local_rank + world_size - 1) % world_size
        # if local_rank & 1:
        #     work = dist.irecv(output_tensor_list[last], last)
        # else:
        #     work = dist.isend(input_tensor_list[next], next)
        # ops = []
        # print(f'local_rank: {local_rank}, in irecv !!!')
        # # work = dist.irecv(output_tensor_list[last], last, group=group)
        # # works.append(work)
        # ops.append(dist.P2POp(dist.irecv, output_tensor_list[last], last))
        # print(f'local_rank: {local_rank}, irecv done !!!')
        # # work = dist.isend(input_tensor_list[next], next, group=group)
        # # works.append(work)
        # ops.append(dist.P2POp(dist.isend, input_tensor_list[next], next))
        # print(f'local_rank: {local_rank}, isend done !!!')
        # batch_isend_irecv
        for work in works:
            work.wait()
    else:
        # [TODO]
        pass

def a2a_p2p_SC2(output_tensor_list, input_tensor_list, async_op=True):
    """
        1	1	2	2
        1	1	2	2
        2	2	1	1
        2	2	1	1
    """
    world_size = gpc.get_world_size()
    local_rank = gpc.get_local_rank()
    assert world_size == len(output_tensor_list) == len(input_tensor_list)
    assert output_tensor_list[0].shape == input_tensor_list[0].shape
    group = gpc.get_group()
    if async_op:
        # round 1
        ops = []
        src = dst = local_rank ^ 0x1
        ops.append(dist.P2POp(dist.irecv, output_tensor_list[src], src))
        ops.append(dist.P2POp(dist.isend, input_tensor_list[dst], dst))
        works = dist.batch_isend_irecv(ops)
        output_tensor_list[local_rank].data.copy_(input_tensor_list[local_rank])    # overlapped with comm, 远小于一轮的通信时间！！！
        for work in works:
            work.wait()
        
        # round 2
        ops = []
        src = dst = local_rank ^ 0x2
        ops.append(dist.P2POp(dist.irecv, output_tensor_list[src], src))
        ops.append(dist.P2POp(dist.isend, input_tensor_list[dst], dst))
        src = dst = local_rank ^ 0x2 ^ 0x1
        ops.append(dist.P2POp(dist.irecv, output_tensor_list[src], src))
        ops.append(dist.P2POp(dist.isend, input_tensor_list[dst], dst))
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
    else:
        # [TODO]
        pass
     
def a2a_p2p_SC3(output_tensor_list, input_tensor_list, async_op=True):
    """
        1	1	2	2
        2	1	1	2
        2	2	1	1
        1	2	2	1
    """
    world_size = gpc.get_world_size()
    local_rank = gpc.get_local_rank()
    assert world_size == len(output_tensor_list) == len(input_tensor_list)
    assert output_tensor_list[0].shape == input_tensor_list[0].shape
    group = gpc.get_group()
    if async_op:
        # round 1
        ops = []
        src = (local_rank + world_size - 1) % world_size
        dst = (local_rank + 1) % world_size
        ops.append(dist.P2POp(dist.irecv, output_tensor_list[src], src))
        ops.append(dist.P2POp(dist.isend, input_tensor_list[dst], dst))
        works = dist.batch_isend_irecv(ops)
        output_tensor_list[local_rank].data.copy_(input_tensor_list[local_rank])    # overlapped with comm, 远小于一轮的通信时间！！！
        for work in works:
            work.wait()
        
        # round 2
        ops = []
        for r in range(2, world_size):
            src = (local_rank + world_size - r) % world_size
            dst = (local_rank + r) % world_size
            ops.append(dist.P2POp(dist.irecv, output_tensor_list[src], src))
            ops.append(dist.P2POp(dist.isend, input_tensor_list[dst], dst))
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
    else:
        # [TODO]
        pass

def a2a_p2p_8SC2(output_tensor_list, input_tensor_list, async_op=True):
    """
        1	1	1	1	2	2	2	2
        1	1	1	1	2	2	2	2
        1	1	1	1	2	2	2	2
        1	1	1	1	2	2	2	2
        2	2	2	2	1	1	1	1
        2	2	2	2	1	1	1	1
        2	2	2	2	1	1	1	1
        2	2	2	2	1	1	1	1
    """
    world_size = gpc.get_world_size()
    rank = gpc.get_rank()
    assert world_size == len(output_tensor_list) == len(input_tensor_list)
    assert output_tensor_list[0].shape == input_tensor_list[0].shape
    group = gpc.get_group()
    ROUND_NUM = 2
    GROUP_SIZE = world_size // ROUND_NUM
    if async_op:
        # round 1
        ops = []
        p_s = rank // GROUP_SIZE * GROUP_SIZE
        p_e = p_s + GROUP_SIZE
        for peer in range(p_s, p_e):
            if peer != rank:
                ops.append(dist.P2POp(dist.irecv, output_tensor_list[peer], peer))
                ops.append(dist.P2POp(dist.isend, input_tensor_list[peer], peer))
        works = dist.batch_isend_irecv(ops)
        output_tensor_list[rank].data.copy_(input_tensor_list[rank])    # overlapped with comm, 远小于一轮的通信时间！！！
        for work in works:
            work.wait()
        
        # round 2
        ops = []
        p_s = (p_s + GROUP_SIZE) % world_size
        p_e = p_s + GROUP_SIZE
        for peer in range(p_s, p_e):
            ops.append(dist.P2POp(dist.irecv, output_tensor_list[peer], peer))
            ops.append(dist.P2POp(dist.isend, input_tensor_list[peer], peer))
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
    else:
        # [TODO]
        pass

def a2a_p2p_8SC3(output_tensor_list, input_tensor_list, async_op=True):
    """
        1	1	2	2	1	1	2	2
        1	1	2	2	1	1	2	2
        2	2	1	1	2	2	1	1
        2	2	1	1	2	2	1	1
        1	1	2	2	1	1	2	2
        1	1	2	2	1	1	2	2
        2	2	1	1	2	2	1	1
        2	2	1	1	2	2	1	1
    """
    world_size = gpc.get_world_size()
    rank = gpc.get_rank()
    assert world_size == len(output_tensor_list) == len(input_tensor_list)
    assert output_tensor_list[0].shape == input_tensor_list[0].shape
    group = gpc.get_group()
    GROUP_NUM = ROUND_NUM = 2
    GROUP_SIZE = world_size // GROUP_NUM
    if async_op:
        # round 1
        ops = []
        p_s = rank // GROUP_SIZE * GROUP_SIZE
        p_e = p_s + GROUP_SIZE // 2
        for level in range(GROUP_NUM):
            for peer in range(p_s, p_e):
                if peer != rank:
                    ops.append(dist.P2POp(dist.irecv, output_tensor_list[peer], peer))
                    ops.append(dist.P2POp(dist.isend, input_tensor_list[peer], peer))
            p_s = (p_s + GROUP_SIZE) % world_size
            p_e = p_s + GROUP_SIZE // 2
        works = dist.batch_isend_irecv(ops)
        output_tensor_list[rank].data.copy_(input_tensor_list[rank])    # overlapped with comm, 远小于一轮的通信时间！！！
        for work in works:
            work.wait()
        
        # round 2
        ops = []
        p_s = rank // GROUP_SIZE * GROUP_SIZE + GROUP_SIZE // 2
        p_e = p_s + GROUP_SIZE // 2
        for level in range(GROUP_NUM):
            for peer in range(p_s, p_e):
                ops.append(dist.P2POp(dist.irecv, output_tensor_list[peer], peer))
                ops.append(dist.P2POp(dist.isend, input_tensor_list[peer], peer))
            p_s = (p_s + GROUP_SIZE) % world_size
            p_e = p_s + GROUP_SIZE // 2
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
    else:
        # [TODO]
        pass

def _all_to_all_p2p(tensor: Tensor, in_dim: int = -1, out_dim: int = -1, method='P2P') -> Tensor:
    world_size = gpc.get_world_size()
    if world_size == 1:
        return tensor
    split_size = divide(tensor.shape[in_dim], world_size)
    input_tensor_list = torch.split(tensor, split_size, dim=in_dim)

    input_tensor_list = [tensor_.contiguous() for tensor_ in input_tensor_list]
    bandwidth = 0
    if out_dim == 1:
        output_shape = list(input_tensor_list[0].shape)
        output_shape[1] *= world_size
        # 为了优化malloc!!! 将多个小malloc合并为一个大的malloc
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        output_tensor_list = output.chunk(world_size, dim=1)   # issue: may not contiguous
        
        comm_vol = calc_comm_vol_a2a(tensor, world_size)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        if method == 'SC0':
            a2a_p2p_SC0(list(output_tensor_list), input_tensor_list, async_op=True)
        elif method == 'SC1':
            a2a_p2p_SC1(list(output_tensor_list), input_tensor_list, async_op=True)
        elif method == 'SC2':
            a2a_p2p_SC2(list(output_tensor_list), input_tensor_list, async_op=True)
        elif method == 'SC3':
            a2a_p2p_SC3(list(output_tensor_list), input_tensor_list, async_op=True)
        elif method == '8SC0':
            a2a_p2p_SC0(list(output_tensor_list), input_tensor_list, async_op=True)
        elif method == '8SC1':
            a2a_p2p_SC1(list(output_tensor_list), input_tensor_list, async_op=True)
        elif method == '8SC2':
            a2a_p2p_8SC2(list(output_tensor_list), input_tensor_list, async_op=True)
        elif method == '8SC3':
            a2a_p2p_8SC3(list(output_tensor_list), input_tensor_list, async_op=True)
        else:
            assert False, f'Wrong method: {method} !!!'
        torch.cuda.synchronize()
        dist.barrier()
        t_d = time.time() - t0
        bandwidth = comm_vol / t_d
        # print(f'{method}: {t_d}, BD: {bandwidth} GB/s')
        
    else:
        # [TODO] 这里也可以优化malloc，将多个malloc合并为一个，然后split
        output_tensor_list = [torch.ones_like(tensor_) for tensor_ in input_tensor_list]
        a2a_p2p_SC0(output_tensor_list, input_tensor_list, async_op=True)
        output = torch.cat(output_tensor_list, dim=out_dim)
        
    return output, bandwidth

def _all_to_all(tensor: Tensor, in_dim: int = -1, out_dim: int = -1, method=None) -> Tensor:
    local_rank = gpc.get_local_rank()
    if method == 'AGD':
        output_ag, bandwidth = _all_to_all_allgather_del(tensor, in_dim, out_dim, method)
        return output_ag, bandwidth
    if method == 'P2P':
        output_p2p, bandwidth = _all_to_all_p2p(tensor, in_dim, out_dim, method)
        return output_p2p, bandwidth
    if 'SC' in method:
        output_SCX, bandwidth = _all_to_all_p2p(tensor, in_dim, out_dim, method)
        return output_SCX, bandwidth

    if gpc.get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[in_dim], gpc.get_world_size())
    input_tensor_list = torch.split(tensor, split_size, dim=in_dim)

    input_tensor_list = [tensor_.contiguous() for tensor_ in input_tensor_list]
    bandwidth = 0
    if out_dim == 1:
        output_shape = list(input_tensor_list[0].shape)
        output_shape[1] *= gpc.get_world_size()
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        output_tensor_list = output.chunk(gpc.get_world_size(), dim=1)   # issue: may not contiguous
        
        comm_vol = calc_comm_vol_a2a(tensor, gpc.get_world_size())
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        dist.all_to_all(list(output_tensor_list),       # need contiguous
                        input_tensor_list,              # need contiguous
                        group=gpc.get_group(),
                        async_op=False)
        torch.cuda.synchronize()
        dist.barrier()
        t_d = time.time() - t0
        bandwidth = comm_vol / t_d
        # print(f'{method}: {t_d}, BD: {bandwidth} GB/s')
    else:
        output_tensor_list = [torch.ones_like(tensor_) for tensor_ in input_tensor_list]

        dist.all_to_all(output_tensor_list,
                        input_tensor_list,
                        group=gpc.get_group(),
                        async_op=False)

        output = torch.cat(output_tensor_list, dim=out_dim)

    # assert output_ag.shape == output.shape
    # assert output_ag.equal(output)
    # print('equal !!!')
    
    # assert output_p2p.shape == output.shape
    # assert output_p2p.equal(output)
    # print('equal !!!')
    
    # assert output_SCX.shape == output.shape
    # assert output_SCX.equal(output)
    # print('equal !!!')
    return output, bandwidth


def col_to_row(input_: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input_.requires_grad:
        input_ = All_to_All.apply(input_, 1, 2)
    else:
        input_ = _all_to_all(input_, in_dim=1, out_dim=2)
    return input_


def row_to_col(input_: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input_.requires_grad:
        input_ = All_to_All.apply(input_, 2, 1)
    else:
        input_ = _all_to_all(input_, in_dim=2, out_dim=1)
    return input_


class All_to_All(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "All_to_All", input_: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([in_dim, out_dim]))
        return _all_to_all(input_, in_dim=in_dim, out_dim=out_dim)

    @staticmethod
    def backward(ctx: "All_to_All", grad_output: Tensor) -> Tuple[Tensor]:
        saved_tensors = ctx.saved_tensors[0]
        return _all_to_all(grad_output, in_dim=int(saved_tensors[1]),
                           out_dim=int(saved_tensors[0])), None, None
