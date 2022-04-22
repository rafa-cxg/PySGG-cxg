# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.distributed as dist

from pysgg.utils.comm import get_world_size
from functools import  reduce
from pysgg.utils.comm import get_world_size, is_main_process, synchronize

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            if loss_dict[k]!=None:
                loss_names.append(k)
                all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)

        #--------------get demention of each all_losses of all gpus-------------#
        each_process_dim=torch.tensor(all_losses.shape[0]).cuda()
        each_process_dim_list = [torch.zeros_like(each_process_dim) for i in range(world_size)]

        dist.all_gather(each_process_dim_list,each_process_dim)
        first = each_process_dim_list[0]

        if all(elem == first for elem in each_process_dim_list):#ans each element number of all gpu is same  so dist.reduce can be applied directly (default)
            dist.reduce(all_losses, dst=0)
            # dist.reduce(all_losses, dst=0)
            if dist.get_rank() == 0:
                # only main process gets accumulated, so only divide by
                # world_size in this case
                all_losses /= world_size
            reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
            # return reduced_losses
        else:
            # -------------padding all_loss to same dim-----------------------#
            each_process_dim_list.sort()
            max_dim = int(each_process_dim_list[-1])
            mask = torch.zeros(max_dim).cuda()
            mask[:all_losses.shape[0]] = all_losses
            all_losses = mask
            all_loss_list = [torch.zeros(max_dim).cuda() for i in range(world_size)]
            dist.all_gather(all_loss_list, all_losses)
            if dist.get_rank() == 0:
                all_losses = reduce(lambda x, y: x + y, (all_loss_list))
                all_losses /= world_size
            reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

