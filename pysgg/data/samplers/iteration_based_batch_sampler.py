# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:#todo 我不明白，难道每个iteration都重新赋值吗？没有意义啊
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)#不同的初始iteration或者epoch,会随机更新一次整个数据集的编号排列
            for batch in self.batch_sampler:#调用一次 batchsampler,过一整个子数据集
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch
            print('finish a epoch!')

    def __len__(self):
        return self.num_iterations
