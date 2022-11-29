import torch
from torch import distributed as dist
from abc import ABC, abstractmethod
import numpy as np



class NoneMemory:
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

class RandomKCompressor: 
    def __init__(self, compress_ratio, timer, average=True, tensors_size_are_same=True):
        super().__init__()
        self.global_step = 0
        self.compress_ratio = compress_ratio
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same
        self.timer = timer

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)

    def sparsify(tensor, compress_ratio):
        #tensor = tensor.flatten()
        numel = tensor.numel()
        k = max(1, int(numel * compress_ratio))
        indices = torch.randint(numel, [k], device=tensor.device)
        values = tensor[indices]
        return indices, values

    def random_k_compress_hook(self,
        process_group: dist.ProcessGroup, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        group_to_use = process_group if process_group is not None else dist.group.WORLD
        world_size = group_to_use.size()

        tensor = bucket.buffer()

        h = sum(bytes(tensor.names[0], encoding='utf8'), self.global_step) 
        self.global_step += 1
        torch.manual_seed(h)
        
        indices, compressed_tensor = self.sparsify(tensor, self.compress_ratio)

        ctx = indices, tensor.numel(), tensor.size()
        with self.timer('randomk.all_reduce'):
            fut = dist.all_reduce(
                compressed_tensor.div_(world_size), group=group_to_use, async_op=True
            ).get_future()
        
        def decompress(fut):
            indices, numel, shape = ctx
            tensor_decompressed = torch.zeros(numel, dtype=fut.value()[0].dtype, layout=fut.value()[0].layout, device=fut.value()[0].device)
            tensor_decompressed.scatter_(0, indices, fut.value()[0])
            
            return tensor_decompressed

        return fut.then(decompress)

