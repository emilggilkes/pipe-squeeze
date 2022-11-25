import torch
from torch import distributed as dist
from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass


# class Compressor(ABC):
#     """Interface for compressing and decompressing a given tensor."""

#     def __init__(self, average=True, tensors_size_are_same=True):
#         self.average = average
#         self.tensors_size_are_same = tensors_size_are_same

#     @abstractmethod
#     def compress(self, tensor, name):
#         """Compresses a tensor and returns it with the context needed to decompress it."""
#         raise NotImplemented("compress was not implemented.")

#     @abstractmethod
#     def decompress(self, tensors, ctx):
#         """Decompress the tensor with the given context."""
#         raise NotImplemented("decompress was not implemented.")

#     def aggregate(self, tensors):
#         """Aggregate a list of tensors."""
#         return sum(tensors)


class Communicator(ABC):
    @abstractmethod
    def send_receive(self, tensors, name, ctx):
        raise NotImplemented("send was not implemented.")

    def __init__(self, compressor, memory, world_size):
        self.compressor = compressor
        self.memory = memory
        self.world_size = world_size

    def step(self, tensor, name):
        tensor = self.memory.compensate(tensor, name)
        tensors_compressed, ctx = self.compressor.compress(tensor, name)
        self.memory.update(tensor, name, self.compressor, tensors_compressed, ctx)
        return self.send_receive(tensors_compressed, name, ctx)

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    # indices = torch.randperm(numel, device=tensor.device)[:k]
    indices = torch.randint(numel, [k], device=tensor.device)
    values = tensor[indices]
    return indices, values


class RandomKCompressor:
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    def __init__(self, compress_ratio, average=True):
        super().__init__()
        self.global_step = 0
        self.compress_ratio = compress_ratio
        self.average = average

    def compress(self, tensor, name):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        indices, values = sparsify(tensor, self.compress_ratio)

        ctx = indices, tensor.numel(), tensor.size()
        return [values], ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, numel, shape = ctx
        values, = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)


class Allreduce(Communicator):

    def send_receive(self, tensors, name, ctx):
        for tensor_compressed in tensors:
            dist.all_reduce(tensor_compressed)
            if self.compressor.average:
                tensor_compressed.div_(self.world_size)
        return self.compressor.decompress(tensors, ctx)


class NoneMemory(Memory):
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass