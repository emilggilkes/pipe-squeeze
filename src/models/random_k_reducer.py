# Code adapted from https://github.com/sands-lab/grace

from abc import ABC, abstractmethod

import numpy as np
import torch


class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors]) # copies
    
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)
    
    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers
    

def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


class Reducer:
    def __init__(self, random_seed, timer):
        self.rng = np.random.RandomState(random_seed)
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.timer = timer

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()



class RandomKReducer(Reducer):
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    def __init__(self, random_seed, timer, compress_ratio, rank):
        super().__init__(random_seed, timer)
        self.global_step = 0
        self.compress_ratio = compress_ratio
        self.rank = rank

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        values_list = []
        indices_list = []

        with self.timer("reduce.block", verbosity=1):
            for i, tensor in enumerate(grad_in):
                block_size = max(1, int(self.compress_ratio * tensor.nelement()))
                indices = self.rng.choice(tensor.nelement(), block_size, replace=False)
                indices_list.append(indices)
                values = tensor.view(-1)[indices]
                values_list.append(values)

        #print(f'Rank: {self.rank}. Indices: {sorted(indices)}')

        with self.timer("reduce.flatpack", verbosity=1):
            flat_values = TensorBuffer(values_list)

        with self.timer("reduce.memory", verbosity=1):
            for tensor, mem, indices in zip(grad_in, memory_out, indices_list):
                mem.data[:] = tensor
                mem.view(-1)[indices] = 0.0

        with self.timer("reduce.reduce", verbosity=1):
            flat_values.all_reduce()
            flat_values.buffer.data /= self.n_workers
            bits_communicated += flat_values.bits()

        with self.timer("reduce.combine", verbosity=1):
            for tensor, out, values, indices in zip(grad_in, grad_out, flat_values, indices_list):
                out.data.zero_()
                out.view(-1)[indices] = values

        return bits_communicated

