# Wrapper for fp16 to test timing utilities
from typing import Any, Callable

import torch
import torch.distributed as dist
from timer import Timer

class FP16Compressor:
    def __init__(self, timer):
        self.timer = timer

    def compress(self, process_group, bucket):
        result = FP16Compressor.fp16_compress_hook(process_group, bucket, self.timer)
        print(self.timer.summary())
        return result

    @staticmethod
    def fp16_compress_hook(
        process_group: dist.ProcessGroup, bucket: dist.GradBucket, timer
    ) -> torch.futures.Future[torch.Tensor]:
        """
        This DDP communication hook implements a simple gradient compression
        approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
        and then divides it by the process group size.
        It allreduces those ``float16`` gradient tensors. Once compressed gradient
        tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``)
        """
        group_to_use = process_group if process_group is not None else dist.group.WORLD
        world_size = group_to_use.size()

        with timer("fp16_compress_tensor"):
            compressed_tensor = bucket.buffer().to(torch.float16).div_(world_size)

        fut = dist.all_reduce(
            compressed_tensor, group=group_to_use, async_op=True
        ).get_future()

        def decompress(fut, timer):
            with timer("fp16_decompress"):
                decompressed_tensor = bucket.buffer()
                # Decompress in place to reduce the peak memory.
                # See: https://github.com/pytorch/pytorch/issues/45968
                decompressed_tensor.copy_(fut.value()[0])
            return decompressed_tensor
        
        from functools import partial
        decompress_f = partial(decompress, timer=timer)
        return fut.then(decompress_f)

