# Wrapper for regular all reduce to test timing utilities
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from timer import Timer


class TimedARWrapper:
    def __init__(self, timer):
        self.timer = timer

    def reduce(self, process_group, bucket):
        with self.timer('pytorch.all_reduce'):
            reduced = allreduce_hook(process_group, bucket)
        return reduced

