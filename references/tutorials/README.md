# Usage

## [ddp_intro_cpu_mac](https://google.com)

This script contains demo functions for regular data parallel, checkpointing, model paralelism, and pipeline parallelism. Links to the original tutorials are [here](https://github.com/pytorch/tutorials/blob/master/intermediate_source/ddp_tutorial.rst) and [here](https://pytorch.org/docs/stable/pipeline.html). The only modifications I made was replacing CUDA device calls with CPU.

## [ddp_pipeline_cpu_mac](https://google.com)

This script demonstrates training a transformer model using distributed data parallel and pipeline parallelism. The tutorial's design uses four devices in total. The pipeline initialized with 8 transformer layers on one device and 8 transformer layers on the other device. One pipe is setup across devices 0 and 1 and another across devices 2 and 3. Both pipes are then replicated using DistributedDataParallel.

The only modifications I made to the original tutorial is replacing CUDA device calls with CPU and switching from NCCL backend to GLOO, since NCCL would raise an error (assuming this is because we're using CPU).

The original tutorial on the Pytorch website has a bug, so use [this version](https://github.com/pytorch/tutorials/blob/master/advanced_source/ddp_pipeline.py) on the Pytorch github instead.

