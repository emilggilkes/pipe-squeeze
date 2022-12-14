# Usage

## [ddp_intro_cpu_mac](https://github.com/emilggilkes/pipe-squeeze/blob/main/references/tutorials/ddp_intro_cpu_mac.py)

This script contains demo functions for regular data parallel, checkpointing, model paralelism, and pipeline parallelism. Links to the original tutorials are [here](https://github.com/pytorch/tutorials/blob/master/intermediate_source/ddp_tutorial.rst) and [here](https://pytorch.org/docs/stable/pipeline.html). The only modifications I made was replacing CUDA device calls with CPU.

## [ddp_pipeline_cpu_mac](https://github.com/emilggilkes/pipe-squeeze/blob/main/references/tutorials/ddp_pipeline_cpu_mac.py)

This script demonstrates training a transformer model using distributed data parallel and pipeline parallelism. The tutorial's design uses four devices in total. The pipeline initialized with 8 transformer layers on one device and 8 transformer layers on the other device. One pipe is setup across devices 0 and 1 and another across devices 2 and 3. Both pipes are then replicated using DistributedDataParallel.

The only modifications I made to the original tutorial is replacing CUDA device calls with CPU and switching from NCCL backend to GLOO, since NCCL would raise an error (assuming this is because we're using CPU).

I ran the scripts in the following environment (automaticall installs pytorch==0.12.0):
```
conda create -n torchtext_env python=3.8
conda install -c pytorch torchtext==0.13.0 torchdata==0.4.0
conda install -c conda-forge numpy

```

The original tutorial on the Pytorch website has a [bug](https://github.com/pytorch/pytorch/issues/68407), so use [this version](https://github.com/pytorch/tutorials/blob/master/advanced_source/ddp_pipeline.py) on the Pytorch github instead.

