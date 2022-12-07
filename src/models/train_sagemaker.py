import os
import argparse
# import numpy as np
import json
import tempfile
from tqdm import tqdm
import importlib
from datetime import datetime
import logging
import sys

# import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn, optim
# from torchvision import datasets, transforms, models
# from torchvision.datasets import ImageFolder, ImageNet
# from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe
# from torch.utils.data.distributed import DistributedSampler
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook, bf16_compress_hook

# import torch.multiprocessing as mp

#from random_k_reducer import RandomKReducer
from timer import Timer
#from grace_random_k import *
from grace_random_k_comm_hook import RandomKCompressor
from all_reduce_timed import TimedARWrapper
from infinite_data_loader import create_infinite_data_loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# SAMPLE_DATA_SET_PATH_PREFIX='../data/images'
# IMAGENET_DATA_SET_PATH_PREFIX='../../data/ImageNet'

# DATA_DIR_MAP = {
#     'sample': SAMPLE_DATA_SET_PATH_PREFIX,
#     'ImageNet': IMAGENET_DATA_SET_PATH_PREFIX,
# }   

# device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', help='Communication backend. Can only be gloo.', type=str, required=True)
    parser.add_argument('--batch-size', '--bs', help='Effective batch size. Will be divided by number of GPUs.', type=int, required=True)
    parser.add_argument('--n-microbatches', '--mb', help ='Number of micro-batches', type=int, required = True)
    parser.add_argument('--learning-rate', '--lr', help='Learning rate', type=float, required=True)
    # parser.add_argument('--num-procs', help='Number of processes to use', type=int, required=True)
    # parser.add_argument('--num-gpus', help='Number of GPUs to use', type=int, required=True)
    parser.add_argument('--compression-type', help='Type of compression to use. Options are fp16, bf16, PowerSGD, None', default=None, required=False)
    parser.add_argument('--compression-ratio', help='Float representing compression ratio', type=float, default=None, required=False)
    #parser.add_argument('--use-pipeline-parallel', help='Whether or not to use pipeline parallelism (GPipe)', default=False, type=bool)
    # parser.add_argument('--save-on-finish', help='Saves model weights upon training completion', type=bool, default=False, required=False)
    parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=2)

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend="gloo", rank = rank, world_size = world_size)
    print(f"Initiated Process Group with rank = {rank} and world_size = {world_size}")

def cleanup():
    dist.destroy_process_group()


def train(args):
    timer = Timer(skip_first=False)
    device = "cuda" 
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Initialize the distributed environment.
    world_size = len(args.hosts)
    os.environ["WORLD_SIZE"] = str(world_size)
    host_rank = args.hosts.index(args.current_host)
    os.environ["RANK"] = str(host_rank)
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    


    # We need to initialize the RPC framework with only a single worker since we're using a
    # single process to drive multiple GPUs.
    tmpfile = tempfile.NamedTemporaryFile()
    torch.distributed.rpc.init_rpc('worker', backend=torch.distributed.rpc.BackendType.TENSORPIPE, rank=0, world_size=1,rpc_backend_options=torch.distributed.rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name),
        _transports=["uv"]),)

    # create stages of the model
    module = importlib.import_module("vgg19.gpus=4_4_pretrained")
    stages = module.model()
    stages["stage0"] = stages["stage0"].to(torch.device(device, 0)) # 2*host_rank))    
    stages["stage1"] = stages["stage1"].to(torch.device(device, 1)) # 2*host_rank+1))
    stages["stage2"] = stages["stage2"].to(torch.device(device, 2)) # 2*host_rank+1))
    stages["stage3"] = stages["stage3"].to(torch.device(device, 3)) # 2*host_rank+1))
    # stage = stages["stage0"].to(torch.device(device, host_rank)) # 2*host_rank))    
    # stage = stages["stage1"].to(torch.device(device, host_rank + 1)) # 2*host_rank+1))

    # build model pipeline
    model = nn.Sequential(stages)
    model = Pipe(model, chunks=args.n_microbatches, checkpoint="never")
    logger.info('Total parameters in model: {:,}'.format(get_total_params(model)))
    for i, stage in enumerate(model):
        logger.info('Total parameters in stage {}: {:,}'.format(i, get_total_params(stage)))
    
    # setup data parallelism
    # setup_ddp(host_rank, world_size)
    # dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
    dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
    logger.info(
        "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
            args.backend, dist.get_world_size()
        )
        + "Current host rank is {}. Number of gpus: {} | cuda.device_count: {}".format(dist.get_rank(), args.num_gpus, torch.cuda.device_count())
    )
    
    train_loader, val_loader = create_infinite_data_loader(host_rank, world_size, args.batch_size, args.data_dir, args.num_gpus)


    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),  # type: ignore
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(val_loader.sampler),
            len(val_loader.dataset),
            100.0 * len(val_loader.sampler) / len(val_loader.dataset),
        )
    )

    model = DDP(model)

    if args.compression_type == 'randomk':
        compressor = RandomKCompressor(args.compression_ratio, timer)
        model.register_comm_hook(state=None, hook=compressor.random_k_compress_hook)
    else:
        all_reduce_wrapper = TimedARWrapper(timer)
        model.register_comm_hook(state=None, hook=all_reduce_wrapper.reduce)
        
    # Add communication hook doing floating point compression
    if args.compression_type == 'fp16':
        model.register_comm_hook(state=None, hook=fp16_compress_hook)
    elif args.compression_type == 'bf16':
        model.register_comm_hook(state=None, hook=bf16_compress_hook)

    # define train settings
    criterion = nn.CrossEntropyLoss()
    momentum = 0.9
    weight_decay = 0.0001
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=momentum, weight_decay=weight_decay)
    step_size = 50
    gamma = 0.9
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    train_params = record_train_params(host_rank, world_size, args.epochs, args.batch_size, args.learning_rate, momentum, weight_decay, step_size, gamma)
   
    logger.info(f"Start Training device {host_rank}")
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_loader.sampler.set_epoch(epoch)
        with timer(f'trainloop_epoch{epoch}_rank{host_rank}'):
            for inputs, labels in tqdm(train_loader):
                optimizer.zero_grad()
                # Since the Pipe is only within a single host and process the ``RRef``
                # returned by forward method is local to this node and can simply
                # retrieved via ``RRef.local_value()``.
                with timer(f'forward_epoch{epoch}_rank{host_rank}'):
                    logps = model(inputs.to(torch.device(device, 0))).local_value() # 2*host_rank))).local_value()
                
                # need to send labels to device with stage 3
                loss = criterion(logps, labels.to(torch.device(device, 3))) # 2*host_rank+1)))

                with timer(f'backward_epoch{epoch}_rank{host_rank}'):
                    loss.backward()
                        
                optimizer.step()
                train_loss += loss.item()
                inputs.detach()
                labels.detach()
                logps.detach()
        scheduler.step()

        val_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            val_loader.sampler.set_epoch(epoch)
            for inputs, labels in val_loader:
                # Since the Pipe is only within a single host and process the ``RRef``
                # returned by forward method is local to this node and can simply
                # retrieved via ``RRef.local_value()``.
                logps = model(inputs.to(torch.device(device, 0))).local_value() # 2 * host_rank))).local_value()
                batch_loss = criterion(logps, labels.to(torch.device(device, 3)))
                val_loss += batch_loss.item()
                ps = torch.exp(logps)
                top_p, top_class = ps.cpu().topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape).cpu()
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                inputs.detach()
                labels.detach()
                logps.detach()

        avg_val_loss = val_loss/len(val_loader)
        val_accuracy = accuracy/len(val_loader)

        train_losses.append(train_loss/len(train_loader))
        
        #avg_val_loss, val_accuracy = val(model, val_loader, criterion, host_rank, epoch)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}   "
                    f"Device {host_rank}   "
                    f"Train loss: {train_loss/len(train_loader):.3f}   "
                    f"Validation loss: {avg_val_loss:.3f}   "
                    f"Validation accuracy: {val_accuracy:.3f}")
    timer_summary_pth = os.path.join(args.output_data_dir, f"rank{host_rank}_{args.n_microbatches}_{args.compression_type}_{args.compression_ratio}_{datetime.now()}.json")
    timer.save_summary(timer_summary_pth, train_params)
    # timer.save_summary(f"../../reports/raw_time_data/timer/rank{host_rank}_{args.n_microbatches}_{args.compression_type}_{args.compression_ratio}_{datetime.now()}.json", train_params)
    logger.info(timer.summary()) 
    performance_df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses, 'val_accuracy': val_accuracies})
    performance_summary_pth = os.path.join(args.output_data_dir, f"rank{host_rank}_{args.n_microbatches}_{args.compression_type}_{args.compression_ratio}_{datetime.now()}.csv")
    performance_df.to_csv(performance_summary_pth)
    # performance_df.to_csv(f"../../reports/model_metrics/rank{host_rank}_{args.n_microbatches}_{args.compression_type}_{args.compression_ratio}_{datetime.now()}.csv")
    
    cleanup()



def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def record_train_params(rank, world_size, epochs, batch_size, learning_rate, momentum, weight_decay, scheduler_step_size, gamma):
    train_params = {}
    train_params['rank'] = rank
    train_params['world_size'] = world_size
    train_params['epochs'] = epochs
    train_params['batch_size'] = batch_size
    train_params['learning_rate'] = learning_rate
    train_params['momentum'] = momentum
    train_params['weight_decay'] = weight_decay
    train_params['scheduler_step_size'] = scheduler_step_size
    train_params['gamma'] = gamma

    return train_params


    


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()
   
    #data_dir_path = DATA_DIR_MAP[args.data_set]

    

    train(args)

   
    
    
