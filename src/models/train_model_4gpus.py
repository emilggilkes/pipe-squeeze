import os
import argparse
import numpy as np
import json
import tempfile
from tqdm import tqdm
import importlib
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook, bf16_compress_hook

import torch.multiprocessing as mp

#from random_k_reducer import RandomKReducer
from timer import Timer
from grace_random_k import *



SAMPLE_DATA_SET_PATH_PREFIX='../data/images'
IMAGENET_DATA_SET_PATH_PREFIX='../../data/ImageNet'

DATA_DIR_MAP = {
    'sample': SAMPLE_DATA_SET_PATH_PREFIX,
    'ImageNet': IMAGENET_DATA_SET_PATH_PREFIX,
}   

device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '--bs', help='Effective batch size. Will be divided by number of GPUs.', type=int, required=True)
    parser.add_argument('--learning-rate', '--lr', help='Learning rate', type=float, required=True)
    parser.add_argument('--num-procs', help='Number of processes to use', type=int, required=True)
    #parser.add_argument('--num-gpus', help='Number of GPUs to use', type=int, required=True)
    parser.add_argument('--compression-type', help='Type of compression to use. Options are fp16, bf16, PowerSGD, None', default=None, required=False)
    parser.add_argument('--compression-ratio', help='Float representing compression ratio', type=float, default=None, required=False)
    #parser.add_argument('--use-pipeline-parallel', help='Whether or not to use pipeline parallelism (GPipe)', default=False, type=bool)
    parser.add_argument('--data-set', help='Whether to use the small sample dataset or Imagenette dataset', type=str, default='sample')
    parser.add_argument('--save-on-finish', help='Saves model weights upon training completion', type=bool, default=False)
    parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=2)
    return parser


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend="gloo", rank = rank, world_size = world_size)
    print(f"Initiated Process Group with rank = {rank} and world_size = {world_size}")

def cleanup():
    dist.destroy_process_group()

def create_data_loader(rank, world_size, batch_size, data_set_dirpath):
    print(f"Create data loader rank {rank}")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    
    if batch_size % world_size != 0:
        raise Exception("Batch size must be a multiple of the number of workers")

    batch_size = batch_size // world_size
    train_set = ImageFolder(f"{data_set_dirpath}/train", transform = train_transform)
    val_set = ImageFolder(f"{data_set_dirpath}/val", transform = val_transform)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size = batch_size,
        num_workers=world_size,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size = batch_size,
        num_workers = world_size,
        sampler=val_sampler,
    )
    return train_loader, val_loader


def val(model, val_loader, criterion, rank, epoch):
    val_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        val_loader.sampler.set_epoch(epoch)
        for inputs, labels in val_loader:
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            logps = model(inputs.to(torch.device(device, 2 * rank))).local_value()
            batch_loss = criterion(logps, labels.to(torch.device(device, 2 * rank + 1)))
            val_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.cpu().topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape).cpu()
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            inputs.detach()
            labels.detach()
            logps.detach()
   # model.train()

    avg_val_loss = val_loss/len(val_loader)
    val_accuracy = accuracy/len(val_loader)
    return avg_val_loss, val_accuracy


def train(model, train_loader, optimizer, criterion, rank, epoch, timer):
    model.train()
    train_loss = 0
    train_loader.sampler.set_epoch(epoch)
    # start_time = torch.cuda.Event(enable_timing=True)
    # stop_time = torch.cuda.Event(enable_timing=True)
    # time_list = list()    
    
    
    iterator = tqdm(train_loader)
    #print(len(train_loader))
    with timer(f'trainloop_epoch{epoch}_rank{rank}'):
        for batch_idx, data in enumerate(iterator):
            iterator.set_postfix_str(f'RANK: {rank}')
            inputs, labels = data[0].to(torch.device(device,2*rank)), data[1].to(torch.device(device, 2*rank+1))
            #print('n_batches:', len(inputs))
            optimizer.zero_grad()
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            with timer(f'forward_epoch{epoch}_rank{rank}'):
                logps = model(inputs).local_value()
            # need to send labels to device with stage 1
            loss = criterion(logps, labels)
            torch.cuda.synchronize()
            # start_time.record()
            iterator.set_postfix_str(iterator.postfix + f' | LOSS: {loss.item():.4f}')
            #print(f'[RANK {rank}] epoch {epoch} loss = {loss.item():.4f}')
            with timer(f'backward_epoch{epoch}_rank{rank}'):
                loss.backward()
            # stop_time.record()
            torch.cuda.synchronize()
            
            optimizer.step()

            # time_list.append(start_time.elapsed_time(stop_time))
            
            # data_dict = dict()
            # data_dict['timing_log'] = time_list
            # file_name = f"test_random_k_training_{rank}_{epoch}.json"
            # with open(file_name, "w+") as fout:
            #     json.dump(data_dict, fout)

            train_loss += loss.item()
            inputs.detach()
            labels.detach()
            logps.detach()
        
    return train_loss

def train_grace(model, train_loader, optimizer, criterion, rank, epoch, timer, grc):
    model.train()
    train_loss = 0
    train_loader.sampler.set_epoch(epoch)
    # start_time = torch.cuda.Event(enable_timing=True)
    # stop_time = torch.cuda.Event(enable_timing=True)
    # time_list = list()    

    iterator = tqdm(train_loader)
    #print(len(train_loader))
    with timer(f'trainloop_epoch{epoch}_rank{rank}'):
        for batch_idx, data in enumerate(iterator):
            iterator.set_postfix_str(f'RANK: {rank}')
            inputs, labels = data[0].to(torch.device(device,2*rank)), data[1].to(torch.device(device, 2*rank+1))
            optimizer.zero_grad()
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            
            with timer(f'forward_epoch{epoch}_rank{rank}'):
                logps = model(inputs).local_value()
            
            # need to send labels to device with stage 1
            loss = criterion(logps, labels)
            torch.cuda.synchronize()
            # start_time.record()
            iterator.set_postfix_str(iterator.postfix + f' | LOSS: {loss.item():.4f}')
            #print(f'[RANK {rank}] epoch {epoch} loss = {loss.item():.4f}')
            with timer(f"backward_rank{rank}"):
                loss.backward()
            with timer(f'randomk_rank{rank}'):
                for index, (name, parameter) in enumerate(model.named_parameters()):
                    grad = parameter.grad.data
                    new_tensor = grc.step(grad, name)
                    grad.copy_(new_tensor)

            # stop_time.record()
            torch.cuda.synchronize()
            
            optimizer.step()

            # time_list.append(start_time.elapsed_time(stop_time))
            
            # data_dict = dict()
            # data_dict['timing_log'] = time_list
            # file_name = f"test_random_k_training_{rank}_{epoch}.json"
            # with open(file_name, "w+") as fout:
            #     json.dump(data_dict, fout)

            train_loss += loss.item()
            inputs.detach()
            labels.detach()
            logps.detach()
    return train_loss

# class Compressor:
#     def __init__(self, compression_type, compression_ratio=None):
#         self.compression_type = compression_type
#         self.compression_ratio = compression_ratio

# class Trainer:
#     def __init__(self, train_loader, val_loader, optimizer, criterion, compressor=None):
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.compressor = compressor
        
#     def train_pipe_ddp(self):
#         pass
            
        
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

def main(
    rank,
    world_size,
    epochs = 2,
    batch_size = 1024,
    learning_rate = 0.003,
    compression_type=None,
    compression_ratio=None,
    save_on_finish=False,
    data_set_dirpath=SAMPLE_DATA_SET_PATH_PREFIX,
):
    timer = Timer(skip_first=False)

    ## DATASET
    train_loader, val_loader = create_data_loader(rank, world_size, batch_size, data_set_dirpath)

    # We need to initialize the RPC framework with only a single worker since we're using a
    # single process to drive multiple GPUs.
    tmpfile = tempfile.NamedTemporaryFile()
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1,rpc_backend_options=torch.distributed.rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name)))

    # create stages of the model
    module = importlib.import_module("vgg16.gpus=4")
    stages = module.model()
    stage = stages["stage0"].to(torch.device(device, 2*rank))    
    stage = stages["stage1"].to(torch.device(device, 2*rank+1))

    # build model pipeline
    model = nn.Sequential(stages)
    n_microbatches = 8
    model = Pipe(model, chunks=n_microbatches, checkpoint="never")
    
    print ('Total parameters in model: {:,}'.format(get_total_params(model)))
    for i, stage in enumerate(model):
        print ('Total parameters in stage {}: {:,}'.format(i, get_total_params(stage)))
    
    # setup data parallelism
    setup_ddp(rank, world_size)
    model = DDP(model)

    # define train settings
    criterion = nn.CrossEntropyLoss()
    momentum = 0.9
    weight_decay = 0.0001
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum, weight_decay=weight_decay)
    step_size = 50
    gamma = 0.9
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    train_params = record_train_params(rank, world_size, epochs, batch_size, learning_rate, momentum, weight_decay, step_size, gamma)
   
    # Add communication hook doing floating point compression
    if compression_type == 'fp16':
        model.register_comm_hook(state=None, hook=fp16_compress_hook)
    elif compression_type == 'bf16':
        model.register_comm_hook(state=None, hook=bf16_compress_hook)

    # if using random k compression, can't use profiler and need to use train_grace() for training
    if compression_type == 'randomk':
        grc = Allreduce(RandomKCompressor(compression_ratio), NoneMemory(), world_size)
        train_losses, val_losses = [], []
        print(f"Start Training device {rank}")
        for epoch in range(epochs):
            train_loss = train_grace(model, train_loader, optimizer, criterion, rank, epoch, timer,grc)

            train_losses.append(train_loss/len(train_loader))
        
            avg_val_loss, val_accuracy = val(model, val_loader, criterion, rank, epoch)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs}   "
                    f"Device {rank}   "
                    f"Train loss: {train_loss/len(train_loader):.3f}   "
                    f"Validation loss: {avg_val_loss:.3f}   "
                    f"Validation accuracy: {val_accuracy:.3f}")
            scheduler.step()
    else:
        print(f"Start Training device {rank}")
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            #with torch.profiler.profile(
            #activities=[
            #    torch.profiler.ProfilerActivity.CPU,
            #    torch.profiler.ProfilerActivity.CUDA,
            #]
            #) as p:
            train_loss = train(model, train_loader, optimizer, criterion, rank, epoch, timer)
            
            #print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=7))
            #p.export_chrome_trace(f"../../reports/raw_time_data/profiler/trace_epoch{epoch}_rank{rank}.json")
            train_losses.append(train_loss/len(train_loader))
            
            avg_val_loss, val_accuracy = val(model, val_loader, criterion, rank, epoch)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs}   "
                    f"Device {rank}   "
                    f"Train loss: {train_loss/len(train_loader):.3f}   "
                    f"Validation loss: {avg_val_loss:.3f}   "
                    f"Validation accuracy: {val_accuracy:.3f}")
            scheduler.step()

    timer.save_summary(f"../../reports/raw_time_data/timer/rank{rank}_summary_{datetime.now()}.json", train_params)
    print(timer.summary()) 

    cleanup()
    
    if save_on_finish:
        torch.save(model.state_dict(), f'../../models/vgg16_{rank}.pth')
    
    print(f"Finished Training device {rank}")
    #if False:
        #plt.plot(train_losses, label='Training loss')
        #plt.plot(val_losses, label='Validation loss')
        #plt.legend(frameon=False)
        #plt.show()



if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

   
    #world_size = min(torch.cuda.device_count(), args.num_gpus)
    #world_size = 2
    data_dir_path = DATA_DIR_MAP[args.data_set]

    print(f'Using device {device} with device count : {args.num_procs}')
    print(f'Training params:\nEpochs: {args.epochs}\nBatch Size: {args.batch_size}\nLearning Rate: {args.learning_rate}')
    print(f'Compression Type: {args.compression_type}\nCompression Ratio: {args.compression_ratio}\nData dir path: {data_dir_path}')

    mp.spawn(
        main,
        args=(
            args.num_procs,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.compression_type,
            args.compression_ratio,
            args.save_on_finish,
            data_dir_path,
        ),
        nprocs=args.num_procs,
        join=True,
    )
    #print(timer.summary())
