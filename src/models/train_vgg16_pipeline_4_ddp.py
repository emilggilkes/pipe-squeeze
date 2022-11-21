import os
import importlib
import time
import datetime
import tempfile
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.pipeline.sync import Pipe
from torch.nn.parallel import DistributedDataParallel as DDP

device='cuda'


def create_data_loader(batch_size = 32, num_workers = 4):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    train_set = ImageFolder("../data/images/train", transform = train_transform) # "../../data/ImageNet/train"
    val_set   = ImageFolder("../data/images/val", transform = val_transform) #"../../data/ImageNet/val"


    train_loader = DataLoader(
        dataset=train_set,
        batch_size = batch_size,
        num_workers= num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return train_loader, val_loader


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, rank, epochs = 1, plot=False):
    print("Start Training...")

    running_loss = 0
    train_losses, val_losses = [], []
    
    nbatches = len(train_loader)
    for epoch in range(epochs):
        model.train() # Turn on the train mode
        epoch_start_time = time.time()
        batch_start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            print("In train loop")
            print(f"inputs device, labels device, {data[0].device}, {data[1].device}")
            inputs, labels = data[0].to(device), data[1].to(device)
            #inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            logps = model(inputs).local_value()
            # Need to move labels to the device where the output of the
            # pipeline resides.
            print(f"logps device: {logps.device}")
            print(f"Rank: {rank}")
            print(f"Labels device: {labels.device}")
            loss = criterion(logps, labels.to(logps.device))
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            #print('loss.item',loss.item())
            running_loss += loss.item()
            log_interval = 10
            if batch_idx % log_interval == 0 and batch_idx > 0:
                #print('running_loss', running_loss)
                #print('log_interval', log_interval)
                cur_loss = running_loss / log_interval
                elapsed = time.time() - batch_start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch+1, batch_idx, nbatches, scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, np.exp(cur_loss)))
                running_loss = 0
                batch_start_time = time.time()
            # inputs.detach()
            # labels.detach()
            # logps.detach()

        val_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logps = model(inputs).local_value()
                # Need to move labels to the device where the output of the
                # pipeline resides.
                print("In train loop")
                print(f"logps device: {logps.device}")
                print(f"Rank: {rank}")
                print(f"Labels device: {labels.device}")
                # logps = logps.to(labels.device)
                batch_loss = criterion(logps, labels.to(logps.device)) # need to change this depending on num_gpus
                val_loss += batch_loss.item()
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class.to(labels.device) == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                # inputs.detach()
                # labels.detach()
                # logps.detach()
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        print(f"Epoch {epoch+1}/{epochs}   "
                f"Train loss: {running_loss/len(train_loader):.3f}   "
                f"Validation loss: {val_loss/len(val_loader):.3f}   "
                f"Validation accuracy: {accuracy/len(val_loader):.3f}   ",
                f"Epoch time (s): {(time.time() - epoch_start_time):.3f}")
        running_loss = 0
        #scheduler.step()
    
    print("Finished Training")
    #if plot:
        #plt.plot(train_losses, label='Training loss')
        #plt.plot(val_losses, label='Validation loss')
        #plt.legend(frameon=False)
        #plt.savefig(f'plot loss - pipelining 4gpus straight full model {str(datetime.date.today())}.png')
        #plt.show()
    #torch.save(model.state_dict(), f'../../models/pipelining_4gpus_straight full model {str(datetime.date.today())}.pth')
    #for i, stage in enumerate(model):
        #torch.save(stage.state_dict(), f'../../models/pipelining_4gpus_straight stage{i} {str(datetime.date.today())}.pth')


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend="gloo", rank = rank, world_size = world_size)
    print(f"Initiated Process Group with rank = {rank} and world_size = {world_size}")


def cleanup():
    dist.destroy_process_group()


def main(
    rank,
    world_size,
    epochs = 2,
    # batch_size = 1024,
    # learning_rate = 0.003,
    # compression_type=None,
    # save_on_finish=False,
    # use_pipeline_parallel=False,
    # data_set_dirpath=SAMPLE_DATA_SET_PATH_PREFIX,
):
    setup(rank, world_size)

    ##DATASET
    train_loader, val_loader = create_data_loader(batch_size = 4, num_workers = 0)
    tmpfile = tempfile.NamedTemporaryFile()
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1,rpc_backend_options=torch.distributed.rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name)))

    # create stages of the model
    criterion = nn.CrossEntropyLoss()
    module = importlib.import_module("vgg16.gpus=4")
    # arch = module.arch()

    #NOT SURE IF THIS WORKS TO GROUP SPECIFIC GPUS TO A STAGE
    stages = module.model(criterion)
    stage = stages["stage0"].to(torch.device(device, 0))
    # stage = stages["stage0"].to(torch.device(device, 1))
    
    stage = stages["stage1"].to(torch.device(device, 2))
    # stage = stages["stage1"].to(torch.device(device, 3))
    
    model = nn.Sequential(stages)
    model = Pipe(model, chunks=8)

    print ('Total parameters in model: {:,}'.format(get_total_params(model)))
    for i, stage in enumerate(model):
        print ('Total parameters in stage {}: {:,}'.format(i, get_total_params(stage)))
    
    model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr = 0.003, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print(f"Start Training device {rank}")
    for epoch in range(epochs):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            train(model, train_loader, val_loader, optimizer, criterion, scheduler, rank, epoch)

        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=7))

    cleanup()


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = 2
    # data_dir_path = DATA_DIR_MAP[args.data_set]

    mp.spawn(
        main,
        args=(
            world_size,
        ),
        nprocs=world_size,
        join=True,
    )
    ##TRAINING



#https://github.com/pytorch/examples/blob/main/imagenet/main.py
