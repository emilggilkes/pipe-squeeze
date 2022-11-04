import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend="nccl", rank = rank, world_size = world_size)
    print(f"Initiated Process Group with rank = {rank} and world_size = {world_size}")

def cleanup():
    dist.destroy_process_group()

def create_data_loader(rank, world_size, batch_size):
    print(f"Create data loader device {rank}")
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
    train_set = ImageFolder("../data/images/train", transform = train_transform)
    val_set   = ImageFolder("../data/images/val", transform = val_transform)

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


def train(rank, world_size, epochs = 10, batch_size = 32):
    setup(rank, world_size)

    vgg19 = models.vgg19(weights = None)
    vgg19.to(rank)
    vgg19 = DDP(vgg19, device_ids=[rank], output_device=rank)

    train_loader, val_loader = create_data_loader(rank, world_size, batch_size)

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(vgg19.parameters(), lr = 0.003, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print(f"Start Training device {rank}")
    running_loss = 0
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            logps = vgg19.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            inputs.detach()
            labels.detach()
            logps.detach()

        val_loss = 0
        accuracy = 0
        vgg19.eval()
        with torch.no_grad():
            val_loader.sampler.set_epoch(epoch)
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                logps = vgg19.forward(inputs)
                batch_loss = criterion(logps, labels)
                val_loss += batch_loss.item()
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                inputs.detach()
                labels.detach()
                logps.detach()
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        print(f"Epoch {epoch+1}/{epochs}   "
		f"Device {rank}   "
                f"Train loss: {running_loss/len(train_loader):.3f}   "
                f"Validation loss: {val_loss/len(val_loader):.3f}   "
                f"Validation accuracy: {accuracy/len(val_loader):.3f}")
        running_loss = 0
        vgg19.train()
    cleanup()
    
    torch.save(vgg19.state_dict(), f'../../models/vgg19_{rank}.pth')
    
    print(f"Finished Training device {rank}")
    if False:
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = min(torch.cuda.device_count(), 2)
    print(f'Using device {device} with device count : {n}')
    world_size = n
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size
    )

#https://github.com/pytorch/examples/blob/main/imagenet/main.py
