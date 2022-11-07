import os
import argparse
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
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook, bf16_compress_hook

import torch.multiprocessing as mp


SAMPLE_DATA_SET_PATH_PREFIX='../data/images'
IMAGENET_DATA_SET_PATH_PREFIX='../data/ImageNet'

DATA_DIR_MAP = {
    'sample': SAMPLE_DATA_SET_PATH_PREFIX,
    'ImageNet': IMAGENET_DATA_SET_PATH_PREFIX,
}   


def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '--bs', help='Effective batch size. Will be divided by number of GPUs.', type=int, required=True)
    parser.add_argument('--learning-rate', '--lr', help='Learning rate', type=float, required=True)
    parser.add_argument('--num-gpus', help='Number of GPUs to use', type=int, required=True)
    parser.add_argument('--compression-type', help='Type of compression to use. Options are fp16, bf16, PowerSGD, None', default=None, required=False)
    parser.add_argument('--use-pipeline-parallel', help='Whether or not to use pipeline parallelism (GPipe)', default=False, type=bool)
    parser.add_argument('--data-set', help='Whether to use the small sample dataset or Imagenette dataset', type=str, default='sample')
    parser.add_argument('--save-on-finish', help='Saves model weights upon training completion', type=bool, default=False)
    parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=2)
    return parser


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend="nccl", rank = rank, world_size = world_size)
    print(f"Initiated Process Group with rank = {rank} and world_size = {world_size}")

def cleanup():
    dist.destroy_process_group()

def create_data_loader(rank, world_size, batch_size, data_set_dirpath):
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
    train_set = ImageFolder(f"{data_set_dirpath}/train", transform = train_transform)
    val_set   = ImageFolder(f"{data_set_dirpath}/val", transform = val_transform)

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
            inputs, labels = inputs.to(rank), labels.to(rank)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            val_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            inputs.detach()
            labels.detach()
            logps.detach()
    model.train()

    avg_val_loss = val_loss/len(val_loader)
    val_accuracy = accuracy/len(val_loader)
    return avg_val_loss, val_accuracy


def train(model, train_loader, optimizer, criterion, rank, epoch):
    train_loss = 0
    train_loader.sampler.set_epoch(epoch)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(rank), labels.to(rank)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        inputs.detach()
        labels.detach()
        logps.detach()
    return train_loss


def main(
    rank,
    world_size,
    epochs = 2,
    batch_size = 1024,
    learning_rate = 0.003,
    compression_type=None,
    save_on_finish=False,
    use_pipeline_parallel=False,
    data_set_dirpath=SAMPLE_DATA_SET_PATH_PREFIX,
):
    setup(rank, world_size)

    vgg19 = models.vgg19(weights = None)
    vgg19.to(rank)
    vgg19 = DDP(vgg19, device_ids=[rank], output_device=rank)

    if compression_type == 'fp16':
        vgg19.register_comm_hook(state=None, hook=fp16_compress_hook)
    elif compression_type == 'bf16':
        vgg19.register_comm_hook(state=None, hook=bf16_compress_hook)

    train_loader, val_loader = create_data_loader(rank, world_size, batch_size, data_set_dirpath)

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(vgg19.parameters(), lr = learning_rate, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print(f"Start Training device {rank}")
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            train_loss = train(vgg19, train_loader, optimizer, criterion, rank, epoch)

        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=7))
        p.export_chrome_trace(f"trace_nocomp_epoch_{epoch}_{rank}.json")
        train_losses.append(train_loss/len(train_loader))
        
        avg_val_loss, val_accuracy = val(vgg19, val_loader, criterion, rank, epoch)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}   "
		f"Device {rank}   "
                f"Train loss: {train_loss/len(train_loader):.3f}   "
                f"Validation loss: {avg_val_loss:.3f}   "
                f"Validation accuracy: {val_accuracy:.3f}")

    cleanup()
    
    if save_on_finish:
        torch.save(vgg19.state_dict(), f'../../models/vgg19_{rank}.pth')
    
    print(f"Finished Training device {rank}")
    if False:
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = min(torch.cuda.device_count(), args.num_gpus)
    data_dir_path = DATA_DIR_MAP[args.data_set]

    print(f'Using device {device} with device count : {world_size}')
    print(f'Training params:\nEpochs: {args.epochs}\nBatch Size: {args.batch_size}\nLearning Rate: {args.learning_rate}')
    print(f'Compression Type: {args.compression_type}\nPipelining: {args.use_pipeline_parallel}\nData dir path: {data_set_dirpath}')

    mp.spawn(
        main,
        args=(
            world_size,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.compression_type,
            args.save_on_finish,
            args.use_pipeline_parallel,
            data_dir_path,
        ),
        nprocs=world_size,
        join=True,
    )
