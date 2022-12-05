import os
from torchvision import datasets, transforms, models
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_infinite_data_loader(rank, world_size, batch_size, data_set_dirpath, n_workers): #(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      #rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):

    
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
    print(f"World size: {world_size}, setting effective batch size to {batch_size}. Should be batch size / num input gpus.")

    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, n_workers*2])  # number of workers

    train_set = ImageFolder(f"{data_set_dirpath}/train", transform = train_transform)
    val_set = ImageFolder(f"{data_set_dirpath}/val", transform = val_transform)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = InfiniteDataLoader(train_set,
                  batch_size=batch_size,
                  num_workers=nw,
                  sampler=train_sampler,
                  pin_memory=True,
                  )
    val_loader = InfiniteDataLoader(val_set,
                  batch_size=batch_size,
                  num_workers=nw,
                  sampler=val_sampler,
                  pin_memory=True,
                  )
    return train_loader, val_loader


class InfiniteDataLoader(DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)  # type: ignore

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
