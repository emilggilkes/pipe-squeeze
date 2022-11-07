import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader



def create_data_loader(batch_size = 32, num_workers = 4):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    train_set = ImageFolder("../data/images/train", transform = train_transform)
    val_set   = ImageFolder("../data/images/val", transform = val_transform)


    train_loader = DataLoader(
        dataset=train_set,
        batch_size = batch_size,
        num_workers= num_workers,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle=True,
    )
    return train_loader, val_loader


def train(model, train_loader, val_loader, epochs = 1, plot = True):
    print("Start Training...")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(vgg19.parameters(), lr = 0.003, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    running_loss = 0
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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
                f"Train loss: {running_loss/len(train_loader):.3f}   "
                f"Validation loss: {val_loss/len(val_loader):.3f}   "
                f"Validation accuracy: {accuracy/len(val_loader):.3f}")
        running_loss = 0
        vgg19.train()
    torch.save(vgg19, '../../models/vgg19.pth')
    print("Finished Training")
    if plot:
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##MODEL
    vgg19 = models.vgg19(weights = None)
    vgg19.to(device)

    ##DATASET
    train_loader, val_loader = create_data_loader(batch_size = 4*4)

    ##TRAINING
    train(vgg19, train_loader, val_loader)




#https://github.com/pytorch/examples/blob/main/imagenet/main.py