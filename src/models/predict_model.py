import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader



def create_data_loader(batch_size = 32, num_workers = 4):
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    test_set = ImageFolder('../data/images/test', transform = test_transform)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle=True,
    )
    return test_loader


def predict(model, test_loader):
    print("Start Predicting...")
    test_losses = []
    test_loss = 0
    accuracy = 0
    vgg19.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = vgg19.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            inputs.detach()
            labels.detach()
            logps.detach()
    print(f"Test loss: {test_loss/len(test_loader):.3f}   "
            f"Test accuracy: {accuracy/len(test_loader):.3f}")
    print("Finished Predicting")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##MODEL
    vgg19 = torch.load("../../models/vgg19.pth")
    vgg19.to(device)

    ##DATASET
    test_loader = create_data_loader(batch_size = 8, num_workers = 4)

    ##PREDICTING
    predict(vgg19, test_loader)




#https://github.com/pytorch/examples/blob/main/imagenet/main.py