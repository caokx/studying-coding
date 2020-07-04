import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2


class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    batch_size = 200
    learning_rate = 0.01
    epochs = 10

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose(
                           [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.Compose(
                           [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)

    for data, target in test_loader:
        print(data.shape)
        print(target.shape)
        break
