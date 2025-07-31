import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

class ColoredMNIST(Dataset):
    """
    MNIST dataset with spurious color feature correlated with label parity.
    Train: color matches parity with prob=0.9; Test: color matches parity with prob=0.1 (inverted).
    """
    def __init__(self, root, train=True, transform=None, download=True, color_prob=0.9):
        self.mnist = MNIST(root=root, train=train, download=download)
        self.transform = transform or transforms.ToTensor()
        # probability that color matches parity
        self.color_prob = color_prob if train else 1 - color_prob
    
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        # compute parity label: 0 even, 1 odd
        parity = label % 2
        # decide color: match parity or flip
        if random.random() < self.color_prob:
            color = parity
        else:
            color = 1 - parity
        # convert to RGB image with background color
        img = self.transform(img)  # [1,28,28]
        # create color channels: two channels for red and green
        # channel0: red, channel1: green
        c_img = torch.zeros(2, 28, 28)
        c_img[0] = img.squeeze(0) if color == 0 else 0
        c_img[1] = img.squeeze(0) if color == 1 else 0
        # return colored image, parity label, and color (spurious) label
        return c_img, parity, color
