import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class MyCifar10(Dataset):
    def __init__(self, path, transform, train=True):
        self.cifar10 = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
        self.transforms = transform
        self.least_image = 2500
        self.newset = self.create_new_set()
    def create_new_set(self):
        bird = []
        deer = []
        truck = []
        other = []
        for t in self.cifar10:
            if t[1] == 2:
                bird.append(t)
            elif t[1] == 4:
                deer.append(t)
            elif t[1] == 9:
                truck.append(t)
            else:
                other.append(t)
        random.seed(10)
        bird_ = random.sample(bird, self.least_image)
        random.seed(20)
        deer_ = random.sample(deer, self.least_image)
        random.seed(30)
        truck_ = random.sample(truck, self.least_image)
        data = other + bird_ + deer_ + truck_
        random.seed(40)
        data = random.sample(data, len(data))
        return data
    def __len__(self):
        return len(self.newset)
    def __getitem__(self, index):
        im, label = self.newset[index]
        return self.transforms(im), label