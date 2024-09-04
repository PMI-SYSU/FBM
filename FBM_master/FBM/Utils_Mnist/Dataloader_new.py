# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 21:17:47 2024

@author: 31458
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from collections import defaultdict

class BalancedMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.dataset = datasets.MNIST(root, train=train, download=True, transform=transform)
        self.label_to_indices = defaultdict(list)

        # 建立标签到索引的映射
        for i, (_, label) in enumerate(self.dataset):
            self.label_to_indices[label].append(i)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, num_classes):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.label_to_indices = dataset.label_to_indices
        self.batch = [[] for _ in range(self.num_classes)]
        self.index_queue = [0] * self.num_classes

    def __iter__(self):
        while True:
            for i in range(self.num_classes):
                try:
                    self.batch[i].append(self.label_to_indices[i][self.index_queue[i]])
                    self.index_queue[i] += 1
                except IndexError:
                    self.index_queue[i] = 0
                    self.batch[i] = []
                    self.batch[i].append(self.label_to_indices[i][self.index_queue[i]])
                    self.index_queue[i] += 1

                if len(self.batch[i]) == self.batch_size // self.num_classes:
                    yield from self.batch[i]
                    self.batch[i] = []

    def __len__(self):
        return len(self.dataset) // self.batch_size

# 创建数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

dataset = BalancedMNISTDataset('D:\Japan\work\G_map\FBM_master\Experiment\Mnist', train=True, transform=transform)

# 创建 Dataloader
batch_size = 32
num_classes = 10
sampler = BalancedBatchSampler(dataset, batch_size, num_classes)
dataloader = DataLoader(dataset, batch_sampler=sampler)

# 使用 Dataloader 迭代数据
for images, labels in dataloader:
    # 在这里处理批量数据
    print(images.shape)  # torch.Size([32, 1, 28, 28])
    print(labels.unique(return_counts=True))  # (tensor([0, 1, 2, ..., 9]), tensor([3, 3, 3, ..., 3]))