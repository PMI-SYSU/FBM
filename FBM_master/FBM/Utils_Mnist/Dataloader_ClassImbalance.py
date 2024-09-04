# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:11:14 2024

@author: 31458
"""

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataset, class_ratio, batch_size, num_dataset, shuffle_data=False):
        self.custom_images = []
        self.custom_labels = []
        
        self.class_ratio = class_ratio
        self.batch_size = batch_size
        
        self.classes = list(class_ratio.keys())
        self.class_counts = {cls: 0 for cls in self.classes}
        self.classes_len = {k: (v * num_dataset) for k, v in class_ratio.items()}

        
        for image, label in dataset:
            #print((image.view(-1)).shape)
            if (int(label) in self.class_ratio) and (self.class_counts[int(label)] < int(self.classes_len[int(label)])):
                self.custom_images.append(image.view(-1))
                self.custom_labels.append(label)
                self.class_counts[int(label)] += 1
                #print('1')
        #print(self.class_counts)
        #print(len(self.custom_images))
        
        #打乱训练数据
        if shuffle_data==True:
            index = [i for i in range(len(self.custom_images))] # test_data为测试数据
            np.random.shuffle(index) # 打乱索引
            self.custom_images = self.custom_images[index]
            self.custom_labels = self.custom_labels[index]
        
        # 找到最少样本数量的类别
        min_class = min([k for k in self.classes if class_ratio[k] != 0], key=lambda k: self.class_counts[k] // (int(class_ratio[k]*self.batch_size) if class_ratio[k] != 0 else 1))
        #print(min_class)
        self.num_batch = int(self.class_counts[min_class] // (int(class_ratio[min_class]*self.batch_size)))

        # 处理多余的样本
        for cls in self.classes:
            #print('cls', cls)
            #print((self.custom_labels))
            while self.class_counts[cls] > self.num_batch*int(class_ratio[cls]*self.batch_size):
                idx = self.custom_labels.index(cls)  # 找到属于当前类别的第一个样本的索引
                del self.custom_images[idx]  # 删除该样本
                del self.custom_labels[idx]
                self.class_counts[cls] -= 1
        #print(self.class_counts)
        #print(len(self.custom_images))
        
        self.custom_images, self.custom_labels = self.generate_batch()
               

        
    def __len__(self):
        return len(self.custom_labels)
    
    def __getitem__(self, idx):
        return self.custom_images[idx], self.custom_labels[idx]

    def generate_batch(self):
        batch_images = []
        batch_labels = []
        
        remaining_indices = {cls: [i for i, label in enumerate(self.custom_labels) if label == cls] for cls in self.classes}
        #print(remaining_indices)
        samples_per_class = {k: v*self.batch_size for k, v in self.class_ratio.items()}
        #print(samples_per_class)
        
        for batch_ in range(self.num_batch):
            for cls in self.classes:
                if self.class_ratio[cls] == 0:
                    continue
                
                for _ in range(int(samples_per_class[cls])):
                    if remaining_indices[cls]:
                        idx = remaining_indices[cls].pop(0)
                        batch_images.append(self.custom_images[idx])
                        batch_labels.append(self.custom_labels[idx])
            
                #print('',np.sum(batch_labels))

        
        return torch.stack(batch_images), torch.tensor(batch_labels)


