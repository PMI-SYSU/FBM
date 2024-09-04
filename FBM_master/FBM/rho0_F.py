
import numpy as np
import torch 



import torch

shape = 50

ratio = 0.4

import torch

def generate_binary_tensor(size, ratio):
    # 计算 0 和 1 的数量
    num_ones =  int(size * ratio)
    num_zeros = size - num_ones
    
    # 创建一个大小为 size 的张量，其中包含指定比例的 0 和 1
    binary_tensor = torch.cat((torch.zeros(num_zeros), torch.ones(num_ones)))
    
    # 将张量打乱顺序
    binary_tensor = binary_tensor[torch.randperm(size)]
    
    return binary_tensor

# 定义张量大小和所需的 0 和 1 的比例
size = 50
ratio = 0.38  # 0 和 1 的比例为 2:3

# 生成包含指定比例的 0 和 1 的张量


def batch_pair(batch_labels):
    '''
        费米子对赋予标签为（-），可以通过反对易关系记忆
        玻色子对赋予标签为（+），可以通过正对易关系记忆
    '''
    labels = []
    batch_size = batch_labels.size()[0]
    for i in range(batch_size):
        for j in range(batch_size):
            if j > i:
                point1 = batch_labels[i].unsqueeze(0)
                point2 = batch_labels[j].unsqueeze(0)
                pair_data = torch.cat((point1, point2), dim=0)
                if batch_labels[i] == batch_labels[j]:
                    labels.append(0)
                else:
                    labels.append(1)
    r = np.sum(labels)/len(labels)

    return r

# 定义起始值、结束值和步长
start = 0.38
stop = 0.62
step = 0.02

# 生成数组
ratios0 = np.arange(start, stop + step, step)

ratiosF = []
for ratio in ratios0:
    binary_tensor = generate_binary_tensor(size, ratio)
    r = batch_pair(binary_tensor)
    ratiosF.append(r)
    print(r)
    
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(4,3),dpi=300)
plt.plot(ratios0, ratiosF, linewidth=3)
plt.xlabel(r'$\rho_0$', size=25)
plt.ylabel(r'$\rho_F$', size=25)
plt.yticks([0.48, 0.49,0.50,0.51])
plt.xticks([0.4, 0.5,0.6])
plt.tick_params(axis='both', labelsize=15) 
plt.tight_layout()
plt.savefig('./result/rho.pdf',bbox_inches='tight')
plt.show()

