# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:36:14 2024

@author: 31458
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fermi_surface = 0.4
lam = 0.01
num_dataset = 200
ratio0 = 0.52
ratio1 = 1 - ratio0
class_ratio = {0: ratio0, 1:ratio1}
run = 0


#%%
fermi_surface_list = [0.3, 0.35,0.4,0.455,0.5,0.55,0.6]
Num_data = [1000,2000,3000,4000,5000,6000,8000,10000]

Acc_list = []

for fermi_surface in fermi_surface_list:
    Acc_list_num = []
    Fermi_D_num = []
    for num_dataset in Num_data:

        acc_path = f"data/acc/FM_test_dF={fermi_surface}_num={num_dataset}_{run}.csv"
        acc = np.loadtxt(acc_path, delimiter=',')
        Acc_list_num.append(np.max(acc))
    
    Acc_list.append(Acc_list_num)
    
#%%
Color_list = [ '#051937', '#33265c', '#6e2a74', '#af207b', '#eb126e', '#F14194','#F161B8']#['#F27970', '#BB9729', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2']
#%%
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

plt.figure(figsize=(4,3),dpi=300)
for i in range (len(fermi_surface_list)):
    plt.plot(Num_data, Acc_list[i], marker='o', color = Color_list[i],)# label= 'dF='+str(fermi_surface_list[i])
plt.ylabel('accuracy')
plt.xlabel('training data size')



#plt.legend()
plt.show()


        
    
