# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:36:14 2024

@author: 31458
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lam = 0.01



#%%
fermi_surface_list = [0.3, 0.35,0.4,0.455,0.5,0.55,0.6]
Num_data = [1000,2000,3000,4000,5000,6000]
runs = [0,1,2,3,4]


Acc_list_mean = []
Acc_list_std = []
Fermi_D_mean = []
Fermi_D_std = []

for fermi_surface in fermi_surface_list:
    Acc_list_num_mean = []
    Acc_list_num_std = []
    Fermi_D_num_mean = []
    Fermi_D_num_std = []
    for num_dataset in Num_data:
        acc_runs = []
        fermi_runs = []

        for run in runs:
            acc_path = f"data/acc/NUM_test_dF={fermi_surface}_lam={lam}_num={num_dataset}_{run}.csv"
            fermi_path = f"data/distance/NUM_distance_fermi_dF={fermi_surface}_lam={lam}_num={num_dataset}_{run}.csv"
            acc = np.loadtxt(acc_path, delimiter=',')
            fermi = np.loadtxt(fermi_path, delimiter=',')
            acc_runs.append(np.max(acc))
            fermi_runs.append(fermi[-1])
            
        Acc_list_num_mean.append(np.mean(acc_runs))
        Acc_list_num_std.append(np.std(acc_runs))
        Fermi_D_num_mean.append(np.mean(fermi_runs))
        Fermi_D_num_std.append(np.std(fermi_runs))
    
    Acc_list_mean.append(Acc_list_num_mean)
    Acc_list_std.append(Acc_list_num_std)
    Fermi_D_mean.append(Fermi_D_num_mean)
    Fermi_D_std.append(Fermi_D_num_std)
    
#%%
Color_list = [ '#051937', '#33265c', '#6e2a74', '#af207b', '#eb126e', '#F14194','#F161B8']#['#F27970', '#BB9729', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2']


Color_list = [
    '#051937',  # 渐变色1
    '#004d7a',  # 渐变色2
    '#008793',  # 渐变色3
    '#00b287',  # 你指定的颜色
    #'#4A90A4',  # 淡青蓝
    #'#5FCBCB',
    '#2cd261',
    '#a8eb12',  # 暗青绿
    '#D8EA2B',  # 深青绿
]

 #051937, #004d7a, #008793, #00bf72, #a8eb12

#%%
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(4,3),dpi=300)
for i in range (len(fermi_surface_list)):
    plt.errorbar(Num_data, Acc_list_mean[i], yerr=Acc_list_std[i], marker='o',capsize=2, color = Color_list[i],)# label= 'dF='+str(fermi_surface_list[i])
plt.ylabel(r'Accuracy($\%$)')
plt.xlabel('Training data size')


#norm = BoundaryNorm(fermi_surface_list, len(Color_list))
#sm = ScalarMappable(norm=norm, cmap=ListedColormap(Color_list))
#sm.set_array([])
#cb = plt.colorbar(sm)
#cb.set_label(r'$d_F$')
plt.tight_layout()
plt.savefig('./result/ND_acc_MNIST.pdf',bbox_inches='tight')

#plt.legend()
plt.show()

plt.figure(figsize=(4,3),dpi=300)
for i in range (len(fermi_surface_list)):

    plt.errorbar(Num_data, Fermi_D_mean[i], yerr= Fermi_D_std[i], marker='o',capsize=2,  color=Color_list[i], linewidth = 2)
    plt.axhline(y=fermi_surface_list[i], linestyle= '--', color=Color_list[i], linewidth = 2.5, alpha= 0.5)
plt.ylabel('Fermion-pair distance')
plt.xlabel('Training data size')


plt.tight_layout()
plt.savefig('./result/ND_FD_MNIST.pdf',bbox_inches='tight')

plt.show() 
        
    
