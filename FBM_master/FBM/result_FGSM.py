# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 10:04:34 2024

@author: 31458
"""

import numpy as np

epsilons = np.arange(0, 1, 0.01)
DF = [0.4,0.5,0.8,1]
num_dataset =1000
runs = [0,1,2,3,4]
Acc_FBM_mean=[]
Acc_FBM_std=[]
for dF in DF:
    acc_FBM = []
    for run in runs:
        acc = np.loadtxt(f"data/FGSM_acc/FM_FGSM_acc_dF={dF}_num={num_dataset}_{run}.csv", delimiter=',')
        
        acc_FBM.append(acc)
    Acc_FBM_mean_dF = np.mean(np.array(acc_FBM), axis = 0)
    Acc_FBM_std_dF = np.std(np.array(acc_FBM), axis = 0)
    select_mean = Acc_FBM_mean_dF[::5]
    select_std = Acc_FBM_std_dF[::5]
    Acc_FBM_mean.append(Acc_FBM_mean_dF[::5])
    Acc_FBM_std.append(Acc_FBM_std_dF[::5])
    

Acc_MLP = []
for run in runs:
    acc = np.loadtxt(f"D:\Japan\work\G_map\FBM_master/MLP/data/FGSM_acc/FM_FGSM_MLP_acc_num={num_dataset}_{run}.csv", delimiter=',')
    Acc_MLP.append(acc)
 
Acc_MLP_mean = (np.mean(np.array(Acc_MLP), axis = 0))[::5]
Acc_MLP_std = (np.std(np.array(Acc_MLP), axis = 0))[::5]

epsilons0 = epsilons[::5]



import matplotlib.pyplot as plt

colors = ['#100150', '#4D028F','#B0007F','#EA2462', '#FF7449','#FFB946']#['#100150', '#4D028F','#B0007F','#EA2462','#FF7449','#FFB946']
fig = plt.figure(figsize= (4,3), dpi=500)
ax = fig.add_subplot(111)
for i in range(len(DF)):
    ax.errorbar(epsilons0,Acc_FBM_mean[i],yerr=Acc_FBM_std[i], marker='s',markersize=4,capsize=2, color=colors[i],label=r'LCLM $d_F$='+str(DF[i]))
ax.errorbar(epsilons0, Acc_MLP_mean, yerr=Acc_MLP_std,marker='s',markersize=4, capsize=2, markerfacecolor='None', color='#CFB0E8', label = 'MLP',)

plt.xlabel(r"$\epsilon$",fontsize=15)
plt.ylabel(r'$ACC_{LCLM}$ ($\%$)', fontsize=12)
plt.yticks(fontsize=10)
plt.legend(fontsize=8)

plt.savefig('./result/FGSM.pdf', bbox_inches='tight')
plt.show()


