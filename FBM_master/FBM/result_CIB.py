# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:36:14 2024

@author: 31458
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fermi_surface_list = [0.4,0.5,0.6]
lam = 0.01
num_dataset = 200
ratio0 = 0.5
ratio1 = 1 - ratio0
class_ratio = {0: ratio0, 1:ratio1}
run = 0

lam = 0.01
num_dataset = 200
runs = [0,1,2,3,4]
ratio_list = [0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62]#[0.35,0.4,0.45,0.5,0.55,0.6,0.65]#
batchsize=50
Fermi_mean = []
Fermi_std = []
Acc_mean = []
Acc_std = []
for fermi_surface in fermi_surface_list:
    Fermi_mean_dF = []
    Fermi_std_dF = []
    Acc_mean_dF = []
    Acc_std_dF = []
    for  ratio0 in ratio_list:
        Acc_list_ratio0=[]
        Fermi_list_ratio0=[]
        for run in runs:
            ratio1 = 1 - ratio0
            class_ratio = {0: ratio0, 1:ratio1}
            acc_path = f"data/acc/CIB_test_dF={fermi_surface}_lam={lam}_num={num_dataset}_batchsize={batchsize}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv"
            fermi_path = f"data/distance/CIB_distance_fermi_dF={fermi_surface}_lam={lam}_num={num_dataset}_batchsize={batchsize}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv"
            acc = np.loadtxt(acc_path, delimiter=',')
            fermi = np.loadtxt(fermi_path, delimiter=',')
            Acc_list_ratio0.append(np.max(acc))
            Fermi_list_ratio0.append(fermi[-1])
        Acc_mean_dF.append(np.mean(Acc_list_ratio0))
        Acc_std_dF.append(np.std(Acc_list_ratio0))
        Fermi_mean_dF.append(np.mean(Fermi_list_ratio0))
        Fermi_std_dF.append(np.std(Fermi_list_ratio0))
    Fermi_mean.append(Fermi_mean_dF)
    Fermi_std.append(Fermi_std_dF)
    Acc_mean.append(Acc_mean_dF)
    Acc_std.append(Acc_std_dF)
        
    
print('acc', acc[-1])
#print('fermi', fermi[-1])



Acc_mean_mlp = []
Acc_std_mlp = []
for  ratio0 in ratio_list:
    Acc_list_ratio0_mlp=[]
    for run in runs:
        ratio1 = 1 - ratio0
        class_ratio = {0: ratio0, 1:ratio1}
        acc_path = f"D:\Japan\work\G_map\FBM_master\MLP\data/acc/CIB_MLP_test_lam={lam}_num={num_dataset}_batchsize={batchsize}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv"
        acc = np.loadtxt(acc_path, delimiter=',')
        Acc_list_ratio0_mlp.append(np.max(acc))
    Acc_mean_mlp.append(np.mean(Acc_list_ratio0_mlp))
    Acc_std_mlp.append(np.std(Acc_list_ratio0_mlp))











Color_list = [ '#051937', '#33265c', '#6e2a74', '#af207b', '#eb126e', '#F14194','#F161B8']#['#F27970', '#BB9729', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2']
Color_list = ['#33265c', '#af207b','#F14194']


plt.figure(figsize=(4,3),dpi=300)
for i in range(len(fermi_surface_list)):
    plt.errorbar(ratio_list, Acc_mean[i], yerr= Acc_std[i],marker='s',markersize=4, capsize=3, color = Color_list[i],label=r'LCLM $d_F$='+str(fermi_surface_list[i]))
plt.errorbar(ratio_list, Acc_mean_mlp, yerr= Acc_std_mlp,marker='s',markersize=4, capsize=3, color='gray', linestyle='--',label=r'MLP')

plt.legend(loc="center left", prop = {'size':8}, markerscale=0.4)
plt.ylabel('Accuracy($\%$)')
plt.xlabel(r'$\rho_0$', size=12)
plt.tight_layout()
plt.savefig('./result/CIB.pdf', bbox_inches='tight')
plt.show()





#%%
fermi_surface_list = [0.4,0.5,0.6]
ratio0_list = [0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62]#[ 0.4,0.42,0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6,  ]#[0.35,0.4, 0.45,0.5,0.55,0.6,0.65]#
batch_size = 50 
Acc_list = []
Fermi_D = []
for fermi_surface in fermi_surface_list:
    for ratio0 in ratio0_list:
        ratio1 = 1 - ratio0
        class_ratio = {0: ratio0, 1:ratio1}
        acc_path = f"data/acc/test_dF={fermi_surface}_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv"
        fermi_path = f"data/distance/distance_fermi_dF={fermi_surface}_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv"
        acc = np.loadtxt(acc_path, delimiter=',')
        fermi = np.loadtxt(fermi_path, delimiter=',')
        Acc_list.append(acc[-1])
        Fermi_D.append(fermi[-1])
#%%
plt.figure(figsize=(4,3),dpi=300)
plt.plot(ratio0_list, Acc_list, marker='o')
plt.ylabel('accuracy')
plt.xlabel('0 ratio (batchsize = '+str(batch_size)+ ', dF='+str(fermi_surface)+')')
plt.show()

plt.figure(figsize=(4,3),dpi=300)
plt.plot(ratio0_list, Fermi_D, marker='o')
plt.ylabel('Fermi Distance')
plt.xlabel('0 ratio (batchsize = '+str(batch_size)+ ', dF='+str(fermi_surface)+')')
plt.show() 
        
    
