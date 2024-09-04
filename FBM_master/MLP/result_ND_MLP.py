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

        acc_path = f"data/acc/NUM_test_dF={fermi_surface}_num={num_dataset}_{run}.csv"
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
plt.xlabel('number of dataset')



#plt.legend()
plt.show()

#%%
xx = range(len([0,1,2]))
tra_mlp_mean =[0,1,2]
tra_mlp_min =tra_mlp_mean
tra_mlp_max = tra_mlp_mean


tra_fbm_new_mean = tra_mlp_mean
tra_fbm_new_max = tra_mlp_mean
tra_fbm_new_min = tra_mlp_mean


test_mlp_mean = tra_mlp_mean
test_mlp_min = tra_mlp_mean
test_mlp_max = tra_mlp_mean

test_fbm_new_mean= tra_mlp_mean
test_fbm_new_min = tra_mlp_mean
test_fbm_new_max = tra_mlp_mean


fig = plt.figure(figsize=(4, 3),dpi=600)
ax1 = fig.add_subplot(111)

ax1.plot(xx, tra_mlp_mean, color='#93B5E3', linestyle='--',label = 'MLP Train',linewidth=2)
ax1.fill_between(xx,tra_mlp_min, tra_mlp_max, color='#93B5E3', alpha=0.2)

ax1.plot(xx, tra_fbm_new_mean,color='#416C9E', linestyle='-',label = 'LCLM Train',linewidth=2)
ax1.fill_between(xx,tra_fbm_new_min, tra_fbm_new_max, color='#416C9E', alpha=0.2)
ax1.set_ylim(72,102)
# ax1.set_yticks([75, 80,85,90,95,100])
# ax1.set_yticklabels([75, 80,85,90,95,100],fontsize=10)
# ax1.set_xticks([0,20,40,60,80,100])
# ax1.set_xticklabels([0,20,40,60,80,100],fontsize=10)


#ax2 = ax1.twinx()
ax1.plot(xx, test_mlp_mean, color='#FF8F7A', linestyle='--', label = 'MLP Test',linewidth=2)
ax1.fill_between(xx,test_mlp_min, test_mlp_max, color='#FF8F7A', alpha=0.1)

ax1.plot(xx, test_fbm_new_mean,color='#D40D12', label = 'LCLM Test',linewidth=2)
ax1.fill_between(xx,test_fbm_new_min, test_fbm_new_max, color='#D40D12', alpha=0.1)
#ax2.set_ylim(55,95)
#ax2.set_yticks([60,75,90])
#ax2.set_yticklabels([60,75,90],fontsize=20)

ax1.set_xlabel("Epoch",fontsize=15)
ax1.set_ylabel("Accuracy (%)",fontsize=15)
# ax2.set_ylabel("Accuracy (%)",fontsize=30,rotation=-90,labelpad=33)


# ax2.tick_params(axis='y', labelrotation=-90)
# ax1.tick_params(axis='y', labelrotation=90)
# plt.tight_layout()

plt.yticks([80,90,100],fontsize=10)
plt.xticks([0,25,50,75,100],fontsize=10)
legend = ax1.legend(loc='center right',bbox_to_anchor=(0.99, 0.70),fontsize=10)
# legend = ax1.legend(loc='center right',fontsize=7)
frame = legend.get_frame()
frame.set_alpha(0.7)

plt.savefig('1',bbox_inches = 'tight')
plt.show()

#%%
plt.figure(figsize=(4,1.8),dpi=300)
x = [0,1,2]
acc_mlp =[0,1,2]
plt.plot(x, acc_mlp, color='#C3A3DF', linestyle='--', label='MLP',linewidth=1.8)
acc_mlp_min = acc_mlp
acc_mlp_max = acc_mlp
n = len(x)
inter = 4

xp = x
yp= x
ypstd =x
acc_yp_min = np.array(yp) 
acc_yp_max = np.array(yp) 
plt.errorbar(xp,y=yp, yerr=ypstd, color='#6B687B',marker= 's',\
             ms=3,linewidth=1, linestyle='-',label='LCLM',capsize=2,capthick=1, ecolor='#C6C2DB')
        
plt.legend()
plt.show()    
#%%
area_WN =x

plt.figure(figsize=(4,1.8),dpi=300)

plt.hlines(y=4.288, xmin=-0.05, xmax=1.21, linestyle='--', linewidth=3, color='#CFB0E8', label='MLP')
plt.plot(x, area_WN,  marker='s',ms=4, linestyle='-', color='#858597',label='LCLM')
plt.xlabel(r'$\epsilon$')
plt.xlabel(r'$ACC_{{\ell}_2}$($\%$)')
plt.legend()
plt.show()    

