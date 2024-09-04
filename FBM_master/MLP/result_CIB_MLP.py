# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:08:31 2024

@author: 31458
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lam = 0.01
num_dataset = 200
runs = [0,1,2,3,4]
ratio_list = [0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62]#[0.35,0.4,0.45,0.5,0.55,0.6,0.65]#
batchsize=50
Fermi_mean = []
Fermi_std = []
Acc_mean = []
Acc_std = []
for  ratio0 in ratio_list:
    Acc_list_ratio0=[]
    #Fermi_list_ratio0=[]
    for run in runs:
        ratio1 = 1 - ratio0
        class_ratio = {0: ratio0, 1:ratio1}
        acc_path = f"data/acc/CIB_MLP_test_lam={lam}_num={num_dataset}_batchsize={batchsize}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv"
        #fermi_path = f"data/distance/CIB_MLP_distance_lam={lam}_num={num_dataset}_batchsize={batchsize}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv"
        acc = np.loadtxt(acc_path, delimiter=',')
        #fermi = np.loadtxt(fermi_path, delimiter=',')
        Acc_list_ratio0.append(np.max(acc))
        #Fermi_list_ratio0.append(fermi[-1])
    Acc_mean.append(np.mean(Acc_list_ratio0))
    Acc_std.append(np.std(Acc_list_ratio0))
    #Fermi_mean.append(np.mean(Fermi_list_ratio0))
    #Fermi_std.append(np.std(Fermi_list_ratio0))
print('acc', acc[-1])
#print('fermi', fermi[-1])

plt.figure(figsize=(4,3),dpi=300)
plt.plot(Acc_mean)
plt.legend('acc_dF={fermi_surface}_num={num_dataset}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}')
plt.ylabel('accuracy')
plt.xlabel(r'$\rho_F$')
plt.show()

#plt.figure(figsize=(4,3),dpi=300)
#plt.plot(Fermi_mean)
#plt.legend('fermi_dF={fermi_surface}_num={num_dataset}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}')

#plt.ylabel('fermi distance')
#plt.xlabel('epoch')
#plt.show()