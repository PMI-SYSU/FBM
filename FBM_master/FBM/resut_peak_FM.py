# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 10:08:37 2024

@author: 31458
"""
import numpy as np
import matplotlib.pyplot as plt

# 测试集
### MLP
final_acc_test_MLP0 = np.loadtxt('D:\Japan\work\G_map\FBM_master\MLP/data/acc/FM_MLP_test_num=1000_0.csv')
final_acc_test_MLP1 = np.loadtxt('D:\Japan\work\G_map\FBM_master\MLP/data/acc/FM_MLP_test_num=1000_1.csv')
final_acc_test_MLP2 = np.loadtxt('D:\Japan\work\G_map\FBM_master\MLP/data/acc/FM_MLP_test_num=1000_2.csv')
final_acc_test_MLP3 = np.loadtxt('D:\Japan\work\G_map\FBM_master\MLP/data/acc/FM_MLP_test_num=1000_3.csv')
final_acc_test_MLP4 = np.loadtxt('D:\Japan\work\G_map\FBM_master\MLP/data/acc/FM_MLP_test_num=1000_4.csv')

final_acc_test_MLP=[final_acc_test_MLP0,final_acc_test_MLP1,final_acc_test_MLP2,final_acc_test_MLP3,final_acc_test_MLP4]
## 简单统计
runs = [0,1,2,3,4]
mlp_list = []
for run in runs:
    _ = final_acc_test_MLP[run]
    acc = np.mean(_[-10:])
    mlp_list.append(acc)
acc_mlp = np.mean(mlp_list)
std_mlp = np.std(mlp_list)

fermi_surface_list = [0.1,0.12,0.15,0.18,0.2,0.22,0.25,0.28,0.3,0.32,0.35,0.38,0.4,0.42,\
                      0.455,0.48,0.5,0.52,0.55,0.58,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
                #0.12,0.18,0.22,0.28,0.32,0.58,0.62,0.85,0.9,0.95,1
acc_fbm_list=[]
std_fbm_list=[]

lam=0.01
num_dataset=1000 

for d_F in fermi_surface_list:
    fbm_list=[]
    print("\nd_F=",d_F)
    for run in range(5):
        _ = np.loadtxt('./data/acc/FM_test_dF={}_lam={}_num={}_{}.csv'.format(d_F,lam,num_dataset,run))
        print('lam={}_{}.csv'.format(lam,run))
        acc = np.max(_)
        fbm_list.append(acc)
    acc_fbm = np.mean(fbm_list)
    std_fbm = np.std(fbm_list)

    acc_fbm_list.append(acc_fbm)
    std_fbm_list.append(std_fbm)
    

#plt.figure(figsize=(4,1.8),dpi=300)
#plt.plot(x, [acc_mlp] * len(x), color='#C3A3DF', linestyle='--', label='MLP',linewidth=1.8)
#acc_mlp_min = acc_mlp-std_mlp
#acc_mlp_max = acc_mlp+std_mlp
#n = len(x)
#inter = 4

#acc_yp_min = np.array(yp)-np.array(ypstd)
#acc_yp_max = np.array(yp)+np.array(ypstd)
#plt.errorbar(xp,y=yp, yerr=ypstd, color='#6B687B',marker= 's',\
#             ms=3,linewidth=1, linestyle='-',label='FBM',capsize=2,capthick=1, ecolor='#C6C2DB')
# 给定数据
x_data = [0.09, 0.2, 0.375, 0.455, 0.56, 1.0]
y_data = []

colors = [
    '#150266',  
    '#4D028F',  
    '#B0007F',
    '#EA2462',
    '#FF7449',
    '#FFB946',
]


# 寻找x值对应的y值
#for x_point in x_data:
#    index = x.index(x_point)
#    y_data.append(acc_fbm_list[index])

#plt.scatter(x_data, y_data, color=colors, marker='^', s=40, zorder=20)  # 绘制选择的点
#plt.text(x_data[0] - 0.01, y_data[0] + 0.4, "$S_1$", ha='right', va='bottom', color=colors[0], weight='bold', fontsize=10)
#plt.text(x_data[1] + 0.03, y_data[1] + 0.3, "$P_1$", ha='right', va='bottom', color=colors[1], weight='bold', fontsize=10)
#plt.text(x_data[2] + 0.02, y_data[2] + 0.3, "$V$", ha='right', va='bottom', color=colors[2], weight='bold', fontsize=10)
#plt.text(x_data[3] + 0.03, y_data[3] + 0.3, "$P_2$", ha='right', va='bottom', color=colors[3], weight='bold', fontsize=10)
#plt.text(x_data[4] + 0.08, y_data[4] + 0.3, "$S_2$", ha='right', va='bottom', color=colors[4], weight='bold', fontsize=10)
#plt.text(x_data[5] + 0.03, y_data[5] + 0.4, "$B_1$", ha='right', va='bottom', color='#FFD039', weight='bold', fontsize=10)
#plt.ylim([77.5,87.5])

plt.figure(figsize=(4,3),dpi=300)
plt.axhline(y=acc_mlp , color='#C3A3DF', linestyle='--', label='MLP',linewidth=1.8)
plt.errorbar(fermi_surface_list, acc_fbm_list, yerr=std_fbm_list,color='#6B687B',marker= 's',\
             ms=3,linewidth=1, linestyle='-',label='LCLM',capsize=2,capthick=1, ecolor='#C6C2DB')
plt.legend(fontsize=10)



plt.ylabel("Accuracy (%)",fontsize=12)
plt.xlabel(r"$d_F$",fontsize=12)
plt.yticks([65,70,75,80],fontsize=10)
plt.xticks([0.2,0.4,0.6,0.8,1.],fontsize=10)
plt.tight_layout()
plt.savefig('./result/peak_FM.pdf',bbox_inches='tight')
plt.show()