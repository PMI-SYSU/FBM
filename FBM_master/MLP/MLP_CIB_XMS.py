#########################
#       设置超参数       #
#########################
learning_rate = 0.01
final_learning_rate =0.001
hidden1_num_epochs = 150
final_num_epochs = 100

#神经网络模型参数
in_features = 784
hidden1_features=500
out_features = 10

# 每类多少张
train_subset= 100
test_subset= 400

# batch00
batch_size0 = 50

# 参数设定
f = 1
b = 1
# 生成从0到1.2的列表，间隔为0.005
start = 0
end = 1.2
interval = 0.005

# fermi_surface_list = [round(start + i * interval, 3) for i in range(int((end - start) / interval) + 1)]
lam = 0.01
# 运行圈数
runs = [0,1,2,3,4]

#########################
#       Libraries       #
#########################
import torch.nn as nn
import torch.optim as optim
import neural_network
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader 
import os
from tqdm import tqdm

#%%
# 实例化
ReLU = nn.ReLU()

#########################
#        logging        #
#########################
import logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
# 设置日志级别
logging.getLogger().setLevel(logging.INFO)

#######################
#       便捷函数       #
#######################
def batch_pair(batch_data, batch_labels):
    '''
        费米子对赋予标签为（-），可以通过反对易关系记忆
        玻色子对赋予标签为（+），可以通过正对易关系记忆
    '''
    pairs = []
    labels = []
    batch_size = batch_data.size()[0]
    for i in range(batch_size):
        for j in range(batch_size):
            if j > i:
                point1 = batch_data[i].unsqueeze(0)
                point2 = batch_data[j].unsqueeze(0)
                pair_data = torch.cat((point1, point2), dim=0)
                pairs.append(pair_data)
                if batch_labels[i] == batch_labels[j]:
                    labels.append(0)
                else:
                    labels.append(1)
    dataset = TensorDataset(torch.stack(pairs), torch.tensor(labels))
    return dataset

def split_dataset(dataset, label):
    split_data = []

    # 分开数据
    for data_point, data_label in dataset:
        if data_label == label:
            split_data.append((data_point, data_label))

    return split_data
#%%

#test_CImB_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
from torchvision import datasets, transforms
from Utils_Mnist.Dataloader_ClassImbalance import CustomDataset

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='D:\Japan\work\G_map\Code\data\MNIST', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='D:\Japan\work\G_map\Code\data\MNIST', train=False, download=False, transform=transform)


# Assuming train_dataset is your original MNIST dataset
class_ratio_test = {0: 0.5, 1: 0.5, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
batch_size = 50
Num_data= [200]
num_dataset=200
ratio_list = [0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62]#[0.35,0.4,0.45,0.5,0.55,0.6,0.65]#
fermi_surface_list = [0.4]

#FashionMNIST数据
#train_dataset_Fashion = datasets.FashionMNIST(root='.\Fashiondata', train=True, download=True, transform=transform)
#test_dataset_Fashion = datasets.FashionMNIST(root='.\Fashiondata', train=False, download=True, transform=transform)

#%%

#########################
#       训练神经网络     #
#########################
# 神经网络模型设定
device = torch.device('cpu')
model = neural_network.R_calibrate3_Net(device, torch.nn.Tanh(), in_features=784, hidden1_features=hidden1_features,
                                 out_features=out_features).to(device)

# model = neural_network.MLLP_Net(device, torch.nn.Tanh(), in_features=784, hidden1_features=200, hidden2_features=200,
#                                 out_features=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

download_path = None
upload_path = None
# upload_path = "Weight/FGSM3_calibrate.pth"

# 加载保存的权重
if download_path != None:
    model.load_state_dict(torch.load(download_path))


# 检查需要训练的权重
for name, param in model.named_parameters():
    if param.requires_grad:
        # print(name, param.Figure)
        print(name)
#########################
#      模型训练及测试     #
#########################

for ratio0 in (ratio_list):
    for run in runs:
        #生成数据
        device=torch.device('cpu')
        ratio1 = 1-ratio0
        class_ratio = {0: ratio0, 1: ratio1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        num_dataset_test = 800
        
        batch_size_test=num_dataset_test
                
        train_dataset_CImB = CustomDataset(train_dataset, class_ratio, batch_size, num_dataset)
        test_dataset_CImB = CustomDataset(test_dataset , class_ratio_test, batch_size_test, num_dataset_test)
        # DataLoader for custom dataset
        train_CImB_loader = DataLoader(train_dataset_CImB, batch_size=batch_size, shuffle=False)
        test_CImB_loader = DataLoader(test_dataset_CImB, batch_size=batch_size_test, shuffle=False)
        
        final_loss_tra_row = []
        final_acc_tra_row = []
        final_loss_test_row = []
        final_acc_test_row = []
        upload_weight = f"data/weight/CIB_MLP_num={num_dataset}_{run}.pth"
        
        for epoch in tqdm(range(final_num_epochs)):
            training_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(train_CImB_loader, 0):
                # 循环要进行dataset/batch次,如：1000/20
        
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
        
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        
                loss.backward()
                optimizer.step()
        
                training_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # print('Epoch %d, training loss: %.3f' % (epoch + 1, training_loss / len(pair_00_set)))
            final_loss_tra_row.append(training_loss / len(train_CImB_loader))
            final_acc_tra_row.append(100 * correct / total)
        
            with torch.no_grad():
                test_loss = 0.0
                correct = 0
                total = 0
                for data in test_CImB_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
        
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            # print('Epoch %d, test loss: %.3f' % (epoch + 1, test_loss / len(pair_test_set)))
            final_loss_test_row.append(test_loss / len(test_CImB_loader))
            final_acc_test_row.append(100 * correct / total)
        
        class_counts_array = [v for k, v in train_dataset_CImB.class_counts.items()]
        print("final_stage", "have completed")
        torch.save(model.state_dict(), upload_weight)
        np.savetxt(f"data/count/CIB_MLP_tra_lam={lam}_num={num_dataset}_{run}.csv", class_counts_array)
        np.savetxt(f"data/count/CIB_MLP_tra_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv", class_counts_array)
        np.savetxt(f"data/acc/CIB_MLP_tra_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv", final_acc_tra_row)
        np.savetxt(f"data/acc/CIB_MLP_test_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv", final_acc_test_row)
        np.savetxt(f"data/loss/CIB_MLP_tra_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv", final_loss_tra_row)
        np.savetxt(f"data/loss/CIB_MLP_test_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv", final_loss_test_row)
        #np.savetxt(f"data/distance/CIB_MLP_loss_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv", loss_tra_row)
        #np.savetxt(f"data/distance/CIB_MLP_distance_bose_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv", distance_bose_row_mini)
        #np.savetxt(f"data/distance/CIB_MLP_distance_fermi_dF={fermi_surface}_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run}.csv", distance_fermi_row_mini)
        logging.info(f" data MLP_lam={lam}_num={num_dataset}_batchsize={batch_size}_0={round(class_ratio[0],2)}_1={round(class_ratio[1],2)}_{run} has saved")
    