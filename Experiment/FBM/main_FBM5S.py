#########################
#       设置超参数       #
#########################
learning_rate = 0.01
final_learning_rate =0.001
hidden1_num_epochs = 100
final_num_epochs = 100

#神经网络模型参数
in_features = 784
hidden1_features=1000
out_features = 10

# 每类多少张
train_subset= 100
test_subset= 400

# batch00
batch_size0 = 50
batch_size_test=10000

# 参数设定
f = 1
b = 1
# 生成从0到1.2的列表，间隔为0.005
start = 0
end = 1.2
interval = 0.005

# fermi_surface_list = [round(start + i * interval, 3) for i in range(int((end - start) / interval) + 1)]
fermi_surface_list = [0.455]
lam = 0.01
# 运行圈数
runs = 5

#########################
#       Libraries       #
#########################
import torch.nn as nn
import torch.optim as optim
import neural_network
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
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

############################
#       生成配对数据         #
############################

### 十分类
from FBM.Utils_Mnist import mnist_10_loader_torch

train_set_0, train_set_1, train_set_2, train_set_3, train_set_4, train_set_5, train_set_6, train_set_7, train_set_8, train_set_9, \
test_set_0, test_set_1, test_set_2, test_set_3, test_set_4, test_set_5, test_set_6, test_set_7, test_set_8, test_set_9 \
    = mnist_10_loader_torch.load_data_wrapper(train_subset=train_subset, test_subset=test_subset, root='../Mnist')

tra_loader = ConcatDataset([
        train_set_0,
        train_set_1,
        train_set_2,
        train_set_3,
        train_set_4,
        train_set_5,
        train_set_6,
        train_set_7,
        train_set_8,
        train_set_9
])
test_loader = ConcatDataset([
        test_set_0,
        test_set_1,
        test_set_2,
        test_set_3,
        test_set_4,
        test_set_5,
        test_set_6,
        test_set_7,
        test_set_8,
        test_set_9
])

train_loader = DataLoader(tra_loader, batch_size= batch_size0, shuffle=True)
test_loader = DataLoader(test_loader, batch_size= batch_size_test, shuffle=True)

for run in range(runs):
    logging.info(f" run{run} begin")
    for fermi_surface in fermi_surface_list:
        logging.info(f" dF={fermi_surface} is running")

        # 每次传递的权重
        Weight = f"Weight_dF={fermi_surface}_lam={lam}_{run}.pth"
        # 每次储存的网络参数
        upload_weight = f"Weight/dF={fermi_surface}_lam={lam}_{run}.pth"

        ###############################
        #       训练神经网络中间层1      #
        ###############################
        # 神经网络模型设定
        device = torch.device('cpu')
        model = neural_network.R_FBMP3_Net(device, torch.nn.Tanh(), in_features=in_features, hidden1_features=hidden1_features,
                                           out_features=out_features).to(device)
        criterion = nn.PairwiseDistance(p=2,keepdim=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        download_path = None
        upload_path = Weight
        #########################
        #      模型训练及测试     #
        #########################
        loss_tra_row = []
        distance_fermi_row_mini = []
        distance_bose_row_mini = []

        for epoch in range(hidden1_num_epochs):
            if epoch % 20 == 0:
                logging.info(f" hidden layer epoch{epoch} has finished")
            # 训练模型
            training_loss = 0.0
            distance_fermi_mini = [[] for _ in range(len(train_loader))]
            distance_bose_mini = [[] for _ in range(len(train_loader))]

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                #################################
                #       配对费米玻色数据集         #
                #################################
                dataset_batch = batch_pair(inputs, labels)
                train_bose_loader = split_dataset(dataset_batch, 0)
                train_fermi_loader = split_dataset(dataset_batch, 1)

                #################################
                #       mini_batch训练           #
                #################################
                pair_inputs = torch.stack([item[0] for item in dataset_batch], dim=0)
                pair_labels = torch.stack([item[1] for item in dataset_batch])
                optimizer.zero_grad()
                outputs = model(pair_inputs)

                # 使用索引操作分割张量
                left = outputs[:, 0:1, :].view(len(pair_inputs), -1)
                right = outputs[:, 1:2, :].view(len(pair_inputs), -1)
                distance = criterion(left, right) ** 2 / hidden1_features
                loss = torch.sum(b * (1 - pair_labels) * distance + pair_labels * ReLU(fermi_surface - distance)) / len(pair_inputs) + 0.5 * lam / hidden1_features * torch.norm(model.fc_W1, p=2) ** 2
                training_loss += loss.item()
                distance_bose_mini[i] = torch.mean((1-pair_labels) * distance).item()
                distance_fermi_mini[i] = torch.mean(pair_labels * distance).item()
                loss.backward()
                optimizer.step()

            loss_tra_row.append(training_loss/len(train_loader))
            distance_bose_row_mini.append(distance_bose_mini)
            distance_fermi_row_mini.append(distance_fermi_mini)

        # 保存神经网络连接权重
        if upload_path != None:
            torch.save(model.state_dict(), upload_path)

        ##############################
        #       训练神经网络读出头      #
        ##############################
        # 神经网络模型设定
        device = torch.device('cpu')
        model = neural_network.R_MLP3_Net(device, torch.nn.Tanh(), in_features=in_features, hidden1_features=hidden1_features,
                                          out_features=out_features).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=final_learning_rate)
        # 加载保存的权重
        download_path = Weight
        upload_path = None
        if download_path != None:
            model.load_state_dict(torch.load(download_path))

        #########################
        #      模型训练及测试     #
        #########################
        final_loss_tra_row = []
        final_acc_tra_row = []
        final_loss_test_row = []
        final_acc_test_row = []

        for epoch in range(final_num_epochs):
            if epoch % 20 == 0:
                logging.info(f" read-out layer epoch{epoch} has finished")
            training_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(train_loader, 0):
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
            final_loss_tra_row.append(training_loss / len(train_loader))
            final_acc_tra_row.append(100 * correct / total)

            with torch.no_grad():
                test_loss = 0.0
                correct = 0
                total = 0
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            final_loss_test_row.append(test_loss / len(test_loader))
            final_acc_test_row.append(100 * correct / total)

        # 删除下载过的神经网络连接权重
        import os
        os.remove(download_path)

        torch.save(model.state_dict(), upload_weight)
        np.savetxt(f"data/acc/tra_dF={fermi_surface}_lam={lam}_{run}.csv", final_acc_tra_row)
        np.savetxt(f"data/acc/test_dF={fermi_surface}_lam={lam}_{run}.csv", final_acc_test_row)
        np.savetxt(f"data/loss/tra_dF={fermi_surface}_lam={lam}_{run}.csv", final_loss_tra_row)
        np.savetxt(f"data/loss/test_dF={fermi_surface}_lam={lam}_{run}.csv", final_loss_test_row)
        np.savetxt(f"data/distance/loss_dF={fermi_surface}_lam={lam}_{run}.csv", loss_tra_row)
        np.savetxt(f"data/distance/distance_bose_dF={fermi_surface}_lam={lam}_{run}.csv", distance_bose_row_mini)
        np.savetxt(f"data/distance/distance_fermi_dF={fermi_surface}_lam={lam}_{run}.csv", distance_fermi_row_mini)
        logging.info(f" data dF={fermi_surface}_lam={lam}_{run} has saved")
