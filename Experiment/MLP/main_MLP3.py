#########################
#       设置超参数       #
#########################
learning_rate = 0.001
batch_size = 100
final_num_epochs = 100

#神经网络模型参数
hidden1_features=1000
out_features=10
upload_weight = 'Weight/March_31_batch100P5000.pth'
# torch.save(model.state_dict(), upload_weight)
# import numpy as np
# np.savetxt("Figure/acc/tra_ep100_n1000_h1000_batch50_Nov16th_3",final_acc_tra_row)
# np.savetxt("Figure/acc/test_ep100_n1000_h1000_batch50_Nov16th_3",final_acc_test_row)

##########################
#       加载数据集10        #
##########################
# out_train_subset= 200
# out_test_subset= 1000
# from Mnist.MNIST import mnist_loader_torch
# train_set,test_set = mnist_loader_torch. \
#     load_data_wrapper(train_subset=out_train_subset, test_subset=out_test_subset, batch_size=batch_size)


### 四分类
from FBM.Utils_Mnist import mnist_10_loader_torch
from torch.utils.data import DataLoader, ConcatDataset

# 每类多少张
train_subset= 500
test_subset= 400

train_set_0, train_set_1, train_set_2, train_set_3, train_set_4, train_set_5, train_set_6, train_set_7, train_set_8, train_set_9, \
test_set_0, test_set_1, test_set_2, test_set_3, test_set_4, test_set_5, test_set_6, test_set_7, test_set_8, test_set_9 \
    = mnist_10_loader_torch.load_data_wrapper(train_subset=train_subset, test_subset=test_subset)

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

train_set = DataLoader(tra_loader, batch_size= batch_size, shuffle=True)
test_set = DataLoader(test_loader, batch_size= 10000, shuffle=True)

#########################
#       神经网络部分      #
#########################
import torch
import torch.nn as nn
import torch.optim as optim
import neural_network
from tqdm import tqdm
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
final_loss_tra_row = []
final_acc_tra_row = []
final_loss_test_row = []
final_acc_test_row = []

for epoch in tqdm(range(final_num_epochs)):
    training_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_set, 0):
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
    final_loss_tra_row.append(training_loss / len(train_set))
    final_acc_tra_row.append(100 * correct / total)

    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        for data in test_set:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print('Epoch %d, test loss: %.3f' % (epoch + 1, test_loss / len(pair_test_set)))
    final_loss_test_row.append(test_loss / len(test_set))
    final_acc_test_row.append(100 * correct / total)

print("final_stage", "have completed")
# 保存神经网络连接权重
# if upload_path != None:
#     torch.save(model.state_dict(), upload_path)



#########################
#       画图环节         #
#########################

import matplotlib.pyplot as plt

xx = range(final_num_epochs)
fig = plt.figure()

ax2 = fig.add_subplot(111)
ax2.plot(xx, final_acc_tra_row, 'y', label='train accuracy')
ax2.plot(xx, final_acc_test_row, 'orange', label='test accuracy')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
# ax2.set_ylim(0,102)

ax1 = ax2.twinx()
ax1.plot(xx, final_loss_tra_row, 'b', label='train loss')
ax1.plot(xx, final_loss_test_row, 'c', label='test loss')
ax1.set_ylabel('loss')
# ax1.set_ylim(0.05,2)

my_title = "Gauge learning, epoch="+str(final_num_epochs)+", batch="+str(batch_size)+", lr="+str(learning_rate)
ax2.set_title(my_title)

legend = fig.legend(loc='center right',bbox_to_anchor=(0.90, 0.5))
frame = legend.get_frame()
frame.set_alpha(0.7)

plt.show()