#########################
#       设置超参数       #
#########################
import numpy as np
import os
import sys

# 定义dF值的范围和步长
start_dF = 0.0
end_dF = 1.2
step_dF = 0.01

# 测试参数
learning_rate = 0.01
### 只能一个一个测试
batch_size = 1
epsilons = np.arange(0, 0.8, 0.01)

# 生成文件名列表
def generate_filenames(start_dF, end_dF, step_dF):
    filenames = []
    for dF in range(int(start_dF * 100), int(end_dF * 100) + 1, int(step_dF * 100)):
        dF_value = dF / 100
        for i in range(5):
            filename = f"Weight/dF={dF_value}_lam=0.01_{i}.pth"
            filenames.append(filename)
    return filenames


# 检查当前目录中是否所有文件都存在
def check_all_files_exist(filenames):
    current_directory = os.getcwd()  # 获取当前工作目录
    for filename in filenames:
        file_path = os.path.join(current_directory, filename)
        if not os.path.isfile(file_path):
            return False
    return True


# 生成文件名列表
# download_path_list = generate_filenames(start_dF, end_dF, step_dF)
download_path_list = [
    "Weight/dF=0.455_lam=0.01_0.pth",
    "Weight/dF=0.455_lam=0.01_1.pth",
    "Weight/dF=0.455_lam=0.01_2.pth",
    "Weight/dF=0.455_lam=0.01_3.pth",
    "Weight/dF=0.455_lam=0.01_4.pth",
]
# 检查所有文件是否都存在
if check_all_files_exist(download_path_list):
    print("OK")
else:
    # print(download_path_list)
    print("不是所有生成的文件都存在于当前目录下。")
    sys.exit(1)  # 非零退出码表示程序异常终止


upload_path = None

def show_image(array):
    import matplotlib.pyplot as plt
    img = 255*array.reshape(28,28)
    plt.imshow(img,cmap='Greys')
    plt.show()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    img = image.reshape(28, 28)
    noise = data_grad.reshape(28, 28)
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = noise.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = img + epsilon*sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image

##########################
#       加载数据集        #
##########################
out_train_subset= 0
out_test_subset= 1000

from Utils_Mnist import mnist_loader_torch

test_set = mnist_loader_torch. \
    load_data_wrapper(train_subset=out_train_subset, test_subset=out_test_subset, batch_size=batch_size,root="../Mnist")


#########################
#       神经网络部分      #
#########################
import torch
import torch.nn as nn
import torch.optim as optim
import neural_network

#########################
#       训练神经网络     #
#########################
# 神经网络模型设定
device = torch.device('cpu')
model = neural_network.R_calibrate3_Net(device, torch.nn.Tanh(), in_features=784, hidden1_features=1000,
                                        out_features=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

for download_path in download_path_list:
    # 加载保存的权重
    if download_path != None:
        model.load_state_dict(torch.load(download_path))
        print('## ',download_path)

    ####################
    #      模型测试     #
    ####################
    final_loss_test_cols = []
    final_acc_test_cols = []

    # for j in tqdm(epsilons):
    for j in (epsilons):
        test_loss = 0.0
        correct = 0
        total = 0
        i=0
        for data in test_set:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Set requires_grad attribute of tensor. Important for Attack
            inputs.requires_grad = True

            # 第一次测试
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # If the initial prediction is wrong, dont bother attacking, just move on
            if predicted.item() != labels.item():
                continue

            i+=1
            # 进行FGSM对抗
            loss = criterion(outputs, labels)
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = inputs.grad.data
            # Call FGSM Attack
            perturbed_data = fgsm_attack(inputs, j, data_grad)

            # 第二次测试
            second_outputs = model(perturbed_data)
            _, second_predicted = torch.max(second_outputs.data, 1)

            test_loss += loss.item()
            total += labels.size(0)

            correct += (second_predicted == labels).sum().item()
        # print('Epoch %d, test loss: %.3f' % (epoch + 1, test_loss / len(pair_test_set)))
        final_loss_test_cols.append(test_loss / len(test_set))
        final_acc_test_cols.append(100 * correct / total)
    print("# ", total)
    print("final_acc_FBM  =")
    print(list(total * 0.001 * np.array(final_acc_test_cols)))
    print("\n")

    # final_acc_test.append(list(total * 0.001 * np.array(final_acc_test_cols)))
    np.savetxt(f"data/FGSM/{download_path}", list(total * 0.001 * np.array(final_acc_test_cols)))


    # 保存神经网络连接权重
    if upload_path != None:
        torch.save(model.state_dict(), upload_path)