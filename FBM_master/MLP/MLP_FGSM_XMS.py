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
epsilons = np.arange(0, 1.0, 0.01)

fermi_surface = 0.4
num_dataset = 1000
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


# 生成文件名列表 MLP_FM_num=1000_0
# download_path_list = generate_filenames(start_dF, end_dF, step_dF)
download_path_list = [
    "./data/weight/MLP_FM_num="+str(num_dataset)+"_0.pth",
    "./data/weight/MLP_FM_num="+str(num_dataset)+"_1.pth",
    "./data/weight/MLP_FM_num="+str(num_dataset)+"_2.pth",
    "./data/weight/MLP_FM_num="+str(num_dataset)+"_3.pth",
    "./data/weight/MLP_FM_num="+str(num_dataset)+"_4.pth",
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

from torchvision import datasets, transforms
from Utils_Mnist.Dataloader_ClassImbalance import CustomDataset
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

#train_dataset = datasets.MNIST(root='D:\Japan\work\G_map\Code\data\MNIST', train=True, download=False, transform=transform)
#test_dataset = datasets.MNIST(root='D:\Japan\work\G_map\Code\data\MNIST', train=False, download=False, transform=transform)


#FashionMNIST数据
train_dataset_Fashion = datasets.FashionMNIST(root='.\Fashiondata', train=True, download=True, transform=transform)
test_dataset_Fashion = datasets.FashionMNIST(root='.\Fashiondata', train=False, download=True, transform=transform)


#class_ratio = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}
#class_ratio_test = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}
#num_dataset_test = 1000            
#num_dataset = 1000            
#batch_size_test=num_dataset_test
                
#train_dataset_CImB = CustomDataset(train_dataset_Fashion, class_ratio, batch_size, num_dataset)
#test_dataset_CImB = CustomDataset(test_dataset_Fashion, class_ratio_test, batch_size_test, num_dataset_test)
        # DataLoader for custom dataset
#train_CImB_loader = DataLoader(train_dataset_CImB, batch_size=batch_size, shuffle=False)
#test_CImB_loader = DataLoader(test_dataset_CImB, batch_size=batch_size_test, shuffle=False)


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
model = neural_network.R_calibrate3_Net(device, torch.nn.Tanh(), in_features=784, hidden1_features=500,
                                        out_features=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


runs=[0,1,2,3,4]
for run in runs:
    # 加载保存的权重
    download_path = download_path_list[run]
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
        for data in test_dataset_Fashion:

            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Set requires_grad attribute of tensor. Important for Attack
            inputs.requires_grad = True

            # 第一次测试
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # If the initial prediction is wrong, dont bother attacking, just move on
            #if predicted.item() != labels.item():
            #    continue
            
            if i == 1000:
                break
            i = i+1

            #i+=1
            
            
            # 进行FGSM对抗

            loss = criterion(outputs, labels.unsqueeze(0))
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
            total += labels.unsqueeze(0).size(0)

            correct += (second_predicted == labels).sum().item()
        # print('Epoch %d, test loss: %.3f' % (epoch + 1, test_loss / len(pair_test_set)))
        final_loss_test_cols.append(test_loss / len(test_dataset_Fashion))
        final_acc_test_cols.append(100 * correct / total)
    
    np.savetxt(f"data/FGSM_acc/FM_FGSM_MLP_acc_num={num_dataset}_{run}.csv", final_acc_test_cols)

    print("# ", total)
    print("final_acc_FBM  =")
    print(list(total * 0.001 * np.array(final_acc_test_cols)))
    print("\n")

    #np.savetxt(f"data/FGSM/{download_path}", list(total * 0.001 * np.array(final_acc_test_cols)))


    # 保存神经网络连接权重
    if upload_path != None:
        torch.save(model.state_dict(), upload_path)