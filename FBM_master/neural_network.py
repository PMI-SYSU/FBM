import torch
import torch.nn as nn

# Define neural network：784--100==10
class R_FBMP3_Net(nn.Module):
    def __init__(self, device, activator, in_features=784, hidden1_features=100,
                 out_features=2):
        super(R_FBMP3_Net, self).__init__()
        self.in_features = in_features
        self.hidden1_features = hidden1_features
        self.out_features = out_features
        self.activator = activator
        # self.fc_W = nn.Linear(in_features, out_features, bias=False)
        self.fc_W1 = torch.nn.Parameter(
            torch.randn(in_features, hidden1_features) / torch.sqrt(torch.tensor(float(in_features)))).to(device)
        self.fc_W2 = torch.nn.Parameter(
            torch.randn(hidden1_features, out_features) / torch.sqrt(torch.tensor(float(hidden1_features)))).to(device)
        self.fc_W2.requires_grad = False

    def forward(self, x):
        x = x.view(-1, 2, self.in_features)
        ### view牛逼，直接整成[batch_size,748],管球你
        # print(x.shape)
        x = torch.matmul(x, self.fc_W1)
        x = 1/2 * (1+self.activator(x))
        '''
        C = torch.randn(batch, 2, 784)
        D = torch.randn(784,100)
        output == (batch, 2, 100)
        '''
        return x

# Define neural network：784==100--2
class R_MLP3_Net(nn.Module):
    def __init__(self, device, activator, in_features=784, hidden1_features=100,
                 out_features=2):
        super(R_MLP3_Net, self).__init__()
        self.in_features = in_features
        self.hidden1_features = hidden1_features
        self.out_features = out_features
        self.activator = activator
        # self.fc_W = nn.Linear(in_features, out_features, bias=False)
        self.fc_W1 = torch.nn.Parameter(
            torch.randn(in_features, hidden1_features) / torch.sqrt(torch.tensor(float(in_features)))).to(device)
        self.fc_W2 = torch.nn.Parameter(
            torch.randn(hidden1_features, out_features) / torch.sqrt(torch.tensor(float(hidden1_features)))).to(device)
        self.fc_W1.requires_grad = False

    def forward(self, x):
        x = x.view(-1, self.in_features)
        ### view牛逼，直接整成[batch_size,748],管球你
        # print(x.shape)
        x = torch.matmul(x, self.fc_W1)
        x = 1/2 * (1+self.activator(x))
        x = torch.matmul(x, self.fc_W2)
        # x = 1/2 * (1+self.activator(x))
        return x

class R_calibrate3_Net(nn.Module):
    def __init__(self, device, activator, in_features=784, hidden1_features=100,
                 out_features=2):
        super(R_calibrate3_Net, self).__init__()
        self.in_features = in_features
        self.hidden1_features = hidden1_features
        self.out_features = out_features
        self.activator = activator
        # self.fc_W = nn.Linear(in_features, out_features, bias=False)
        self.fc_W1 = torch.nn.Parameter(
            torch.randn(in_features, hidden1_features) / torch.sqrt(torch.tensor(float(in_features)))).to(device)
        self.fc_W2 = torch.nn.Parameter(
            torch.randn(hidden1_features, out_features) / torch.sqrt(torch.tensor(float(hidden1_features)))).to(device)
    def forward(self, x):
        x = x.view(-1, self.in_features)
        ### view牛逼，直接整成[batch_size,748],管球你
        # print(x.shape)
        x = torch.matmul(x, self.fc_W1)
        x = 1 / 2 * (1 + self.activator(x))
        x = torch.matmul(x, self.fc_W2)
        # x = 1 / 2 * (1 + self.activator(x))
        return x
