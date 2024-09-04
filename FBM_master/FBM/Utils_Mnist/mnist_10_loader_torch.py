from torch.utils.data import Subset
from torchvision import datasets, transforms
import torch

def load_data_wrapper(train_subset, test_subset, root='D:\Japan\work\G_map\Code\data\MNIST'):
    '''
    
    '''
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_set_0 = Subset(train_set, torch.where(train_set.targets == 0)[0])
    train_set_1 = Subset(train_set, torch.where(train_set.targets == 1)[0])
    train_set_2 = Subset(train_set, torch.where(train_set.targets == 2)[0])
    train_set_3 = Subset(train_set, torch.where(train_set.targets == 3)[0])
    train_set_4 = Subset(train_set, torch.where(train_set.targets == 4)[0])
    train_set_5 = Subset(train_set, torch.where(train_set.targets == 5)[0])
    train_set_6 = Subset(train_set, torch.where(train_set.targets == 6)[0])
    train_set_7 = Subset(train_set, torch.where(train_set.targets == 7)[0])
    train_set_8 = Subset(train_set, torch.where(train_set.targets == 8)[0])
    train_set_9 = Subset(train_set, torch.where(train_set.targets == 9)[0])

    test_set_0 = Subset(test_set, torch.where(test_set.targets == 0)[0])
    test_set_1 = Subset(test_set, torch.where(test_set.targets == 1)[0])
    test_set_2 = Subset(test_set, torch.where(test_set.targets == 2)[0])
    test_set_3 = Subset(test_set, torch.where(test_set.targets == 3)[0])
    test_set_4 = Subset(test_set, torch.where(test_set.targets == 4)[0])
    test_set_5 = Subset(test_set, torch.where(test_set.targets == 5)[0])
    test_set_6 = Subset(test_set, torch.where(test_set.targets == 6)[0])
    test_set_7 = Subset(test_set, torch.where(test_set.targets == 7)[0])
    test_set_8 = Subset(test_set, torch.where(test_set.targets == 8)[0])
    test_set_9 = Subset(test_set, torch.where(test_set.targets == 9)[0])

    train_set_0 = Subset(train_set_0, range(train_subset))
    train_set_1 = Subset(train_set_1, range(train_subset))
    train_set_2 = Subset(train_set_2, range(train_subset))
    train_set_3 = Subset(train_set_3, range(train_subset))
    train_set_4 = Subset(train_set_4, range(train_subset))
    train_set_5 = Subset(train_set_5, range(train_subset))
    train_set_6 = Subset(train_set_6, range(train_subset))
    train_set_7 = Subset(train_set_7, range(train_subset))
    train_set_8 = Subset(train_set_8, range(train_subset))
    train_set_9 = Subset(train_set_9, range(train_subset))

    test_set_0 = Subset(test_set_0, range(test_subset))
    test_set_1 = Subset(test_set_1, range(test_subset))
    test_set_2 = Subset(test_set_2, range(test_subset))
    test_set_3 = Subset(test_set_3, range(test_subset))
    test_set_4 = Subset(test_set_4, range(test_subset))
    test_set_5 = Subset(test_set_5, range(test_subset))
    test_set_6 = Subset(test_set_6, range(test_subset))
    test_set_7 = Subset(test_set_7, range(test_subset))
    test_set_8 = Subset(test_set_8, range(test_subset))
    test_set_9 = Subset(test_set_9, range(test_subset))

    
    return (train_set_0,train_set_1,train_set_2,train_set_3,train_set_4,train_set_5,train_set_6,train_set_7,train_set_8,train_set_9,
            test_set_0, test_set_1, test_set_2, test_set_3, test_set_4, test_set_5, test_set_6, test_set_7,  test_set_8,  test_set_9
            )