from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def load_data_wrapper(train_subset, test_subset, batch_size, root='D:/Software/pythonProject/Huang/Mnist'):

    if train_subset==0:
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_set = datasets.MNIST(root=root, train=False, download=True,
                                  transform=transform)

        # Subset the datasets
        test_set = Subset(test_set, range(test_subset))

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return test_loader

    else:
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)

        # Subset the datasets
        train_set = Subset(train_set, range(train_subset))
        test_set = Subset(test_set, range(test_subset))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return (train_loader, test_loader)

