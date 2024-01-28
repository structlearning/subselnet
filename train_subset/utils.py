import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def generate_dataset(name, data_dir):
    

    if name == 'fmnist':
        train_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
        val_set = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=test_transform)

        return train_set, val_set

    elif name == 'cifar10':
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        val_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)

        return train_set, val_set


    elif name == 'cifar100':
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor(), 
                                transforms.Normalize(*stats,inplace=True)
                                ])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)
                                ])

        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        val_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=test_transform)

        return train_set, val_set