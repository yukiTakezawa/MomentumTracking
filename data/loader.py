import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
import random
from torch.utils.data.dataset import Subset
from more_itertools import chunked


    
def load_MNIST(batch=128, val_rate=0.2):
    
    train_val_dataset = datasets.FashionMNIST('./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(28, padding=4),
                           transforms.ToTensor()
                       ]))
    
    test_dataset = datasets.FashionMNIST('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    

    n_val = int(len(train_val_dataset)*val_rate)
    n_train = len(train_val_dataset) - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [n_train, n_val])

    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


def load_MNIST_hetero(node_label, batch=128, val_rate=0.2):
    """
    node_label : 
        the list of labes that each node has. (example. [[0,1],[1,2],[0,2]] (n_node=3, n_class=2))
    """
    
    train_val_dataset = datasets.FashionMNIST('./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(28, padding=4),
                           transforms.ToTensor()
                       ]))
    
    test_dataset = datasets.FashionMNIST('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

    n_val = int(len(train_val_dataset)*val_rate)
    n_train = len(train_val_dataset) - n_val
    n_node = len(node_label)
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [n_train, n_val])
        
    # the list of nodes that have class i.
    label_node = []
    for i in range(10):
        label_node.append([])
        for node in range(n_node):
            if i in node_label[node]:
                label_node[-1].append(node)
                
    node_indices = [[] for i in range(n_node)]

    for label in range(10):
        # list of index whose label is "label".
        indices = [train_dataset.indices[i] for i in range(len(train_dataset.indices)) if train_val_dataset.targets[train_dataset.indices[i]] == label]
    
        random.shuffle(indices)
        chunked_indices = list(chunked(indices, int(len(indices)/len(label_node[label]))))
    
        for i in range(len(label_node[label])):
            node_indices[label_node[label][i]] += chunked_indices[i]

    for i in range(n_node):
        random.shuffle(node_indices[i])
    n_data = min([len(node_indices[i]) for i in range(n_node)])
            
    train_subset_list = [Subset(train_val_dataset, node_indices[i][:n_data]) for i in range(n_node)]

    return {'train': train_subset_list, 'val': val_dataset, 'all_train': train_val_dataset, 'test': test_dataset}

    
    
def load_CIFAR10_hetero(node_label, batch=128, val_rate=0.2):
    """
    node_label : 
        the list of labes that each node has. (example. [[0,1],[1,2],[0,2]] (n_node=3, n_class=2))
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_val_dataset = datasets.CIFAR10('./data',
                       train=True,
                       download=True,
                       transform=transform_train)
    
    test_dataset = datasets.CIFAR10('./data',
                       train=False,
                       transform=transform_test)


    n_val = int(len(train_val_dataset)*val_rate)
    n_train = len(train_val_dataset) - n_val
    n_node = len(node_label)
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [n_train, n_val])
        
    # the list of nodes that have class i.
    label_node = []
    for i in range(10):
        label_node.append([])
        for node in range(n_node):
            if i in node_label[node]:
                label_node[-1].append(node)
                
    node_indices = [[] for i in range(n_node)]

    for label in range(10):
        # list of index whose label is "label".
        indices = [train_dataset.indices[i] for i in range(len(train_dataset.indices)) if train_val_dataset.targets[train_dataset.indices[i]] == label]
    
        random.shuffle(indices)
        chunked_indices = list(chunked(indices, int(len(indices)/len(label_node[label]))))
    
        for i in range(len(label_node[label])):
            node_indices[label_node[label][i]] += chunked_indices[i]

    for i in range(n_node):
        random.shuffle(node_indices[i])
    n_data = min([len(node_indices[i]) for i in range(n_node)])
            
    train_subset_list = [Subset(train_val_dataset, node_indices[i][:n_data]) for i in range(n_node)]


    return {'train': train_subset_list, 'val': val_dataset, 'all_train': train_val_dataset, 'test': test_dataset}


def load_CIFAR10(batch=128, val_rate=0.2):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_val_dataset = datasets.CIFAR10('./data',
                       train=True,
                       download=True,
                       transform=transform_train)
    
    test_dataset = datasets.CIFAR10('./data',
                       train=False,
                       transform=transform_test)

    n_val = int(len(train_val_dataset)*val_rate)
    n_train = len(train_val_dataset) - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [n_train, n_val])

    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


def load_SVHN_hetero(node_label, batch=128, val_rate=0.2):
    """
    node_label : 
        the list of labes that each node has. (example. [[0,1],[1,2],[0,2]] (n_node=3, n_class=2))
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_val_dataset = datasets.SVHN('./data',
                       split="train",
                       download=True,
                       transform=transform_train)
    
    test_dataset = datasets.SVHN('./data',
                       split="test",
                       download=True,
                       transform=transform_test)


    n_val = int(len(train_val_dataset)*val_rate)
    n_train = len(train_val_dataset) - n_val
    n_node = len(node_label)
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [n_train, n_val])
        
    # the list of nodes that have class i.
    label_node = []
    for i in range(10):
        label_node.append([])
        for node in range(n_node):
            if i in node_label[node]:
                label_node[-1].append(node)
                
    node_indices = [[] for i in range(n_node)]

    for label in range(10):
        # list of index whose label is "label".
        indices = [train_dataset.indices[i] for i in range(len(train_dataset.indices)) if train_val_dataset.labels[train_dataset.indices[i]] == label]
    
        random.shuffle(indices)
        chunked_indices = list(chunked(indices, int(len(indices)/len(label_node[label]))))
    
        for i in range(len(label_node[label])):
            node_indices[label_node[label][i]] += chunked_indices[i]

    for i in range(n_node):
        random.shuffle(node_indices[i])
    n_data = min([len(node_indices[i]) for i in range(n_node)])
            
    train_subset_list = [Subset(train_val_dataset, node_indices[i][:n_data]) for i in range(n_node)]


    return {'train': train_subset_list, 'val': val_dataset, 'all_train': train_val_dataset, 'test': test_dataset}


def datasets_to_loaders(datasets, batch_size=128):
    """
    datasets dict:
    """
    train_loader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=False, persistent_workers=True)

    all_train_loader = torch.utils.data.DataLoader(
        datasets["all_train"],
        batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=False)

    
    val_loader = torch.utils.data.DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=False)

    return {"train": train_loader, "val": val_loader, "all_train": all_train_loader, "test": test_loader}
