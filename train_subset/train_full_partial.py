import random
from functools import partial
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from model_builder import Network, ModelSpec
from main_logger import get_logger
import json
import torch.nn as nn
import time
import datetime
import os
import argparse
from utils import generate_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--subset_size', '-s', type=int, help='Subset size (if None: considers entire set)', default=None)
parser.add_argument('--dataset', type=str, help='Dataset name')
parser.add_argument('--root', help='Path to dataset', default='data/')
parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3)
parser.add_argument('--weight_decay', type=float, help='Weight decay', default=1e-5)
parser.add_argument('--out_folder', type=str, help='Path to save checkpoints', default='weights/')
parser.add_argument('--test_every', type=int, help='Eval loop frequency', default=1)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=150)
parser.add_argument('--seed', type=int, help='Seed for RNG', default=123)
parser.add_argument('--batch_size', type=int, help='Training batch size', default=128)
parser.add_argument('--test_batch_size', type=int, help='Testing batch_size', default=128)
parser.add_argument('--json_folder', help='Path to json files')
parser.add_argument('--subset_folder', help='Path to indices')
parser.add_argument('--out_channels', type=int, help='Channels in CNN', default=128)
parser.add_argument('--num_cells', type=int, help='Number of stacked cells externally', default=3)
parser.add_argument('--num_internal_cells', type=int, help='Number of stacked cells internally', default=3)
args = parser.parse_args()

def seed_worker(seed, worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_checkpoint(net, name, arch):
    file_logger.info(f'Saving Checkpoint')

    if not os.path.isdir(f'{args.out_folder}/{arch}'):
        os.mkdir(f'{args.out_folder}/{arch}')
        
    torch.save(net.state_dict(), f'{args.out_folder}/{arch}/ckpt_{name}.pt')

@torch.no_grad()
def eval_training(epoch):
    loss_function = nn.CrossEntropyLoss()
    start = time.time()
    model.eval()

    test_loss = 0.0 
    correct = 0.0

    for (images, labels) in test_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    file_logger.info('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    return correct.float() / len(test_loader.dataset)

def train(arch_name, set):
    best_test_acc = 0

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay)
    step = 1
    all_acc = []
    for epoch in range(1, args.epochs + 1):
        model.train()

        # cal one epoch time
        start = time.time()

        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        end = time.time()
        file_logger.info(f"Epoch [{epoch}/{args.epochs}], "
              f"time: {end - start} sec")
        if epoch % args.test_every == 0:
            test_acc = eval_training(epoch)
            all_acc.append(test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_checkpoint(model, f'{args.dataset}_{args.subset_size}', arch_name)

    return best_test_acc, all_acc



if __name__ == '__main__':
    log_file = f'logfiles/{args.dataset}/{args.subset_size}.log'
    if not os.path.exists(f'logfiles/{args.dataset}/'):
        os.makedirs(f'logfiles/{args.dataset}', exist_ok=True)
    file_logger = get_logger(f'subset_{args.subset_size}', log_file)

    

    if args.dataset == 'fmnist':
        test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        train_set, _ = generate_dataset(args.dataset, args.root)
        test_set = torchvision.datasets.FashionMNIST(root=args.root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar10':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set, _ = generate_dataset(args.dataset, args.root)
        test_set = torchvision.datasets.CIFAR10(root=args.root, train=False, download=True, transform=test_transform)
    
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set, _ = generate_dataset(args.dataset, args.root)
        test_set = torchvision.datasets.CIFAR100(root=args.root, train=False, download=True, transform=test_transform)

    seed_worker(args.seed, 0)
    worker_fn = partial(seed_worker,0)
    all_test_acc = []
    all_seq_test_acc = []
    num_archs = len(os.listdir(args.json_folder))
    for i in range(num_archs):
        arch_number = i + 1
        arch = f'arch{arch_number+1}'
        with open(f'{args.json_folder}/{arch}.json') as f:
            data = json.load(f)
        
        adj, ops = data['module_adjacency'], data['module_operations']
        
        if args.subset_size != 0:
            with open(f'{args.subset_folder}/{arch}_{args.subset_size}_indices.pkl', 'rb') as f:
                indices = pickle.load(f)
            subtrainset = torch.utils.data.Subset(train_set, indices)
        else:
            subtrainset = train_set
        train_loader = torch.utils.data.DataLoader(subtrainset, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_fn)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, worker_init_fn=worker_fn)
        
        spec = ModelSpec(adj, ops)

        if args.dataset == 'cifar10':
            num_cls, in_ch = 10, 3
        elif args.dataset == 'cifar100':
            num_cls, in_ch = 100, 3
        elif args.dataset == 'fmnist':
            num_cls, in_ch = 10, 1

        model = Network(spec,num_cls, in_ch, args.out_channels, args.num_cells, args.num_internal_cells).cuda()
        best_test_acc, all_acc = train(arch, args.subset_size)
        file_logger.info(f"Best Test Acc: {best_test_acc:.3f}")
        all_test_acc.append(best_test_acc)
        all_seq_test_acc.append(all_acc)
        
    with open(f'{args.out_folder}/{args.subset_size}_best.pkl', 'wb') as f:
        pickle.dump(all_test_acc, f)

    with open(f'{args.out_folder}/{args.dataset}/{args.subset_size}_all.pkl', 'wb') as f:
        pickle.dump(all_seq_test_acc, f)
    