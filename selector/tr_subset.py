import torch
import os
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from datetime import datetime
from helper import *
from gallery_approx import *

parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
parser.add_argument('--data_file', type=str, help='Path to the data embeddings')
parser.add_argument('--targets_file', type=str, help='Path to the targets')
parser.add_argument('--archemb_file', type=str, help='Path to the architecture embeddings')
parser.add_argument('--approx_checkpoint', type=str, help='Path to the approx weights')
parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--json_folder', help='Path to the folder containing architecture json files')
parser.add_argument('--subset_size', type=int, help='Subset size (as integer)')
parser.add_argument('--num_iter', type=int, help='Number of iterations')
parser.add_argument('--lambda_1', type=float, help='Entropy weightage', default=1)
parser.add_argument('--lambda_2', type=float, help='Regularizer weightage', default=0.1)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--output_folder', help='Path to save the generated indices')
parser.add_argument('--ma_dropout', type=float, default=0.3)
args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    data = torch.load(args.data_file).cuda()
    targets = torch.load(args.targets_file).cuda()
    archembs = torch.load(args.archemb_file, map_location='cuda')
    if args.dataset == 'cifar10' or args.dataset == 'fmnist':
        num_cls = 10
    else:
        num_cls = 100
    target_distr = torch.zeros(targets.shape[0], num_cls)
    for i, j in enumerate(targets):
        target_distr[i,j] = 1.0
    target_indices = []
    for i in range(10):
        target_indices.append(torch.where(targets==i))
        
    num_archs = len(os.listdir(args.json_folder))

    print(len(archembs))
    print(num_archs)

    for i in range(num_archs):
        ma = AttendApproximator(archembs[i].shape[-1], data.shape[-1], args.ma_dropout, num_cls)
        ma.load_state_dict(torch.load(args.approx_checkpoint))
        ma = ma.cuda()
        criterion = nn.CrossEntropyLoss(reduction='mean')
        y_true = target_distr
        losses = []
        with torch.no_grad():
            ma.eval()
            y_pred = ma(archembs[i].repeat(targets.shape[0], 1, 1), data)
            for y1, y2 in zip(y_pred, targets):
                losses.append(criterion(y1.unsqueeze(0), y2.unsqueeze(0)))

        losses = torch.cat([loss.unsqueeze(0) for loss in losses], dim=0).cuda()

        pi = torch.tensor(np.random.rand(targets.shape[0]), dtype=torch.float32).cuda().requires_grad_(True)

        for j in range(args.num_iter):
            print(i)
            if pi.grad is not None:
                pi.grad.data.zero_()
            obj_fn = torch.dot(pi, losses) + args.lambda_1*torch.distributions.Categorical(pi.softmax(dim=0)).entropy() + args.lambda_2*torch.abs(torch.sum(pi) - args.subset_size)
            obj_fn.backward()
            pi.data = pi.data - args.learning_rate * pi.grad.data
            with torch.no_grad():
                pi[torch.where(pi < 0)] = 0.
            
        points = torch.topk(pi, args.subset_size)[1]
        points_list = [x.item() for x in points]

        with open(f'{args.output_folder}/arch{i+1}_{args.subset_size}.pkl', 'wb') as f:
            pickle.dump(points_list, f)



    
    
    