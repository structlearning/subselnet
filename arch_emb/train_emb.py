import os 
import torch 
import torch.nn as nn
import numpy as np
import torch.optim as optim
from gnn_emb import * 
import argparse
import json
from helper import *
parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
parser.add_argument('--nasbench_path', help='Path to the nasbench record file')
parser.add_argument('--data_folder', help='Path to the data.json folder')
parser.add_argument('--checkpoint_folder', help='Path to the checkpoint save folder')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--seed', type=int, default=0, help='Seed for RNG')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
parser.add_argument('--input_dim', type=int, default=5, help='GNN input dim')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN hidden dim')
parser.add_argument('--output_dim', type=int, default=16, help='GNN output dim')
parser.add_argument('--num_rec', type=int, default=5, help='GNN rec')
parser.add_argument('--num_layers', type=int, default=2, help='GNN layers')
args = parser.parse_args()

class recon_loss():
    def __init__(self):
        super(recon_loss, self).__init__()
    
    def __call__(self, inputs, targets, mu, sigma):
        new_ops, new_adj = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * sigma - mu.pow(2) - sigma.exp().pow(2), 2))
        return loss

def save_model(epoch, loss, net, opt):
    ckpt = {'epoch': epoch,
            'loss': loss,
            'model_state': net.state_dict(),
            'optimizer_state': opt.state_dict()}

    torch.save(ckpt, f'{args.checkpoint_folder}/Epoch{epoch}_Loss{round(loss, 3)}.pt')

    

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(f'{args.data_folder}/data.json', 'r') as f:
        archs = json.loads(f.read())

    train_idxs, val_idxs = range(int(len(archs) * 0.9)), range(int(len(archs) * 0.1))
    idxs = np.random.permutation(train_idxs)
    train_adj, train_ops = [], []
    for idx in idxs:
        train_adj.append(torch.Tensor(archs[str(idx)]['module_adjacency']))
        train_ops.append(torch.Tensor(archs[str(idx)]['module_operations']))
    train_indices = torch.Tensor(idxs)
    train_adj, train_ops = torch.stack(train_adj), torch.stack(train_ops)
    idxs = np.random.permutation(val_idxs)
    net = GNN(args.input_dim, args.hidden_dim, args.output_dim, args.num_rec, args.num_layers, args.dropout).cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    criterion = recon_loss()
    all_loss = []
    for epoch in range(args.epochs):
        num_batches = len(train_idxs) // args.batch_size
        if len(train_idxs) % args.batch_size > 0:
            num_batches += 1
        train_split_adj = torch.split(train_adj, args.batch_size, dim=0)
        train_split_ops = torch.split(train_ops, args.batch_size, dim=0)
        indices_split = torch.split(train_indices, args.batch_size, dim=0)
        losses = []
        for i, (adj, ops, idx) in enumerate(zip(train_split_adj, train_split_ops, indices_split)):
            optimizer.zero_grad()
            ops, adj = ops.cuda(), adj.cuda()
            adj += adj.triu(1).transpose(-1,-2)
            new_ops, new_adj, mu, sigma = net(ops, adj.to(torch.long))
            ops, adj = ops.triu(1), adj.triu(1)
            new_ops, new_adj = new_ops.triu(1), new_adj.triu(1)

            loss = criterion((new_ops, new_adj), (ops, adj), mu, sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
            if i % 100 == 0:
                print(f'Epoch {epoch} | Batch {i}/{num_batches} | Loss {loss.item()} ')

        all_loss.append(sum(losses)/len(losses))
        save_model(epoch, loss, net, optimizer)