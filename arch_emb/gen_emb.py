import os
import sys
import torch
import torch.nn as nn
import numpy as np
from gnn_emb import *
import argparse
import networkx as nx
import json
from helper import *

parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
parser.add_argument('--json_folder', help='Path to the folder containing architecture json files')
parser.add_argument('--model_path', help='Path to the checkpoint file')
parser.add_argument('--out_folder', help='Path to where to save embeddings')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
parser.add_argument('--input_dim', type=int, default=5, help='GNN input dim')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN hidden dim')
parser.add_argument('--output_dim', type=int, default=16, help='GNN output dim')
parser.add_argument('--num_rec', type=int, default=5, help='GNN rec')
parser.add_argument('--num_layers', type=int, default=2, help='GNN layers')
parser.add_argument('--train', action='store_true', help='Train or test mode')
args = parser.parse_args()


def bfs_seq(G, start_id):
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output

if __name__ == '__main__':
    adj_list = []
    ops_list = []
    INPUT = 'input'
    OUTPUT = 'output'
    CONV1X1 = 'conv1x1-bn-relu'
    CONV3X3 = 'conv3x3-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'
    setting = 'train' if args.train else 'test'
    num_files = len(os.listdir(f'{args.json_folder}/{setting}'))
    for i in range(1, num_files+1):
        with open(f'{args.json_folder}/{setting}/arch{i}.json') as f:
            data = json.load(f)
            adj_matrix = data['module_adjacency']
            if len(adj_matrix) <= 6:
                for r in range(len(adj_matrix)):
                    for i in range(7-len(adj_matrix)):
                        adj_list[r].append(0)
                for i in range(7-len(adj_matrix)):
                    adj_matrix.append([0, 0, 0, 0, 0, 0, 0])
            adj_list.append(torch.Tensor(adj_matrix))
            ops_list.append(torch.Tensor(data['module_operations']))

    adj_list = torch.split(torch.stack(adj_list), 1, dim=0)

    ops_list = torch.split(torch.stack(ops_list), 1, dim=0)            
    emb_list = []
    net = GNN(args.input_dim, args.hidden_dim, args.output_dim, args.num_rec, args.num_layers, args.dropout)
    
    #net.load_state_dict(torch.load(args.model_path)['model_state'])
    net.eval().cuda()

    with torch.no_grad():
        for i, (adj, ops) in enumerate(zip(adj_list, ops_list)):
            G = nx.from_numpy_matrix(adj[0].numpy().astype(int), create_using=nx.DiGraph)
            adj, ops = adj.cuda(), ops.cuda()
            _, _, emb, _ = net(ops.float(), adj.float())  
            emb = emb.squeeze(0).cpu()  
            seq = bfs_seq(G, 0)
            if len(seq) < 7:
                adder = list(range(len(seq), 7))
                seq.extend(adder)
            emb[[0, 1, 2, 3, 4, 5, 6]] = emb[seq]
            emb_list.append(emb)
            print(f"Arch {i} done...\n")

    os.mkdir(f'{args.out_folder}/emb')
            
    torch.save(emb_list, f'{args.out_folder}/emb/archemb_bfs.pt')
    print('file saved')