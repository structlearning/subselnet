from nasbench import api
import numpy as np
import json
import argparse
from collections import OrderedDict
import os
# import tensorflow as tf
from helper import *

parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
parser.add_argument('--nasbench_path', help='Path to the nasbench record file')
parser.add_argument('--out_folder', help='Path to the directory to save the json files')
parser.add_argument('--num_train', help='Number of training architectures', type=int)
parser.add_argument('--num_test', help='Number of testing architectures', type=int)
parser.add_argument('--seed', help='Seed for RNG', type=int, default=0)
args = parser.parse_args()


def get_ops(ops):
    mapping = {INPUT: 0, CONV1X1: 1, CONV3X3: 2, MAXPOOL3X3: 3, OUTPUT: 4}
    ops_ = np.zeros([7,5], dtype=np.int8)
    for i, op in enumerate(ops):
        ops_[i, mapping[op]] = 1

    return ops_

def get_arch_adj_ops(nb):
    for h in nb.hash_iterator():
        arch_struct, _ = nb.get_metrics_from_hash(h)
        adj, ops = arch_struct['module_adjacency'], arch_struct['module_operations']
        model_spec = api.ModelSpec(adj, ops)
        adj_list = adj.tolist()
        if len(adj_list) <= 6:
            for r in range(len(adj_list)):
                for i in range(7-len(adj_list)):
                    adj_list[r].append(0)
            for i in range(7-len(adj_list)):
                adj_list.append([0, 0, 0, 0, 0, 0, 0])

        yield {
                    'module_adjacency': adj_list,
                    'module_operations': get_ops(ops).tolist()
                }
        

if __name__ == '__main__':
    np.random.seed(args.seed)
    INPUT = 'input'
    OUTPUT = 'output'
    CONV1X1 = 'conv1x1-bn-relu'
    CONV3X3 = 'conv3x3-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'

    nb = api.NASBench(args.nasbench_path)
    all_archs = get_arch_adj_ops(nb)
    train_count = 1
    test_count = 1

    chosen_indices = np.random.choice(423624, args.num_train + args.num_test, replace=False)
    train_indices = chosen_indices[:args.num_train]
    test_indices = chosen_indices[args.num_train: args.num_train + args.num_test]

    print(chosen_indices)
    os.mkdir(f'{args.out_folder}/train/')
    os.mkdir(f'{args.out_folder}/test/')

    for i, arch in enumerate(all_archs):
        if i in train_indices:
            print('tr')
            with open(f'{args.out_folder}/train/arch{train_count}.json', 'w') as f:
                json.dump(arch, f)
            train_count += 1
        
        if i in test_indices:
            print('te')
            with open(f'{args.out_folder}/test/arch{test_count}.json', 'w') as f:
                json.dump(arch, f)
            test_count += 1


    
