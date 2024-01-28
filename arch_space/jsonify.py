from nasbench import api
import numpy as np
import json
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--nasbench_path', help='Path to the nasbench record file')
parser.add_argument('--out_folder', help='Path to the directory to save the json files')
args = parser.parse_args()


def get_ops(ops):
    mapping = {INPUT: 0, CONV1X1: 1, CONV3X3: 2, MAXPOOL3X3: 3, OUTPUT: 4}
    ops_ = np.zeros([7,5], dtype=np.int8)
    for i, op in enumerate(ops):
        ops_[i, mapping[op]] = 1

    return ops_

def get_arch_adj_ops(nb):
    count = 0
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

    
        yield {count:{
                    'module_adjacency': adj_list,
                    'module_operations': get_ops(ops).tolist()
                }
        }
        count += 1

if __name__ == '__main__':

    INPUT = 'input'
    OUTPUT = 'output'
    CONV1X1 = 'conv1x1-bn-relu'
    CONV3X3 = 'conv3x3-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'

    nb = api.NASBench(args.nasbench_path)
    all_archs = get_arch_adj_ops(nb)
    arch_dict = OrderedDict()
    for arch in all_archs:
        arch_dict.update(arch)

    if args.out_folder is None:
        args.out_folder = 'log'
        os.mkdir('log')
        
    with open(f'{args.out_folder}/data.json', 'w') as f:
        json.dump(arch_dict, f)


    
