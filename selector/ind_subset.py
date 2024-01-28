import torch
import os
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import numpy as np
from gallery_approx import *
import argparse
from helper import *


parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
parser.add_argument('--data_file', type=str, help='Path to the data embeddings')
parser.add_argument('--targets_file', type=str, help='Path to the targets')
parser.add_argument("--y_onehot_file", type=str, required=True, help='Path to one-hot targets')
parser.add_argument('--archemb_file', type=str, help='Path to the architecture embeddings')
parser.add_argument('--model_encoder_file', type=str, help='Path to the approx predictions')
parser.add_argument('--subset_size', type=int, help='Subset size (as integer)')
parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--json_folder', help='Path to the folder containing architecture json files')
parser.add_argument('--num_iter', type=int, help='Number of iterations')
parser.add_argument('--lambda_1', type=float, help='Entropy weightage', default=1)
parser.add_argument('--lambda_2', type=float, help='Regularizer weightage', default=0.1)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--output_folder', help='Path to save the generated indices')
parser.add_argument("--num_classes", type=int, default=10, help='Number of classes in dataset')
args = parser.parse_args()


class SoftSelector(nn.Module):
    def __init__(self, targets, data_dim=2048, arch_dim=16, hidden_dim=256, subset_size=500, lambda1=1, lambda2=0.1, num_classes=10):
        super(SoftSelector, self).__init__()
        self.pi = nn.Sequential(nn.Linear(data_dim                  # data input
                                            +arch_dim                 # h.mean()
                                            +num_classes        # one-hot y_true
                                            +num_classes,       # sa_feature
                                            hidden_dim),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_dim, arch_dim), nn.LeakyReLU(), nn.Linear(arch_dim, 1), nn.Sigmoid())
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.subset_size = subset_size
        self.tgt_indices = []
        self.num_classes = num_classes
        self.data_len = len(targets)
        for i3 in range(num_classes):
            self.tgt_indices.append(torch.where(targets == i3))

    def forward(self,  h, m_logits, x, y, loss_xy):
        input = torch.cat([h.mean(dim=0).unsqueeze(0).repeat(self.data_len,1), x, y, m_logits], dim=1)
        pi_xy = self.pi(input).squeeze()
        top_k, _ = torch.topk(pi_xy,self.subset_size)
        obj_fn = torch.dot(pi_xy, loss_xy) + self.lambda1*(pi_xy.sum()-self.subset_size).norm() - self.lambda2*self.entropy(pi_xy)
        return obj_fn, pi_xy, top_k
            
    def entropy(self, pi_xy):
        return torch.distributions.Categorical(probs=pi_xy.softmax(dim=0)).entropy()
    

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(device)

train_indices = len(os.listdir(args.json_folder))
x_data = torch.load(args.data_file)
targets_data = torch.load(args.targets_file)

y_onehot = torch.load(args.y_onehot_file)
arch_embeddings = torch.load(args.archemb_file)
y_pred = torch.load(args.model_encoder_file)

loss_fn = nn.CrossEntropyLoss(reduction='none')

loss_with_ma = []
with torch.no_grad():
    for i2 in range(len(arch_embeddings)):
        loss_with_ma.append(loss_fn(y_pred[i2], targets_data).unsqueeze(0))
loss_with_ma = torch.cat(loss_with_ma, dim=0)

print("Started Training...")

subset_size =args.subset_size

selector = SoftSelector(targets_data, subset_size=subset_size, lambda1=args.lambda_1, lambda2=args.lambda_2, num_classes=args.num_classes).to(device)
optimizer = optim.Adam(selector.parameters(), lr=0.001)

checkpoints = []
pi_list = 0.0
num_epochs = 20
sample_size = 20

for epoch in range(num_epochs):
    print(f"========== epoch: {epoch+1}==========")
    
    main_obj_func = 0.0
    optimizer.zero_grad()
    
    for i in range(len(arch_embeddings)):
        obj_fn, pi_xy, top_k = selector(arch_embeddings[i].to(device), y_pred[i].to(device), x_data.to(device), y_onehot.to(device), loss_with_ma[i].to(device))
        
        main_obj_func += obj_fn
        
        if epoch == num_epochs-1:
            pi_list += pi_xy.cpu().data.numpy()
    
    
    print(main_obj_func.item())
    checkpoints.append(main_obj_func.item())
    main_obj_func.backward()
    optimizer.step()


print('Finished Training')

print(pi_list.sum()/sample_size)
print(pi_xy.sum())
torch.save(selector.state_dict(), f"selector_weights_{subset_size}.pt")
