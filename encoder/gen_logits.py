import torch
import numpy as np
from all_approx import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--archemb_file", type=str, required=True, help='Path to architecture embeddings')
parser.add_argument("--dataemb_file", type=str, required=True, help='Path to data embeddings')
parser.add_argument("--load_checkpoint", type=str, required=True, help='Path to approx weights')
parser.add_argument("--experiment_name", type=str, default="model_encoder", help='Name')

args = parser.parse_args()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
print(device)

ma = AttendApproximator(dim_logits=10).to(device)
checkpoint = torch.load(args.load_checkpoint)
ma.load_state_dict(checkpoint['model_state'])

arch_embeddings = torch.load(args.archemb_file)
data_embeddings = torch.load(args.dataemb_file)
num_data = data_embeddings.shape[0]

all_logits = []
with torch.no_grad():
    ma.eval()
    for j in range(len(arch_embeddings)):
        logits = ma(arch_embeddings[j].unsqueeze(0).repeat(num_data,1,1).to(device),
            data_embeddings.to(device))
        all_logits.append(logits.detach().cpu().unsqueeze(0).numpy())

all_logits = np.concatenate(all_logits)
all_logits = torch.from_numpy(all_logits)
if "ma_logits" not in os.listdir():
    os.mkdir("ma_logits")
torch.save(all_logits, f"ma_logits/{args.experiment_name}_logits.pt")
