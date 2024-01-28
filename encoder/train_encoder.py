import argparse
import pickle
import random
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
import os
from all_approx import *
from helper import *


parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
parser.add_argument("--device", type=int, default=0, help='cuda device id')
parser.add_argument("--archemb_file", type=str,required=True, help='Path to architecture embeddings')
parser.add_argument("--dataemb_file", type=str,required=True, help='Path to data embeddings')
parser.add_argument("--logit_train_file", type=str, help='Path to training logits')
parser.add_argument("--logit_test_file", type=str, help='Path to testing logits')
parser.add_argument("--logit_train_indices", type=str, help='Path to architecture idxs for training set')
parser.add_argument("--logit_test_indices", type=str, help='Path to architecture idxs for testing set')
parser.add_argument("--load_checkpoint", type=str, help='Load checkpoint to resume')
parser.add_argument("--save_checkpoint", type=str, default="model_encoder_checkpoint.pt", help='Path to save weights')
parser.add_argument("--experiment_name", type=str, default="model_encoder", help='Name')
parser.add_argument("--hidden_dim", type=int, default=256, help='Hidden dimension for FFN')
parser.add_argument("--arch_dim", type=int, default=16, help='Hidden dimension for embeddings')
parser.add_argument("--num_classes", type=int, default=10, help='Number of classes in dataset')
args = parser.parse_args()

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
print(device)

class CrossProductInMemoryDataset(Dataset):
    def __init__(self, arch_embeddings_file, data_embeddings_file, logit_files_root, indices):
        self.data_embeddings = torch.load(data_embeddings_file)
        self.arch_embeddings = torch.load(arch_embeddings_file)
        self.logits_file = torch.load(logit_files_root)
        self.num_data, self.data_dim = self.data_embeddings.shape
        self.num_archs, self.arch_dim = len(indices), args.arch_dim
        self.indices = indices

    def __len__(self):
        return self.num_data*self.num_archs

    def __getitem__(self, idx):
        arch_idx = self.indices[int(idx) // self.num_data]
        data_idx = idx % self.num_data
        logits = self.logits_file[f'arch{arch_idx}']
        
        return self.arch_embeddings[arch_idx], self.data_embeddings[data_idx], logits[data_idx]

def save_files(kl_array, acc_array):
    if "kl_files" not in os.listdir():
        os.mkdir("kl_files")
    with open(f"kl_files/{args.experiment_name}_kl.pkl", "wb") as f:
        pickle.dump(kl_array, f)
    with open(f"acc_files/{args.experiment_name}_acc.pkl", "wb") as f:
        pickle.dump(acc_array, f)
    

def validate(ma, queue, criterion_logits):
    with torch.no_grad():
        acc_tracker = utils.AvgrageMeter()
        kl_tracker = utils.AvgrageMeter()
        ma.eval()
        for i, (arch_emb, data_emb, logits) in enumerate(queue):
            optimizer.zero_grad()
            arch_emb, data_emb = arch_emb.to(device), data_emb.to(device)
            logits = logits.to(device)
            prob = F.softmax(logits, dim=1)
            pred_logits = model(arch_emb, data_emb) 
            approx_loss = criterion_logits(pred_logits.softmax(dim=1).log(), prob)
            kl_tracker.update(approx_loss.item(), 1)
            acc = utils.accuracy(pred_logits, torch.argmax(prob, dim=1), topk=(1,))[0]
            acc_tracker.update(acc.item(), 1)
            
        return kl_tracker.avg, acc_tracker.avg            

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    with open(args.logit_train_indices, "rb") as f:
        train_indices = pickle.load(f)
    with open(args.logit_test_indices, "rb") as f:
        test_indices = pickle.load(f)
    
    embedding_dataset_trainA_trainD = CrossProductInMemoryDataset(args.archemb_file, args.dataemb_file, args.logit_train_file, train_indices)
    embedding_dataset_valA_trainD = CrossProductInMemoryDataset(args.archemb_file, args.dataemb_file, args.logit_test_file, test_indices)
    
    print("Dataset Sizes:")
    print("Train Arch + Data:", len(embedding_dataset_trainA_trainD))
    print("Val Arch + Data:", len(embedding_dataset_valA_trainD))
    
    trainA_trainD_queue = torch.utils.data.DataLoader(embedding_dataset_trainA_trainD, batch_size=64, pin_memory=True, num_workers=0)
    valA_trainD_queue = torch.utils.data.DataLoader(embedding_dataset_valA_trainD, batch_size=64, pin_memory=True, num_workers=0)
    
    queue_dict = {}
    queue_dict['TATD'] = trainA_trainD_queue
    queue_dict['VATD'] = valA_trainD_queue
    print("Loaded Data")
    
    ########### Load from checkpoint ###########
    model = AttendApproximator(dim_logits=args.num_classes, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
    kl_array = []
    acc_array = []
    epochs_done = 0
    
    if args.load_checkpoint != None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        kl_array = checkpoint['kl']
        acc_array = checkpoint['acc']
        epochs_done = checkpoint['epochs_done']
    print("Loaded Model")
    
    total_epochs = 1000
    loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)
    criterion_logits = nn.KLDivLoss(reduction='batchmean').to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    
    best_avg_kl = 100
    

    print("Beginning Training")
    for epoch in range(epochs_done, total_epochs):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"Starting Epoch {epoch+1} at {dt_string}")
        
        loss_tracker = utils.AvgrageMeter()
        loss_tracker.reset()
        
        model.train()
        for i, (arch_emb, data_emb, logits) in enumerate(trainA_trainD_queue):
            optimizer.zero_grad()
            arch_emb, data_emb = arch_emb.to(device), data_emb.to(device)
            logits = logits.to(device)
            prob = F.softmax(logits, dim=1)
            pred_logits = model(arch_emb, data_emb) 
        
            approx_loss = criterion_logits(pred_logits.softmax(dim=1).log(), prob)
            
            approx_loss.backward()
            loss_tracker.update(approx_loss.item(), 1)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if i == 1 or i % 5000 == 4999:
                print(f'Train epoch {epoch+1}:{i+1}/{len(trainA_trainD_queue)} loss {loss_tracker.avg:.3f}')
                loss_tracker.reset()
                
                avg_kl, avg_acc = validate(model, queue_dict['VATD'], criterion_logits)
                kl_array.append(avg_kl)
                acc_array.append(avg_acc)
                
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print(f"{dt_string}\tVATD: Avg KLDiv: {avg_kl} | Avg Acc: {avg_acc}")
                if avg_kl < best_avg_kl:
                    print("Saving model")
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'kl': kl_array,
                        'acc': acc_array,
                        'epochs_done': epoch
                    }, args.save_checkpoint)
                    best_avg_kl = avg_kl
                
                model.train()
                
            if approx_loss.item() == 0:
                break
        
        scheduler.step()
        