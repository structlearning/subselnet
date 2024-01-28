import torch
import torch.nn as nn
from torch.nn import Linear, LSTM, Dropout, KLDivLoss, ReLU
import numpy as np

# random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class LSTMApproximator(nn.Module):
    def __init__(self, arch_dim=16, data_dim=2048, hidden_dim=256, dropout_linear=0.3, data_h_dim=16, dropout_compressor=0.3, rnn_layers=1, dropout_rnn=0.0, dim_logits=10):
        super(LSTMApproximator, self).__init__()
        # self.data_compressor = nn.Sequential(Linear(data_dim, 512),
        #                                      Dropout(p=dropout_compressor),
        #                                      ReLU(inplace=True),
        #                                      Linear(512, 32),
        #                                      Dropout(p=dropout_compressor),
        #                                      ReLU(inplace=True),
        #                                      Linear(32, data_h_dim))
        self.rnn = LSTM(arch_dim, arch_dim, rnn_layers, batch_first=True, dropout=dropout_rnn)
        if dim_logits // 10 == 0:
            self.model = nn.Sequential(
                nn.Linear(arch_dim+data_dim, hidden_dim),
                nn.Dropout(p=dropout_linear),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim,dim_logits),
                )
        else:
            self.model = nn.Sequential(
                nn.Linear(arch_dim+data_dim, dim_logits),
                nn.Dropout(p=dropout_linear),
                nn.ReLU(inplace=True),
                nn.Linear(dim_logits,dim_logits),
                )

    def forward(self, h_sequence, data):
        out, (_, _) = self.rnn(h_sequence)
        x = torch.cat([data, out[:,-1]], dim=1)
        logits = self.model(x)
        return logits


class VanillaApproximator(nn.Module):
    def __init__(self, arch_in_features=16, data_in_features=2048, hidden_dim=256, dim_logits=10):
        super(VanillaApproximator, self).__init__()
        self.arch_in_features = arch_in_features
        self.data_in_features = data_in_features

        if dim_logits // 10 == 0:
            self.model = nn.Sequential(
                nn.Linear(arch_in_features+data_in_features, hidden_dim),
                nn.Dropout(p=0.3),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim,dim_logits),
                )
        else:
            self.model = nn.Sequential(
                nn.Linear(arch_in_features+data_in_features, dim_logits),
                nn.Dropout(p=0.3),
                nn.ReLU(inplace=True),
                nn.Linear(dim_logits,dim_logits),
                )
            
    def forward(self, x):
        logits = self.model(x)
        return logits

class AttendApproximator(nn.Module):
    def __init__(self, arch_dim=16, data_dim=2048, dropout_linear=0.3, dim_logits=10, hidden_dim=256):
        super(AttendApproximator, self).__init__()
        d = arch_dim; h = arch_dim//2; m = h**2
        self.Q = Linear(d, h, bias=False)
        self.K = Linear(d, h, bias=False)
        self.V = Linear(d, h, bias=False)
        self.C = Linear(h, d, bias=False)
        self.Norm1 = nn.LayerNorm(d)
        self.LRL = nn.Sequential(Linear(d,m), ReLU(), Linear(m,d))
        self.Norm2 = nn.LayerNorm(d)

        if dim_logits // 100 == 0:
            self.model = nn.Sequential(nn.Linear(arch_dim+data_dim, hidden_dim),
                                   nn.Dropout(p=dropout_linear),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.Dropout(p=dropout_linear),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(hidden_dim,dim_logits))
        else:
            self.model = nn.Sequential(nn.Linear(arch_dim+data_dim, dim_logits),
                                   nn.Dropout(p=dropout_linear),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(dim_logits, dim_logits),
                                   nn.Dropout(p=dropout_linear),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(dim_logits,dim_logits))

    def attention(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        alpha_prime = torch.bmm(q, k.permute(0,2,1))
        alpha = alpha_prime.softmax(dim=2)
        alpha_v = torch.bmm(alpha, v)
        u_prime = self.C(alpha_v)
        u = self.Norm1(x+u_prime)
        z_prime = self.LRL(u)
        z = self.Norm2(z_prime+u)
        return z[:,-1]

    def forward(self, h_sequence, data):
        z = self.attention(h_sequence)
        x = torch.cat([data, z], dim=1)
        logits = self.model(x)
        return logits


class SmallDeepSet(nn.Module):
    def __init__(self, pool="max", inp_dim=16, h_dim=64, output_dim=10):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=inp_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Linear(in_features=h_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Linear(in_features=h_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Linear(in_features=h_dim, out_features=h_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=h_dim, out_features=output_dim),
            nn.ReLU(),
            nn.Linear(in_features=output_dim, out_features=output_dim),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x

class MasterApproximator(nn.Module):
    def __init__(self, step1='rnn', step2='attention-block', hidden_dim=256, rnn_layers=1, dropout_rnn=0.0, arch_dim=16, data_dim=2048, u_dim=16):
        super(MasterApproximator, self).__init__()
        self.arch_dim = arch_dim
        self.data_dim = data_dim
        
        ######## Block for step 1 ########
        if step1 == 'rnn':
            self.f1 = nn.Sequential(Linear(data_dim, hidden_dim), ReLU(), Linear(hidden_dim, u_dim))
            self.lstm = LSTM(arch_dim, u_dim, rnn_layers, batch_first=True, dropout=dropout_rnn)
            
            self.step1_fn = self.rnn
        
        elif step1 == 'ffn':
            self.f1 = nn.Sequential(Linear(arch_dim+data_dim, hidden_dim), ReLU(), Linear(hidden_dim, u_dim))
            self.f2 = nn.Sequential(Linear(arch_dim+u_dim, u_dim), ReLU(), Linear(u_dim,u_dim))
        
            self.step1_fn = self.ffn
        
        else:
            raise Exception(f"step1 can only be 'rnn' or 'ffn' but you entered '{step1}'")

        ######## Block for step 2 ########
        if step2 == 'attention-block':
            d = u_dim; h = arch_dim//2; m = h**2
            self.Q = Linear(d, h, bias=False)
            self.K = Linear(d, h, bias=False)
            self.V = Linear(d, h, bias=False)
            self.C = Linear(h, d, bias=False)
            self.Norm1 = nn.LayerNorm(d)
            self.LRL = nn.Sequential(Linear(d,m), ReLU(), Linear(m,d))
            self.Norm2 = nn.LayerNorm(d)

            self.step2_fn = self.self_attention

        ######## Block for step 3 ########
        self.step3_fn = SmallDeepSet()

    def forward(self, arch, data):
        u = self.step1_fn(arch, data)
        z = self.step2_fn(u)
        logits = self.step3_fn(z)
        return logits

    def rnn(self, arch, data):
        h = self.f1(data).unsqueeze(0)
        c = torch.zeros_like(h)
        u, (_,_) = self.lstm(arch, (h,c))
        return u

    def ffn(self, arch, data):
        u = []
        u1 = self.f1(torch.cat([data, arch[:,0]], dim=1)).unsqueeze(1)
        u.append(u1)
        for i in range(1, arch.shape[1]):
            u_ = self.f2(torch.cat([u[i-1][:,0], arch[:,i]], dim=1)).unsqueeze(1)
            u.append(u_)
        return torch.cat(u, dim=1)

    def self_attention(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        alpha_prime = torch.bmm(q, k.permute(0,2,1))
        alpha = alpha_prime.softmax(dim=2)
        alpha_v = torch.bmm(alpha, v)
        u_prime = self.C(alpha_v)
        u = self.Norm1(x+u_prime)
        z_prime = self.LRL(u)
        z = self.Norm2(z_prime+u)
        return z