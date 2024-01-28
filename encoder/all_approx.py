import torch
import torch.nn as nn
from torch.nn import Linear, LSTM, Dropout, KLDivLoss, ReLU
import numpy as np

# random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class LSTMApproximator(nn.Module):
    def __init__(self, arch_dim=16, data_dim=2048, dropout_linear=0.3, rnn_layers=1, dropout_rnn=0.3, dim_logits=10, hidden_dim=256):
        super(LSTMApproximator, self).__init__()
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
    def __init__(self, arch_in_features=16, data_in_features=2048, dim_logits=10, hidden_dim=256):
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
    def __init__(self, pool="max", inp_dim=16, h_dim=64, data_dim=2048, output_dim=10):
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
            nn.Linear(in_features=h_dim+data_dim, out_features=output_dim),
            nn.ReLU(),
            nn.Linear(in_features=output_dim, out_features=output_dim),
        )
        self.pool = pool

    def forward(self, x, data):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(torch.cat([x,data],dim=0))
        return x