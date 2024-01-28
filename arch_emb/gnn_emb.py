import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear = torch.nn.ModuleList()
            self.bn = torch.nn.ModuleList()
            self.linear.append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_layers-2):
                self.linear.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear.append(nn.Linear(hidden_dim, output_dim))

            for i in range(num_layers-1):
                self.bn.append(nn.BatchNorm1d((hidden_dim)))

        
    def forward(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        else:
            x_ = x
            for i in range(self.num_layers-1):
                x_ = F.relu(self.bn[i](self.linear[i](x_)))

            return self.linear[-1](x_)


class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim, dropout):
        super(Decoder, self).__init__()
        self.fc = torch.nn.Linear(output_dim, input_dim)
        self.dropout = dropout

    def forward(self, mu):
        mu = F.dropout(mu, p=self.dropout, training=self.training)
        ops = torch.softmax(self.fc(mu), dim=0)
        adj = torch.sigmoid(torch.matmul(mu, mu.permute(0,2,1)))
        return ops, adj


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_rec, num_layers, dropout):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_rec = num_rec
        self.e = nn.Parameter(torch.zeros(self.num_rec - 1))
        self.linear = torch.nn.ModuleList()
        self.bnorm = torch.nn.ModuleList()
        self.linear.append(MultiLayerPerceptron(num_layers, input_dim, hidden_dim, hidden_dim))
        self.bnorm.append(nn.BatchNorm1d(hidden_dim))
        for i in range(1, self.num_rec - 1):
            self.linear.append(MultiLayerPerceptron(num_layers, hidden_dim, hidden_dim, hidden_dim))
            self.bnorm.append(nn.BatchNorm1d(hidden_dim))

        self.mu = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigma = nn.Linear(self.hidden_dim, self.output_dim)

        self.decoder = Decoder(self.output_dim, self.input_dim, dropout)


    def encoder(self, ops, adj):
        bs, num_nodes, num_opt = ops.shape
        ops_ = ops
        for i in range(self.num_rec - 1):
            a = torch.matmul(adj.float(), ops_)
            gin_update = (1+self.e[i]) * ops_.view(bs*num_nodes, -1) + a.view(bs*num_nodes, -1)
            ops_ = F.relu(self.bnorm[i](self.linear[i](gin_update))).view(bs, num_nodes, -1)

        return self.mu(ops_), self.sigma(ops_)


    def forward(self, ops, adj):
        mu, sigma = self.encoder(ops, adj)
        mu_ = torch.randn_like(sigma).mul(torch.exp(sigma)).add_(mu) if self.training else mu
        new_ops, new_adj = self.decoder(mu_)
        return new_ops, new_adj, mu_, sigma

    


