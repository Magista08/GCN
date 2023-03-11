# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    def __init__(self, _in_feature, _out_feature):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(_in_feature, _out_feature, bias=False)

    def forward(self, _x: torch.Tensor, _adjacency_hat: torch.sparse_coo_tensor):
        x = self.linear(_x)
        x = torch.sparse.mm(_adjacency_hat, x)
        return x


class GCN(nn.Module):
    def __init__(self, _input_size, _hidden_size, _output_size, _num_hidden_layers=0, _dropout=0.1, _residual=False):
        super(GCN, self).__init__()

        self.dropout = _dropout
        self.residual = _residual

        self.input_conv = GCNConv(_input_size, _hidden_size)
        self.output_conv = GCNConv(_hidden_size, _output_size)

        self.hidden_convs = nn.ModuleList([GCNConv(_hidden_size, _hidden_size) for _ in range(_num_hidden_layers)])

    def forward(self, _x: torch.Tensor, _adjacency_hat: torch.sparse_coo_tensor, _labels: torch.Tensor = None):
        x = F.dropout(_x, p=self.dropout, training=self.training)
        x = F.relu(self.input_conv(x, _adjacency_hat))
        for conv in self.hidden_convs:
            if self.residual:
                x = F.relu(conv(x, _adjacency_hat)) + x
            else:
                x = F.relu(conv(x, _adjacency_hat))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_conv(x, _adjacency_hat)

        if _labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, _labels)
        return x, loss
