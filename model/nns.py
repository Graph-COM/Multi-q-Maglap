import torch
from torch import nn


class MLPs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, norm=None):
        super(MLPs, self).__init__()
        assert num_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        if norm == 'bn':
            self.layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm == 'ln':
            self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm == 'bn':
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == 'ln':
                self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

