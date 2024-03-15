from typing import List
import torch
from torch import nn


class FC_XAS(nn.Module):
    def __init__(
        self,
        widths: List[int],
        dropout_rate=0.5,
    ):
        super().__init__()
        self.widths = widths
        self.pairs = [(w1, w2) for w1, w2 in zip(self.widths[:-1], self.widths[1:])]
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.dropout_rate = dropout_rate  # to be accessed during fine-tuning
        for i, (w1, w2) in enumerate(self.pairs):
            self.layers.append(nn.Linear(w1, w2))
            if i != len(self.pairs) - 1:
                self.batch_norms.append(nn.BatchNorm1d(w2))
                self.dropouts.append(nn.Dropout(p=self.dropout_rate))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.pairs) - 1:
                x = self.batch_norms[i](x)
                x = x * torch.sigmoid(x)  # swish activation
                x = self.dropouts[i](x)
            else:
                x = self.softplus(x)
        return x
