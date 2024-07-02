from typing import List, Union
import torch
from torch import nn
from typing import Literal
from src.data.ml_data import load_xas_ml_data, DataQuery


class FC_XAS(nn.Module):
    def __init__(
        self,
        widths: List[int],
        input_dim: Union[int, Literal["DATADIM"], None] = "DATADIM",
        output_dim: Union[int, Literal["DATADIM"], None] = "DATADIM",
        dropout_rate=0.5,
        compound: str = None,
        simulation_type: str = None,
    ):
        super().__init__()
        self.widths = widths

        if input_dim == "DATADIM" or output_dim == "DATADIM":
            ml_data = load_xas_ml_data(DataQuery(compound, simulation_type))
            if input_dim == "DATADIM":
                self.widhts = [ml_data.train.X.shape[1]] + self.widths
            if output_dim == "DATADIM":
                self.width = self.widths + [ml_data.train.y.shape[1]]
        if input_dim is not None:
            self.widths = [input_dim] + self.widths
        if output_dim is not None:
            self.widths = self.widths + [output_dim]

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
