from torch import nn


class XASBlock(nn.Sequential):
    DROPOUT = 0.5

    def __init__(self, dims):
        layers = []
        for i, (w1, w2) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(w1, w2))
            if i < len(dims) - 2:  # not last layer
                layers.append(nn.BatchNorm1d(w2))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(self.DROPOUT))
            else:
                layers.append(nn.Softplus())  # last layer

        super().__init__(*layers)
