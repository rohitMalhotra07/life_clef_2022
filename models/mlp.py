import torch
import torch.nn as nn
from typing import List

# def log_softmax(preds):
#     temperature = preds.max()
#     ex = torch.exp(preds/temperature)
#     return torch.log(ex / torch.sum(ex, axis=0))

class MLP(nn.Module):
    def __init__(self,
                 dropout_mlp: float,
                 input_dim: int,
                 n_classes: int,
                 hidden_dims: List[int]
                ):

        super().__init__()
        
        mlp_layers: List[nn.Module] = []
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_mlp = dropout_mlp
        self.n_classes = n_classes

        for hidden_dim in self.hidden_dims:
            mlp_layers.append(nn.Linear(self.input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            # mlp_layers.append(nn.Dropout(self.dropout_mlp))
            self.input_dim = hidden_dim

        mlp_layers.append(nn.Linear(self.input_dim, self.n_classes))

        self.mlp_nn: nn.Module = nn.Sequential(*mlp_layers)
        self.m = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.mlp_nn(x)
        
        output = self.m(output)
        
        return output