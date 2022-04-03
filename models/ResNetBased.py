import torch
import torch.nn as nn
from typing import List



class ResnetBasedModel(nn.Module):
    def __init__(self,
                 res_net_out_dim:int,
                 dropout_mlp: float,
                 input_dim: int,
                 n_classes: int,
                 hidden_dims: List[int]
                ):

        super().__init__()
        
        mlp_layers: List[nn.Module] = []
        self.input_dim = input_dim + res_net_out_dim
        self.hidden_dims = hidden_dims
        self.dropout_mlp = dropout_mlp
        self.n_classes = n_classes
        self.res_net_out_dim = res_net_out_dim

        for hidden_dim in self.hidden_dims:
            mlp_layers.append(nn.Linear(self.input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            # mlp_layers.append(nn.Dropout(self.dropout_mlp))
            self.input_dim = hidden_dim

        mlp_layers.append(nn.Linear(self.input_dim , self.n_classes))

        self.mlp_nn: nn.Module = nn.Sequential(*mlp_layers)
        self.m = nn.LogSoftmax(dim=1)
        
        self.res_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # self.res_model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,bias=False)

    def forward(self, img_data, x):
        res_out = self.res_model(img_data)
        concat = torch.cat([res_out, x], 1)
        output = self.mlp_nn(concat)
        
        output = self.m(output)
        
        return output