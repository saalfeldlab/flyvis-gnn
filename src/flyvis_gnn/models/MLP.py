
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size=None, output_size=None, nlayers=None, hidden_size=None, device=None, activation=None, initialisation=None):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = 0.0
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        if nlayers > 2:
            for i in range(1, nlayers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)

        if initialisation == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif initialisation == 'ones':
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        else :
            nn.init.normal_(layer.weight, std=0.1)
            nn.init.zeros_(layer.bias)

        self.layers.append(layer)

        if activation=='none':
            self.activation = lambda x: x
        elif activation=='tanh':
            self.activation = F.tanh
        elif activation=='sigmoid':
            self.activation = torch.sigmoid
        elif activation=='leaky_relu':
            self.activation = F.leaky_relu
        elif activation=='soft_relu':
            self.activation = F.softplus
        else:
            self.activation = F.relu

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = self.activation(x)
            if self.dropout_rate > 0 and self.training:
                x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.layers[-1](x)
        return x










