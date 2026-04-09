"""Problem 07: MLP Forward Pass - Starter"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = self._get_activation(act)
    
    def _get_activation(self, act_name):
        if act_name == "relu":
            return nn.ReLU()
        elif act_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act_name}")
    
    def forward(self, x):
        raise NotImplementedError()
