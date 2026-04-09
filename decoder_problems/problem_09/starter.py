"""Problem 09: Gate"""
import torch
import torch.nn as nn
import torch.nn.init as init

class Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        init.constant_(self.gate.bias, 0)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x1, x2):
        raise NotImplementedError()
