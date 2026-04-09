"""Problem 09: Solution"""
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
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)
