"""Problem 08: Integral"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class Integral(nn.Module):
    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max
    
    def forward(self, x, project):
        raise NotImplementedError()
