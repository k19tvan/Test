"""Problem 08: Solution"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class Integral(nn.Module):
    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max
    
    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])
