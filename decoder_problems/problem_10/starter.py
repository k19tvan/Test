"""Problem 10: LQE"""
import torch
import torch.nn as nn
import torch.nn.init as init

class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = self._build_mlp(4 * (k + 1), hidden_dim, 1, num_layers)
    
    def _build_mlp(self, in_dim, hidden_dim, out_dim, num_layers):
        raise NotImplementedError()
    
    def forward(self, scores, pred_corners):
        raise NotImplementedError()
