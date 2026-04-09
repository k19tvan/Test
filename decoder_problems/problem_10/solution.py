"""Problem 10: Solution"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        layers = []
        for i in range(num_layers):
            in_dim = 4 * (k + 1) if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.reg_conf = nn.Sequential(*layers)
        init.constant_(self.reg_conf[-1].bias, 0)
        init.constant_(self.reg_conf[-1].weight, 0)
    
    def forward(self, scores, pred_corners):
        B, L, C, num_bins = pred_corners.size()
        prob = F.softmax(pred_corners, dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score
