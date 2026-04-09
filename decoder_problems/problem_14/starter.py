"""Problem 14: Full TransformerDecoder (Capstone)"""
import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_head, reg_max):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_head = num_head
        self.reg_max = reg_max
        self.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
    
    def forward(self, target, ref_points_unact, memory, spatial_shapes, 
                bbox_head, score_head, query_pos_head, pre_bbox_head, 
                integral, up, reg_scale):
        raise NotImplementedError()
