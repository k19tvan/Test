"""Problem 12: MSDeformableAttention"""
import torch
import torch.nn as nn

class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points if isinstance(num_points, list) else [num_points]*num_levels
        self.total_points = sum(self.num_points) * num_heads
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
    
    def forward(self, query, reference_points, value, value_spatial_shapes):
        raise NotImplementedError()
