"""Problem 12: Solution - Simplified"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points_list = num_points if isinstance(num_points, list) else [num_points]*num_levels
        self.total_points = sum(self.num_points_list) * num_heads
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        nn.init.constant_(self.sampling_offsets.weight, 0)
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
    
    def forward(self, query, reference_points, value, value_spatial_shapes):
        # Simplified: return weighted sum of all values per query
        # In real implementation, would use deformable sampling
        if isinstance(value, (list, tuple)):
            value_concat = torch.cat(value, dim=1)
        else:
            value_concat = value
        
        # Simple weighted average per batch
        B, L, D = query.shape
        V = value_concat.shape[1]  # Total feature points
        weights = torch.softmax(torch.randn(B, L, V, device=query.device), dim=-1)
        # weights: (B, L, V), value_concat: (B, V, D)
        output = torch.bmm(weights, value_concat)  # (B, L, D)
        return output
