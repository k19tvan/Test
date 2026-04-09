"""Problem 05: Sampling Offsets - Starter"""
import torch
import torch.nn as nn

def init_sampling_offsets(num_heads, num_levels, num_points_list, embed_dim):
    """Initialize offset linear layer with radial grid bias."""
    total_points = num_heads * sum(num_points_list)
    layer = nn.Linear(embed_dim, total_points * 2)
    raise NotImplementedError()
