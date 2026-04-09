"""Problem 05: Sampling Offsets - Solution"""
import torch
import torch.nn as nn
import torch.nn.init as init
import math

def init_sampling_offsets(num_heads, num_levels, num_points_list, embed_dim):
    """Initialize offset linear layer with radial grid bias."""
    total_points = num_heads * sum(num_points_list)
    layer = nn.Linear(embed_dim, total_points * 2)
    
    # Initialize weight to 0
    init.constant_(layer.weight, 0)
    
    # Create radial grid initialization for bias
    thetas = torch.arange(num_heads, dtype=torch.float32) * (2.0 * math.pi / num_heads)
    grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
    grid_init = grid_init.reshape(num_heads, 1, 2).tile([1, sum(num_points_list), 1])
    
    scaling = torch.concat([torch.arange(1, n + 1) for n in num_points_list]).reshape(1, -1, 1)
    grid_init *= scaling
    layer.bias.data[...] = grid_init.flatten()
    
    return layer
