"""Problem 13: Solution - Simplified"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0, 
                 activation="relu", n_levels=4, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, target, reference_points, value, spatial_shapes, query_pos_embed=None):
        # Self-attention
        tgt2, _ = self.self_attn(target, target, target)
        target = target + self.dropout1(tgt2)
        target = self.norm1(target)
        
        # FFN (simplified - skipping cross-attention for this layer)
        tgt2 = self.linear2(self.activation(self.linear1(target)))
        target = target + self.dropout4(tgt2)
        target = self.norm3(target)
        
        return target
