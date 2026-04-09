"""Problem 11: Solution"""
import torch
import torch.nn.functional as F

def deformable_attention_single_level(value, spatial_shape, sampling_locations, attention_weights):
    """Bilinear sampling with attention weighting."""
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape
    h, w = spatial_shape
    
    value_l_ = value.flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
    sampling_grid_l_ = sampling_locations[:, :, :, 0].permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_value_l_ = F.grid_sample(
        value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    
    attn_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points
    )
    output = (
        (sampling_value_l_.reshape(bs * n_head, c, Len_q, -1) * attn_weights)
        .sum(-1)
        .reshape(bs, n_head * c, Len_q)
    )
    
    return output.permute(0, 2, 1)
