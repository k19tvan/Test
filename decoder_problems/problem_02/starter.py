"""
Problem 02: Translate GT to Distribution Bins

Starter skeleton for converting continuous GT values into discrete bins.
"""

import torch


def translate_gt(gt, reg_max, reg_scale, up):
    """
    Translate continuous ground truth values into distribution bins.
    
    Maps each continuous GT value to the two nearest discrete bins,
    computing interpolation weights for soft label generation.
    
    Args:
        gt (torch.Tensor): Ground truth values. Shape: (B,), dtype: float32
        reg_max (int): Maximum regression value. Range: [1, 32]
        reg_scale (torch.Tensor): Scaling factor. Shape: (1,), dtype: float32
        up (torch.Tensor): Upsampling factor. Shape: (1,), dtype: float32
    
    Returns:
        tuple: Three tensors:
            - indices (torch.Tensor): Left bin indices. Shape: (B,), dtype: float32
            - weight_left (torch.Tensor): Left bin weights. Shape: (B,), dtype: float32
            - weight_right (torch.Tensor): Right bin weights. Shape: (B,), dtype: float32
    """
    raise NotImplementedError()
