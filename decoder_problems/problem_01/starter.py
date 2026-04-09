"""
Problem 01: Weighting Function Generation

Starter skeleton for implementing the non-uniform weighting function
used in D-FINE's distribution refinement approach.
"""

import torch


def weighting_function(reg_max, up, reg_scale):
    """
    Generate the non-uniform Weighting Function W(n) for bounding box regression.
    
    Args:
        reg_max (int): Max number of the discrete bins. Range: [1, 32]
        up (torch.Tensor): Upsampling factor. Shape: (1,), dtype: float32
        reg_scale (torch.Tensor): Scaling factor. Shape: (1,), dtype: float32
    
    Returns:
        torch.Tensor: Weighting function values. Shape: (reg_max+1,), dtype: float32
                      Each element w(i) represents importance weight for bin i.
    """
    raise NotImplementedError()
