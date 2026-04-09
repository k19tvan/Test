"""
Problem 04: Reference Point Generation

Starter skeleton for multi-scale anchor grid generation.
"""

import torch


def _generate_anchors(eval_spatial_size, feat_strides, eps=1e-2, dtype=torch.float32, device="cpu"):
    """
    Generate reference points on a multi-scale feature pyramid.
    
    Creates normalized spatial coordinates for each point in the feature pyramid,
    then transforms to log-sigmoid space for unconstrained refinement.
    
    Args:
        eval_spatial_size (tuple): (eval_h, eval_w) evaluation spatial dimensions
        feat_strides (list): Feature strides for each pyramid level, e.g., [8, 16, 32]
        eps (float): Epsilon for validity mask. Default: 1e-2
        dtype (torch.dtype): Output tensor dtype. Default: torch.float32
        device (str): Output tensor device. Default: "cpu"
    
    Returns:
        tuple:
            - anchors (torch.Tensor): Reference points in log-sigmoid space.
                                     Shape: (1, total_points, 4),  coordinates [x,y,w,h]
            - valid_mask (torch.Tensor): Validity flags. Shape: (1, total_points, 1)
                                        Value 1 if valid (in [eps, 1-eps]), 0 otherwise
    """
    raise NotImplementedError()
