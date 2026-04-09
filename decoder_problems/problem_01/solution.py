"""
Problem 01: Weighting Function Generation

Complete solution implementing the non-uniform weighting function
for D-FINE's distribution refinement.
"""

import torch


def weighting_function(reg_max, up, reg_scale):
    """
    Generate the non-uniform Weighting Function W(n) for bounding box regression.
    
    This function creates importance weights for each bin in a regression distribution,
    with higher weights for central bins and lower weights for edge bins. This guides
    the model to produce confident, centered predictions.
    
    Args:
        reg_max (int): Max number of the discrete bins. Range: [1, 32]
        up (torch.Tensor): Upsampling factor controlling decay steepness.
                          Shape: (1,), dtype: float32. Range: [1, 8]
        reg_scale (torch.Tensor): Scaling multiplier for final weights.
                                 Shape: (1,), dtype: float32. Range: [0.5, 2.0]
    
    Returns:
        torch.Tensor: Weighting function values.
                     Shape: (reg_max+1,), dtype: float32
                     Property: All values are non-negative, symmetric around center,
                              with center having maximum value.
    
    Formula:
        w(i) = (1 / (1 + up * |i - center|)) * reg_scale
        where center = reg_max / 2
    
    Example:
        >>> reg_max = 3
        >>> weights = weighting_function(3, torch.tensor([2.0]), torch.tensor([1.0]))
        >>> weights.shape
        torch.Size([4])
    """
    # Get device from input tensors
    device = up.device
    dtype = up.dtype
    
    # Create bin indices
    indices = torch.arange(reg_max + 1, dtype=dtype, device=device)
    
    # Compute center position
    center = reg_max / 2.0
    
    # Calculate distance from center
    distance = torch.abs(indices - center)
    
    # Apply weighting formula: w(i) = 1 / (1 + up * distance)
    denominator = 1.0 + up * distance
    weights = 1.0 / denominator
    
    # Apply scaling
    weights = weights * reg_scale
    
    return weights
