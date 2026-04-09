"""
Problem 02: Translate GT to Distribution Bins

Complete solution for converting continuous GT values into discrete bins
with linear interpolation weights.
"""

import torch
import sys
import importlib.util

# Load solution from problem_01 explicitly
spec = importlib.util.spec_from_file_location("problem_01_solution", "/home/enn/workspace/Test/decoder_problems/problem_01/solution.py")
problem_01_solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problem_01_solution)
weighting_function = problem_01_solution.weighting_function


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
    
    Example:
        gt = torch.tensor([1.5])
        indices, wl, wr = translate_gt(gt, 8, torch.tensor([4.0]), torch.tensor([2.0]))
        # Returns: indices at bin position, wl + wr = 1.0
    """
    # Reshape GT to 1D to handle edge cases
    gt = gt.reshape(-1)
    
    # Get device and dtype
    device = gt.device
    dtype = gt.dtype
    
    # Get the weighting function values (these represent bin positions)
    function_values = weighting_function(reg_max, up, reg_scale)
    
    # Find the closest left-side indices for each value
    # Create boolean mask: function_values[i] <= gt[j]
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)  # (B, reg_max+1)
    mask = diffs <= 0  # True where function_values <= gt
    
    # Count how many values satisfy the condition (cumulative)
    closest_left_indices = torch.sum(mask, dim=1) - 1  # (B,)
    
    # Convert to float for later operations
    indices = closest_left_indices.float()
    
    # Clamp to valid range [0, reg_max-1]
    indices_clamped = torch.clamp(indices, 0, reg_max - 1).long()
    
    # Prepare output tensors
    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)
    
    # Create mask for valid indices
    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices_clamped[valid_idx_mask]
    
    if valid_indices.numel() > 0:
        # Get left and right boundary values
        left_values = function_values[valid_indices]
        right_values = function_values[valid_indices + 1]
        
        # Get GT values for valid indices
        valid_gt = gt[valid_idx_mask]
        
        # Compute distances
        left_diffs = torch.abs(valid_gt - left_values)
        right_diffs = torch.abs(right_values - valid_gt)
        
        # Compute total distance
        total_diffs = left_diffs + right_diffs
        
        # Avoid division by zero
        total_diffs = torch.clamp(total_diffs, min=1e-8)
        
        # Compute interpolation weights (proportional to opposite side's distance)
        weight_right[valid_idx_mask] = left_diffs / total_diffs
        weight_left[valid_idx_mask] = right_diffs / total_diffs
    
    # Clamp weights to [0, 1]
    weight_right = torch.clamp(weight_right, 0, 1)
    weight_left = torch.clamp(weight_left, 0, 1)
    
    # Ensure weights sum to approximately 1.0
    weight_sum = weight_left + weight_right
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    weight_left = weight_left / weight_sum
    weight_right = weight_right / weight_sum
    
    return indices_clamped.float(), weight_left, weight_right
