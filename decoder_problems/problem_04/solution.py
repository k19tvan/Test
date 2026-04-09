"""
Problem 04: Reference Point Generation

Complete solution for multi-scale anchor grid generation.
"""

import torch
import importlib.util

# Load solution from problem_03 explicitly
spec = importlib.util.spec_from_file_location("problem_03_solution", "/home/enn/workspace/Test/decoder_problems/problem_03/solution.py")
problem_03_solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problem_03_solution)
inverse_sigmoid = problem_03_solution.inverse_sigmoid


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
        device (str or torch.device): Output tensor device. Default: "cpu"
    
    Returns:
        tuple:
            - anchors (torch.Tensor): Reference points in log-sigmoid space.
                                     Shape: (1, total_points, 4), coordinates [x,y,w,h]
            - valid_mask (torch.Tensor): Validity flags. Shape: (1, total_points, 1)
                                        Value 1 if valid (in [eps, 1-eps]), 0 otherwise
    
    Formula:
        1. For each level ℓ with stride s_ℓ:
           - Level dimensions: H_ℓ = eval_h / s_ℓ, W_ℓ = eval_w / s_ℓ
           - Grid coordinates: Normalize to [0,1]
           - Width/height: grid_size * 2^ℓ where grid_size = 0.05
        2. Transform to log-sigmoid space: log(x/(1-x))
        3. Create validity mask for points in [eps, 1-eps]
    """
    eval_h, eval_w = eval_spatial_size
    
    anchors = []
    valid_masks = []
    
    for level_idx, stride in enumerate(feat_strides):
        # Compute level grid dimensions
        level_h = int(eval_h / stride)
        level_w = int(eval_w / stride)
        
        # Create 2D meshgrid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(level_h, dtype=dtype, device=device),
            torch.arange(level_w, dtype=dtype, device=device),
            indexing='ij'
        )
        
        # Normalize to [0, 1], placing coordinates at pixel centers
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)
        grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([level_w, level_h], dtype=dtype, device=device)
        
        # Add width/height channels
        # grid_size provides base scale; multiply by 2^level for scale-specific sizes
        grid_size = 0.05  # Base scale factor
        wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** level_idx)
        
        # Concatenate to get [x, y, w, h]
        level_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, level_h * level_w, 4)
        anchors.append(level_anchors)
        
        # Create validity mask: points in [eps, 1-eps] are valid
        valid_mask = ((grid_xy > eps) * (grid_xy < 1 - eps)).all(-1, keepdim=True)
        valid_masks.append(valid_mask.reshape(-1, level_h * level_w, 1).float())
    
    # Concatenate across levels
    anchors = torch.concat(anchors, dim=1)  # (1, total_points, 4)
    valid_mask = torch.concat(valid_masks, dim=1)  # (1, total_points, 1)
    
    # Transform to log-sigmoid space (unconstrained coordinates)
    anchors_transformed = inverse_sigmoid(anchors)
    
    # Mask invalid points: set their anchors to inf
    anchors_transformed = torch.where(valid_mask.bool(), anchors_transformed, torch.inf)
    
    return anchors_transformed, valid_mask
