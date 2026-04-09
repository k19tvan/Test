"""
Problem 03: Inverse Sigmoid Transformation

Starter skeleton for numerically stable inverse logit transformation.
"""

import torch


def inverse_sigmoid(x, eps=1e-5):
    """
    Compute the numerically stable inverse sigmoid (logit) transformation.
    
    Maps values from [0, 1] to (-∞, +∞) using the logit function:
    logit(x) = log(x / (1 - x))
    
    Args:
        x (torch.Tensor): Input values in range [0, 1]. Shape: (B,), dtype: float32
        eps (float): Epsilon for numerical stability. Default: 1e-5
    
    Returns:
        torch.Tensor: Output values in unconstrained space. Shape: same as x, dtype: float32
    
    Notes:
        - Uses clamping to avoid log(0) and division by zero
        - Result is symmetric around 0: inverse_sigmoid(0.5) = 0
        - For edge values: inverse_sigmoid(0) ≈ -10.6, inverse_sigmoid(1) ≈ 10.6
    """
    raise NotImplementedError()
