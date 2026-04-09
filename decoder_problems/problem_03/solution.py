"""
Problem 03: Inverse Sigmoid Transformation

Complete solution for numerically stable inverse logit computation.
"""

import torch


def inverse_sigmoid(x, eps=1e-5):
    """
    Compute the numerically stable inverse sigmoid (logit) transformation.
    
    Maps values from [0, 1] to (-∞, +∞) using:
    logit(x) = log(x_clamped / (1 - x_clamped))
    where x_clamped = clip(x, eps, 1 - eps)
    
    Args:
        x (torch.Tensor): Input values in range [0, 1]. Shape: (B,...), dtype: float32
        eps (float): Epsilon for numerical stability. Default: 1e-5. Prevents log(0).
    
    Returns:
        torch.Tensor: Output logit values in unconstrained space.
                     Shape: same as x, dtype: float32
    
    Properties:
        - sigmoid(inverse_sigmoid(x)) ≈ x for x in (0, 1)
        - inverse_sigmoid(0.5) = 0 (antisymmetric around 0.5)
        - inverse_sigmoid(x) = -inverse_sigmoid(1-x)
        - Output range for typical eps=1e-5: approximately [-10.6, 10.6]
    
    Example:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> y = inverse_sigmoid(x)
        >>> torch.sigmoid(y)  # Approximately x
    """
    # Clamp input to avoid log(0) and division by zero
    x = x.clamp(min=eps, max=1 - eps)
    
    # Compute inverse sigmoid: log(x / (1-x))
    return torch.log(x / (1 - x))
