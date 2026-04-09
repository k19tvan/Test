# Problem 03 - Inverse Sigmoid Transformation

## Description
Bounding box coordinates in neural networks are typically normalized to [0, 1] range using sigmoid activation. However, when we need to predict offsets or refinements to these coordinates, we must work in the unconstrained space (-∞, +∞). The inverse sigmoid function (logit) maps [0, 1] back to real space. Numerical stability is critical to avoid log(0) or division issues.

## Input Format
- **x** (torch.Tensor): Input values to be transformed. Shape: (B,) where B is batch size, dtype: float32, Range: [0, 1]

Optional parameters:
- **eps** (float): Epsilon for numerical stability. Default: 1e-5, Range: [1e-8, 1e-3]

## Output Format
- Return type: torch.Tensor
- Shape: Same as input x, (B,)
- Dtype: float32
- Value range: (-∞, +∞), typically in range [-10, +10] for normal inputs
- Property: Inverse of sigmoid function: sigmoid(inverse_sigmoid(x)) ≈ x

## Constraints
- Input x must be in [0, 1] range
- Output can be any real value
- Must handle edge cases: x=0 and x=1 gracefully
- Must not produce NaN or Inf for valid inputs (after clamping)
- Device: Same device as input x tensor
- Numerical stability required: Use clamping and epsilon to prevent log(0)

## Example
Input:
```
x = torch.tensor([0.0, 0.5, 1.0])
eps = 1e-5
```

Expected Output:
```
tensor([-10.5966,   0.0000,  10.5966])
# Approximate: invlogit(0) ≈ -10.6 (clamped)
#              invlogit(0.5) = 0
#              invlogit(1) ≈ 10.6 (clamped)
```

## Hints
- The mathematical formula is: invlogit(x) = log(x / (1 - x))
- Avoid log(0) by clamping: x_clamped = clip(x, eps, 1-eps)
- Use the clamped version for log computation: log(x_clamped / (1 - x_clamped))
- For x=0.5, the result should be exactly 0 (since log(1) = 0)
- For values near 0, result approaches log(eps / (1-eps)), which is negative
- For values near 1, result approaches log((1-eps) / eps), which is positive
- Use torch.log for computation

## Checker
Run with: `python problem_03/checker.py`
Expected output: `All Problem 03 checks passed`
