# Problem 01 - Weighting Function Generation

## Description
D-FINE refines bounding box predictions using a distribution-based approach. The weighting function assigns importance scores to each discrete bin in the regression distribution. This function creates non-uniform weights that emphasize central bins over edge bins, enabling the model to focus on meaningful predictions during training.

## Input Format
- **reg_max** (int): Maximum regression value defining the number of bins. Range: 1 ≤ reg_max ≤ 32
- **up** (torch.Tensor): Scalar tensor, upsampling factor for weight decay. Shape: (1,), dtype: float32, Range: 1 ≤ up ≤ 8
- **reg_scale** (torch.Tensor): Scalar tensor, scaling multiplier. Shape: (1,), dtype: float32, Range: 0.5 ≤ reg_scale ≤ 2.0

## Output Format
- Return type: torch.Tensor
- Shape: (reg_max + 1,) - one weight per bin from 0 to reg_max inclusive
- Dtype: float32
- Value range: w(i) ∈ [0, ∞) where center weights are larger than edge weights
- Property: All weights must be non-negative floats

## Constraints
- Output must have exactly reg_max + 1 elements
- All weights must be ≥ 0 (no negative weights)
- Weights must follow a center-biased distribution (higher weights near center)
- Numerical stability required for large reg_max values
- Device: Same device as input tensors (`up` and `reg_scale`)

## Example
Input:
```
reg_max = 3
up = torch.tensor([2.0])
reg_scale = torch.tensor([1.0])
```

Expected Output:
```
Numeric example (centers are higher):
tensor([0.5000, 1.0000, 2.0000, 1.0000, 0.5000])  # Shape: (5,)
# Center at index 2 has highest weight (2.0)
# Edges at indices 0, 4 have lowest weights (0.5)
```

## Hints
- The weighting function creates a Gaussian-like distribution centered at reg_max/2
- Use the formula: w(i) = 1 / (1 + up * |i - center|)
- Center position is reg_max / 2
- For log-space stability with large reg_max, work with scaled version then normalize
- Create indices 0 to reg_max using torch.arange, compute absolute distance from center
- Multiply by the up parameter to scale decay rate
- Add 1 to avoid division by zero, then invert to get weights
- Finally scale by reg_scale parameter
- Edge case: When reg_max = 1, output should be shape (2,) with 2 weights

## Checker
Run with: `python problem_01/checker.py`
Expected output: `All Problem 01 checks passed`
