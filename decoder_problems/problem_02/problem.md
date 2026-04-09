# Problem 02 - Translate GT to Distribution Bins

## Description
After defining the weighting function, we need to map ground truth (GT) continuous bounding box coordinates into discrete distribution bins. This process converts target values into a label representation that can optimize the distribution-based regression head. For each GT value, we find the two closest bins and compute linear interpolation weights, enabling smooth gradient flow.

## Input Format
- **gt** (torch.Tensor): Ground truth values to discretize. Shape: (B,) where B is batch size, dtype: float32, Range: arbitrary (usually [0, reg_scale])
- **reg_max** (int): Maximum regression value defining bin count. Range: 1 ≤ reg_max ≤ 32
- **reg_scale** (torch.Tensor): Scaling factor. Shape: (1,), dtype: float32
- **up** (torch.Tensor): Upsampling factor. Shape: (1,), dtype: float32

## Output Format
Returns a tuple of 3 tensors:
- **indices** (torch.Tensor): Shape (B,), dtype: float32. Index of left bin for interpolation.
- **weight_left** (torch.Tensor): Shape (B,), dtype: float32. Interpolation weight for left bin. Range: [0, 1]
- **weight_right** (torch.Tensor): Shape (B,), dtype: float32. Interpolation weight for right bin. Range: [0, 1]

Property: For valid GT values, weight_left + weight_right ≈ 1.0

## Constraints
- All output tensors must be float32
- Shapes must match input batch size B
- Weights must be in [0, 1]
- weights_left + weights_right = 1.0 for valid GT values (approximately)
- Boundary GT values should be handled gracefully (edge clipping)
- Device: Same device as input gt tensor

## Example
Input:
```
gt = torch.tensor([0.0, 1.5, 3.0])  # Shape (3,)
reg_max = 3
reg_scale = torch.tensor([4.0])
up = torch.tensor([2.0])
```

Expected Output:
```
indices: tensor([0., 1., 2.])            # Left bin indices
weight_left: tensor([1.0, 0.5, 0.0])     # Weight for left bin
weight_right: tensor([0.0, 0.5, 1.0])    # Weight for right bin
# Total weights: [1.0, 1.0, 1.0] - sums to 1
```

## Hints
- First, get the weighting function values using the provided weights tensor from problem 01
- The function maps a continuous value to its position in the discrete bin space
- Find the closest left bin by calculating: sum of (weights <= gt_value)
- For boundary cases, clamp indices to valid range [0, reg_max-1]
- Use linear interpolation: weight_right = (gt_value - left_value) / (right_value - left_value)
- weight_left = 1.0 - weight_right
- Handle GT values outside the weighting function range by clipping to valid range
- Vectorize the computation to avoid Python loops
- Edge case: When GT is at exact bin location, weight_left should be 1.0, weight_right should be 0.0

## Checker
Run with: `python problem_02/checker.py`
Expected output: `All Problem 02 checks passed`
