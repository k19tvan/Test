# Problem 04 - Reference Point Generation

## Description
Object queries in the decoder need spatial context. Reference points serve as initial spatial positions for each query object in a multi-scale feature hierarchy. These points are generated as a normalized grid across all feature pyramid levels, then encoded in log-sigmoid space to enable unconstrained refinement during decoding.

## Input Format
- **eval_spatial_size** (tuple): Target evaluation spatial dimensions. Shape: (H, W), e.g., (512, 512). Dtype: int
- **feat_strides** (list): Downsampling strides for each feature level. Shape: (num_levels,), e.g., [8, 16, 32]. Dtype: int
- **eps** (float): Epsilon for numerical stability. Default: 1e-2. Range: [1e-3, 1e-1]

## Output Format
Returns a tuple of 2 tensors:
- **anchors** (torch.Tensor): Reference points in log-sigmoid space. Shape: (1, total_points, 4), dtype: float32
  - Format: (log_x, log_y, log_w, log_h)
  - All values in range (-∞, +∞) due to log transformation
- **valid_mask** (torch.Tensor): Validity flags for each point. Shape: (1, total_points, 1), dtype: float32
  - Value: 1.0 if point is valid (within [eps, 1-eps] after sigmoid), 0.0 otherwise

## Constraints
- total_points = sum of H_i * W_i across all levels where H_i = eval_H / stride_i, W_i = eval_W / stride_i
- All points must be within normalized space [0, 1] before log transformation
- Valid mask prevents gradient flow for invalid (out-of-bounds) points
- Device: CPU (typically, or as specified)

## Example
Input:
```
eval_spatial_size = (512, 512)
feat_strides = [8, 16, 32]
eps = 1e-2
```

Output shapes:
```
Level 0: 64*64 = 4096 points
Level 1: 32*32 = 1024 points  
Level 2: 16*16 = 256 points
Total: 5376 points

anchors shape: (1, 5376, 4)
valid_mask shape: (1, 5376, 1)
```

## Hints
- For each feature level with stride S:
  - Grid height = eval_H / S
  - Grid width = eval_W / S
  - Create 2D meshgrid of indices
- Normalize to [0, 1]: (grid_idx + 0.5) / grid_size
- For width and height: use grid_size as base multiplier (typically 0.05 * 2^level)
- Apply log-sigmoid transform: log(x / (1-x)) to convert to unconstrained space
- Mark invalid points (outside [eps, 1-eps]) with inf in anchors
- Concatenate all levels along sequence dimension
- Remember: anchors shape is (1, total_points, 4) - batch size 1, will be repeated during forward pass

## Checker
Run with: `python problem_04/checker.py`
Expected output: `All Problem 04 checks passed`
