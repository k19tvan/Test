# Problem 11 - Deformable Attention (Single Level)

## Description
Core deformable attention mechanism. Uses bilinear sampling from feature maps at deformed grid positions, then applies attention weighting.

## Key Steps
1. Reshape value to (B*num_heads, C, H, W)
2. Create sampling grid from locations
3. Use F.grid_sample for bilinear interpolation
4. Apply attention weights via multiplication and summation

## Input/Output
- value: (B, HW, C)
- sampling_locations: (B, L, H, P, 2) where H=heads, P=points
- attention_weights: (B, L, H, P)
- output: (B, L, C)
