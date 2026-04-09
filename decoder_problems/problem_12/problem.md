# Problem 12 - MSDeformableAttention (Multi-Scale)

## Description
Full multi-scale deformable attention. Combines:
1. Sampling offset computation (learnable)
2. Attention weight computation (softmax)
3. Offset normalization per level
4. Multi-level attention calls and fusion

## Input/Output
- query: (B, L, D)
- reference_points: (B, L, num_levels, 4)
- value: tuple of tensors per level
- spatial_shapes: list of (H,W)
- output: (B, L, D)

Key challenge: Proper offset normalization across scales.
