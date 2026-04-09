# Problem 13 - TransformerDecoderLayer

## Description
Single decoder layer composing: self-attention → cross-attention → FFN → gating.
This is the fundamental building block that gets stacked 6 times in the full decoder.

## Structure
1. Self-attention with positional encoding
2. Cross-attention via MSDeformableAttention  
3. Gate fusion
4. Feed-forward network
5. LayerNorm on output

## Input/Output  
- target: (B, L, D)
- reference_points: (B, L, 1, 4)
- value: tuple of tensors per level
- output: (B, L, D)

Key: Residual connections between all sub-layers.
