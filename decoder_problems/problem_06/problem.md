# Problem 06 - Attention Weights Softmax

## Description
Attention weights are raw logits produced by a linear layer. These must be normalized to form a probability distribution using softmax. This ensures weights sum to 1.0 and are in [0,1] range for interpolating features.

## Mathematical Definition
$$\text{attention\_weights} = \text{softmax}(\text{logits}, \text{dim}=-1) = \frac{\exp(\text{logits}[i])}{\sum_j \exp(\text{logits}[j])}$$

## Input/Output
- Input logits: (B, L, num_heads, total_points), dtype float32
- Output weights: (B, L, num_heads, total_points), dtype float32
- Property: Sum to 1.0 along last dimension

## Solution
Use `F.softmax(logits, dim=-1)` for numerical stability via PyTorch's built-in implementation.
