# Problem 06 Theory - Attention Weights Softmax

## Key Equation
$$\sigma(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Properties:
- Output in [0, 1]
- Sum to 1.0
- Differentiable with stable gradients via log-sum-exp trick

## Numerical Stability
Use: $\text{softmax}(x) = \text{softmax}(x - \max(x))$ to avoid overflow.
PyTorch's `F.softmax` implements this automatically.
