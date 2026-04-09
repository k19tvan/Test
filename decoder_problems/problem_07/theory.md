# Problem 07 Theory - MLP Forward Pass

## Core Definitions
**Multi-Layer Perceptron**: A sequence of linear transformations with non-linear activations: $y = \sigma(W_n \sigma(W_{n-1} \cdots \sigma(W_1 x + b_1) \cdots + b_{n-1}) + b_n)$ where no activation on output layer.

**Non-linearity**: Functions like ReLU, GELU, Sigmoid that enable learning complex patterns. Without them, stacked linear layers collapse to single linear transformation.

**Residual Design Alternative**: Can add skip connections, but basic MLP is feedforward-only.

## Main Equations (LaTeX)

### 1. Layer-wise Forward
$$h_0 = x$$
$$h_i = \sigma(W_i h_{i-1} + b_i) \quad \text{for } i = 1, \ldots, n-1$$
$$y = W_n h_{n-1} + b_n$$

### 2. ReLU Activation
$$\text{ReLU}(x) = \max(0, x)$$

### 3. Shape Transformations
$$x \in \mathbb{R}^{B \times L \times D_{\text{in}}} \to y \in \mathbb{R}^{B \times L \times D_{\text{out}}}$$

## Step-by-Step

1. Reshape input for linear operations
2. For each layer 0 to num_layers-2: Apply Linear + Activation
3. For last layer: Apply Linear only (no activation)
4. Reshape to original batch/sequence dimensions
