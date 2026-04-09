# Problem 05 - MLP (Multi-Layer Perceptron) Forward Pass

## Description
A Multi-Layer Perceptron (MLP) is a sequence of fully-connected layers with non-linear activations. MLPs serve as building blocks throughout D-FINE for positional encoding projections, feature transformations, and prediction heads. Implementing a clean MLP with flexible layer count and activation functions is fundamental.

## Input Format
- **input_dim** (int): Input feature dimension
- **hidden_dim** (int): Dimension of hidden layers
- **output_dim** (int): Output feature dimension
- **num_layers** (int): Number of layers (num_layers ≥ 2)
- **x** (torch.Tensor): Input activations. Shape: (B, L, input_dim), dtype: float32
- **activation** (str): Activation function name, e.g., "relu", "gelu". Default: "relu"

## Output Format
- Shape: (B, L, output_dim)
- Dtype: float32
- Property: No activation applied to output layer

## Constraints
- Activation applied only to hidden layers (not output)
- All intermediate layers use hidden_dim
- Must support variable batch and sequence dimensions
- Output should not have NaN/Inf for normal inputs

## Example
Input: x shape (2, 300, 256), num_layers=2, hidden_dim=512
MLP: Linear(256→512) → ReLU → Linear(512→256)
Output: (2, 300, 256)

## Hints
- Create nn.ModuleList of Linear layers
- Use get_activation() utility for activation function
- Iterate through layers, applying activation to all but last
- Properly initialize weights with Xavier uniform
- See original code for complete parameter details

## Checker
Run with: `python problem_05/checker.py`
Expected output: `All Problem 05 checks passed`
