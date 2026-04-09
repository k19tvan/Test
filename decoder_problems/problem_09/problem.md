# Problem 09 - Gate Mechanism

## Description
Adaptive feature fusion using learned sigmoid gating. Linearly combines two feature sets with learned weights that depend on their concatenation.

## Formula
$$\text{gate} = \sigma(W[\mathbf{x}_1; \mathbf{x}_2] + b)$$
$$\text{output} = \text{LN}(\text{gate}_1 \odot \mathbf{x}_1 + \text{gate}_2 \odot \mathbf{x}_2)$$

where sigmoid output is split into two gate vectors.

## Input/Output
- x1, x2: (B, L, D)
- output: (B, L, D) after gating and layer norm
