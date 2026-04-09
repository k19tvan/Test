# Problem 13 Theory

Layer composition:
$$ \mathbf{h}_1 = \text{LN}(\mathbf{x} + \text{MHA}(\mathbf{x}, \mathbf{x}, \mathbf{x})) $$
$$ \mathbf{h}_2 = \text{Gate}(\mathbf{h}_1, \text{DeformAttn}(\mathbf{h}_1, \mathbf{V})) $$
$$ \mathbf{y} = \text{LN}(\mathbf{h}_2 + \text{FFN}(\mathbf{h}_2)) $$

Benefits: Stable gradients via residuals, multi-scale context via deformable attention.
