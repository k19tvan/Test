# Problem 11 Theory

Deformable attention learns offsets to sampling locations, enabling:
1. Adaptive focus based on object content
2. Multi-scale reasoning  
3. Gradient-based refinement

Formula:
$$\text{output} = \sum_p w_p \cdot \text{sample}(\mathbf{V}, \mathbf{x}_p + \Delta_p)$$

where sample uses bilinear interpolation.
