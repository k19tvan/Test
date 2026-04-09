# Problem 08 - Integral Computation

## Description  
Convert distribution predictions back to bounding box values via weighted integral. The distribution over bins is converted to a single continuous value using the precomputed weighting function.

## Formula
$$\text{value} = \sum_i \text{softmax}(\text{distribution})_i \times \text{weight}(i)$$

## Input/Output
- distribution: (B, L, 4*(reg_max+1))
- weight_proj: (reg_max+1,)
- output: (B, L, 4)

Each of 4 coordinates scaled independently.
