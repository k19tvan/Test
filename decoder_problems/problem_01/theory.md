# Problem 01 Theory - Weighting Function Generation

## Core Definitions

**Distribution-based Regression**: Instead of predicting a single continuous value, we predict a probability distribution over discrete bins. This allows the model to express uncertainty and multi-modal predictions.

**Weighting Function W(n)**: A function that assigns importance scores to each bin. Higher weights correspond to bins whose predictions should contribute more to the loss, guiding the model toward confident central predictions.

**Non-uniform Weights**: Unlike uniform weighting (all bins equally important), non-uniform weights emphasize certain bins over others. In D-FINE, we emphasize bins near the center because they represent confident predictions.

## Variables and Shape Dictionary

| Variable | Shape | Dtype | Meaning |
|----------|-------|-------|---------|
| reg_max | scalar | int | Total number of discrete bins = reg_max + 1 |
| up | (1,) | float32 | Decay rate multiplier (upsampling factor) |
| reg_scale | (1,) | float32 | Final scaling coefficient |
| center | scalar | float | Position of highest weight = reg_max / 2 |
| i | scalar | int | Bin index in range [0, reg_max] |
| distance | (reg_max+1,) | float32 | Absolute distance from center for each bin |
| output | (reg_max+1,) | float32 | Final weight for each bin |

## Main Equations (LaTeX)

### 1. Distance from Center
$$d_i = |i - \text{center}|$$

where $\text{center} = \frac{\text{reg\_max}}{2}$ and $i \in [0, \text{reg\_max}]$

### 2. Logarithmic Decay Formula
$$w(i) = \frac{1}{1 + \text{up} \cdot d_i}$$

This creates inverse-proportional decay: weights decrease as distance from center increases.

### 3. Scaled Output
$$\text{output}(i) = w(i) \cdot \text{reg\_scale}$$

The reg_scale parameter allows fine-tuning the importance without recomputing the decay curve.

## Step-by-Step Derivation or Computation Flow

1. **Generate Bin Indices**
   - Create index array: $\mathbf{i} = [0, 1, 2, \ldots, \text{reg\_max}]$
   - Shape: $(reg\_max + 1,)$

2. **Compute Center Position**
   - $\text{center} = \text{reg\_max} / 2$
   - This is the midpoint of all bins

3. **Calculate Distance from Center**
   - For each bin $i$: $d_i = |i - \text{center}|$
   - Distance is 0 at center, increases toward edges
   - Vector form: $\mathbf{d} = |\mathbf{i} - \text{center}|$

4. **Apply Decay Formula**
   - Compute decay term: $1 + \text{up} \cdot \mathbf{d}$
   - Invert to get weights: $\mathbf{w} = \frac{1}{1 + \text{up} \cdot \mathbf{d}}$
   - Higher up value → steeper decay toward edges

5. **Apply Scaling**
   - Final output: $\text{output} = \mathbf{w} \cdot \text{reg\_scale}$

## Tensor Shape Flow (Input → Intermediate → Output)

```
Input Processing:
  up (1,)            → Extract scalar value
  reg_scale (1,)     → Extract scalar value
  reg_max (int)      → Use as count parameter

Computation:
  indices (reg_max+1,)         ← Create 0 to reg_max
  center (scalar)              ← Compute reg_max / 2
  distance (reg_max+1,)        ← |indices - center|
  unscaled_weights (reg_max+1,) ← 1 / (1 + up * distance)
  
Output:
  weights (reg_max+1,)         ← unscaled_weights * reg_scale
```

## Practical Interpretation

**Why Center Bias?**
- Center bins (near reg_max/2) represent confident, well-calibrated predictions
- Edge bins are extreme predictions that may indicate model uncertainty
- Higher weight on center bins encourages the model to make centered, confident predictions

**Effect of up Parameter**
- `up = 1`: Gradual decay, edges still have ~33% weight
- `up = 2`: Steep decay, edges get ~20% weight  
- `up = 4`: Very steep decay, edges get ~11% weight
- Larger up → sharper focus on center predictions

**Why Linear Decay?**
- Inverse proportional function is computationally efficient (single division)
- Smooth decay without sharp discontinuities
- Numerically stable across wide range of reg_max values
- Aligns with human intuition: importance decreases linearly with distance

**D-FINE Application**
In D-FINE, after the model predicts a distribution over bins for each coordinate (left, top, right, bottom), this weighting function ensures that:
1. Predictions near the true value contribute more to loss
2. The model learns to concentrate probability mass around ground truth
3. Outlier predictions are downweighted, improving robustness
