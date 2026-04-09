# Problem 02 Theory - Translate GT to Distribution Bins

## Core Definitions

**Ground Truth (GT)**: The true continuous value we want to regress (e.g., distance from left edge of image to left edge of object). Typically normalized to [0, reg_scale].

**Discretization**: The process of converting a continuous value into discrete bin indices with interpolation weights. This allows us to train with cross-entropy-style losses on distributions rather than smooth L1 losses.

**Linear Interpolation**: A method to distribute a continuous value across its two nearest discrete bins. If a value falls between bins i and i+1, we assign weights (1-α) to bin i and α to bin i+1, where α is the fractional part.

**Soft Labels**: Unlike hard labels (one value = 1, rest = 0), soft labels use interpolation weights to create smoother targets. This improves gradient flow and training stability.

## Variables and Shape Dictionary

| Variable | Shape | Dtype | Meaning |
|----------|-------|-------|---------|
| gt | (B,) | float32 | Ground truth continuous values |
| reg_max | scalar | int | Number of bins = reg_max + 1 |
| weighting_fn | (reg_max+1,) | float32 | Pre-computed weights from problem 01 |
| left_indices | (B,) | float32 | Index i where w(i) ≤ gt_value < w(i+1) |
| left_values | (B,) | float32 | Weight value at left_indices |
| right_values | (B,) | float32 | Weight value at left_indices + 1 |
| weight_left | (B,) | float32 | Interpolation weight for bin i, range [0,1] |
| weight_right | (B,) | float32 | Interpolation weight for bin i+1, range [0,1] |

## Main Equations (LaTeX)

### 1. Closest Left Bin via Cumulative Masking
$$\text{mask}_i = [\text{weights}(i) \leq \text{gt}]$$

$$\text{left\_indices} = \sum_{i=0}^{n} \text{mask}_i - 1$$

For each GT value, find how many weight values are ≤ it, then subtract 1 to get left bin index.

### 2. Clamp Indices to Valid Range
$$\text{clamped\_indices} = \text{clip}(\text{left\_indices}, 0, \text{reg\_max} - 1)$$

Ensures indices fall within [0, reg_max-1] to avoid out-of-bounds access.

### 3. Linear Interpolation Weights
$$\text{left\_value} = \text{weights}[\text{clamped\_indices}]$$

$$\text{right\_value} = \text{weights}[\text{clamped\_indices} + 1]$$

$$\text{alpha} = \frac{\text{gt} - \text{left\_value}}{\text{right\_value} - \text{left\_value}}$$

$$\text{weight\_right} = \text{clip}(\text{alpha}, 0, 1)$$

$$\text{weight\_left} = 1 - \text{weight\_right}$$

Compute fractional distance between left and right bin values.

## Step-by-Step Derivation or Computation Flow

1. **Reshape GT to 1D**
   - Input: gt with shape (B,)
   - Operation: Flatten to ensure it's 1D

2. **Compute Bin Boundaries**
   - Get weighting function values
   - These represent the bin positions in value space

3. **Find Left Bin Index**
   - Create mask where weights ≤ each GT value
   - Sum mask along dimension: counts how many bins are ≤ GT
   - Subtract 1 to get 0-indexed left bin
   
4. **Clamp to Valid Range**
   - Ensure indices stay in [0, reg_max-1]
   - Prevents index out of bounds errors

5. **Get Bin Values**
   - Retrieve weight values at left_idx and left_idx + 1
   - These are the boundaries for interpolation

6. **Compute Interpolation**
   - Calculate distance from left_value to GT_value
   - Normalize by distance between left_value and right_value
   - Result is fractional position (0 = left bin, 1 = right bin)

7. **Create Label Weights**
   - weight_right = normalized distance (alpha)
   - weight_left = 1 - alpha
   - These sum to 1.0 and represent soft labels

## Tensor Shape Flow (Input → Intermediate → Output)

```
Inputs:
  gt (B,)                           # Ground truth values
  weighting_fn (reg_max+1,)         # From problem 01

Processing:
  mask (B, reg_max+1)               # Boolean mask: weights <= gt for each element
  mask_sum (B,)                     # Sum of True values per element
  left_indices_raw (B,)             # Sum - 1 (pre-clamp)
  left_indices (B,)                 # Clamped to [0, reg_max-1]
  
  left_values (B,)                  # weights[left_indices]
  right_values (B,)                 # weights[left_indices+1]
  
  distance (B,)                     # gt - left_values
  bin_width (B,)                    # right_values - left_values
  alpha (B,)                        # distance / bin_width, clamped [0,1]

Outputs:
  weight_right (B,)                 # alpha (right bin weight)
  weight_left (B,)                  # 1 - alpha (left bin weight)
  indices (B,)                      # left_indices (for reference)
```

## Practical Interpretation

**Why Soft Labels Instead of Hard Labels?**
- Hard label: GT=2.3 → label=[0,0,1,0,0] (bin 2 only)
- Soft label: GT=2.3 → label=[0,0,0.7,0.3,0] (bins 2 and 3 weighted)
- Soft labels improve gradient flow and training stability
- The model learns smoother decision boundaries

**Linear Interpolation Choice**
- Smooth transition between bins
- Simple to compute (just division)
- Reflects linear assumption about value distribution
- Better than nearest-neighbor (harder training)

**Boundary Handling**
- GT values outside weighting function range → clipped to edge
- Ensures gradient flow even for outlier GT values
- Common in practice for robust training

**D-FINE Integration**
Once soft labels are computed:
1. Model outputs distribution for each coordinate
2. Compute cross-entropy between predicted distribution and soft labels
3. Use weighting function to scale loss per bin
4. Encourages model to predict peaked distributions near true values
