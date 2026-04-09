# Problem 04 Theory - Reference Point Generation

## Core Definitions

**Reference Points**: Spatial coordinates used as starting guesses for object locations in the decoder. Each point represents a potential object center in normalized image coordinates.

**Multi-scale Feature Pyramid**: A hierarchy of feature maps at different resolutions. Coarser levels (higher strides) capture semantic information; finer levels capture detail. Multiple scales help detect objects of various sizes.

**Normalized Coordinates**: Image coordinates scaled to [0, 1] where (0,0) is top-left and (1,1) is bottom-right. Independent of image size, enabling size-generalization.

**Log-Sigmoid Space**: The unconstrained representation of normalized coordinates, obtained by applying inverse-sigmoid. Enables gradient-free refinement through addition of deltas.

## Variables and Shape Dictionary

| Variable | Shape | Dtype | Meaning |
|----------|-------|-------|---------|
| eval_h, eval_w | scalar | int | Target evaluation spatial dimensions |
| stride | scalar | int | Downsampling factor for current level |
| level_h, level_w | scalar | int | Feature map dimensions at level |
| grid_x, grid_y | (level_h, level_w) | float | Coordinate grids |
| xy_normalized | (level_h*level_w, 2) | float | Normalized [0,1] coordinates |
| wh_scale | (level_h*level_w, 2) | float | [grid_size * 2^level, ...] |
| anchors_level | (level_h*level_w, 4) | float | [x,y,w,h] for this level |
| anchors_logsig | (level_h*level_w, 4) | float | Log-sigmoid transformed anchors |
| valid_mask | (level_h*level_w, 1) | bool | Valid for valid points |
| output | (1, total_points, 4) | float | Batched, stacked across levels |

## Main Equations (LaTeX)

### 1. Grid Generation per Level
For level $\ell$ with stride $s_\ell$:
$$\text{level\_h} = \frac{\text{eval\_h}}{s_\ell}, \quad \text{level\_w} = \frac{\text{eval\_w}}{s_\ell}$$

$$\text{grid\_x}, \text{grid\_y} = \text{meshgrid}(0, 1, \ldots, \text{level\_w}-1, 0, 1, \ldots, \text{level\_h}-1)$$

### 2. Normalization to [0, 1]
$$x_{\text{norm}} = \frac{\text{grid\_x} + 0.5}{\text{level\_w}}, \quad y_{\text{norm}} = \frac{\text{grid\_y} + 0.5}{\text{level\_h}}$$

The +0.5 places coordinates at pixel centers rather than corners.

### 3. Width and Height Computation
$$\text{grid\_size} = 0.05 \quad \text{(base scale factor)}$$

$$w = \text{grid\_size} \cdot 2^\ell, \quad h = \text{grid\_size} \cdot 2^\ell$$

Larger scales for coarser levels to maintain detection capability across scales.

### 4. Log-Sigmoid Transform
$$\text{anchor} = \log\left(\frac{\text{normalized}}{1 - \text{normalized}}\right)$$

Converts [0,1] coordinates to unconstrained space for refinement.

### 5. Validity Mask
$$\text{valid}(p) = \begin{cases} 1 & \text{if } \epsilon \leq p \leq 1-\epsilon \\ 0 & \text{otherwise} \end{cases}$$

Prevents gradients for boundary points prone to numerical instability.

## Step-by-Step Derivation or Computation Flow

1. **Initialize Storage**
   - List to accumulate anchors across levels
   - Track total points

2. **For Each Feature Level**
   - Compute level grid dimensions: H_L = eval_H / stride_L
   - Create 2D meshgrid of integer indices
   
3. **Normalize Coordinates**
   - Add 0.5 to place at pixel centers
   - Divide by level dimensions to map to [0,1]
   
4. **Add Width/Height**
   - Compute scale: base * 2^level
   - Create [wh, wh] arrays
   - Concatenate with xy to get [x,y,w,h]
   
5. **Log-Sigmoid Transform**
   - For each coordinate: log(x/(1-x))
   - Result is in unconstrained space (-∞, +∞)
   
6. **Create Validity Mask**
   - Check: eps <= coordinate <= 1-eps (before transform)
   - Mark invalid with 0, valid with 1
   - Replace invalid anchors with inf
   
7. **Stack and Reshape**
   - Concatenate all levels: (total_points, 4)
   - Add batch dimension: (1, total_points, 4)

## Tensor Shape Flow (Input → Intermediate → Output)

```
Input: eval_spatial_size, feat_strides, eps

Per Level:
  meshgrid: (level_h, level_w, 2)
  normalized: (level_h*level_w, 2)
  wh: (level_h*level_w, 2)
  xywh: (level_h*level_w, 4)
  valid_mask_level: (level_h*level_w, 1)
  anchors_level (transformed): (level_h*level_w, 4)

Across Levels:
  anchors_list: [Level 0 anchors, Level 1 anchors, ...]
  concatenated: (total_points, 4)
  
Output:
  anchors_batched: (1, total_points, 4)
  valid_mask_batched: (1, total_points, 1)
```

## Practical Interpretation

**Why Multi-scale Grids?**
- Small objects: Need fine grid (small stride, many points)
- Large objects: Need coarse grid (large stride, fewer points)
- Trade-off: Fine grids expensive; coarse grids miss details
- Solution: Use all scales, weight by detection size

**Why Normalized Coordinates?**
- Independent of image resolution
- Model generalizes to different input sizes
- Enables efficient batch processing
- Natural representation for regression targets

**Log-Sigmoid Space Motivation**
- Predictions in this space can be unrestricted (add any delta)
- After sigmoid, automatically clipped to [0,1]
- Enables natural refinement: $x_{i+1} = \sigma(x_i + \Delta)$
- Prevents explosion of values that could cause numerical issues

**Validity Mask Purpose**
- Edge points near 0 or 1 → log becomes large
- Large log → large gradients → unstable training
- Masking prevents gradient backflow to these areas
- Accepts small performance hit for training stability
