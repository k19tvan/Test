# Problem 03 Theory - Inverse Sigmoid Transformation

## Core Definitions

**Sigmoid Function**: $\sigma(x) = \frac{1}{1 + e^{-x}}$, maps $(-\infty, \infty) \to (0, 1)$

**Inverse Sigmoid (Logit)**: $\sigma^{-1}(x) = \log\left(\frac{x}{1-x}\right)$, maps $(0, 1) \to (-\infty, \infty)$

**Numerical Stability**: Techniques to avoid computational issues like log(0), division by zero, or overflow when implementing mathematical functions on finite-precision computers.

**Epsilon (ε)**: A small positive constant used to clamp values away from problematic boundaries, preventing division by zero or log of zero.

## Variables and Shape Dictionary

| Variable | Shape | Dtype | Meaning |
|----------|-------|-------|---------|
| x_input | (B,) | float32 | Original sigmoid-normalized values in [0,1] |
| eps | scalar | float | Epsilon for clamping, typically 1e-5 |
| x_clamped | (B,) | float32 | Input clamped to [eps, 1-eps] |
| ratio | (B,) | float32 | Intermediate: x_clamped / (1 - x_clamped) |
| output | (B,) | float32 | Final logit values in (-∞, +∞) |

## Main Equations (LaTeX)

### 1. Standard Logit Formula
$$\text{logit}(x) = \log\left(\frac{x}{1-x}\right)$$

Inverse of sigmoid, used to transform predictions back to unconstrained space.

### 2. Clamped Version (Numerically Stable)
$$x_{\text{clamped}} = \text{clip}(x, \epsilon, 1-\epsilon)$$

$$\text{logit}(x) = \log\left(\frac{x_{\text{clamped}}}{1-x_{\text{clamped}}}\right)$$

Prevents log(0) by keeping arguments away from boundaries.

### 3. Properties of Inverse Sigmoid
$$\sigma(\text{logit}(x)) = x \quad \text{for } x \in (0,1)$$

$$\text{logit}(0.5) = 0$$

$$\text{logit}(x) + \text{logit}(1-x) = 0 \quad \text{(antisymmetry)}$$

## Step-by-Step Derivation or Computation Flow

1. **Start with Sigmoid Output**
   - Input: $x \in [0, 1]$, output of sigmoid function

2. **Derive Inverse**
   - Starting from: $y = \frac{1}{1 + e^{-z}}$
   - Invert: $z = \log\left(\frac{y}{1-y}\right)$

3. **Handle Numerical Issues**
   - Problem: Arguments can be 0 or 1, causing log(0) or division by zero
   - Solution: Clamp to $[\epsilon, 1-\epsilon]$ where $\epsilon > 0$ is small

4. **Compute Ratio and Log**
   - Compute: $r = \frac{x_{\text{clamped}}}{1-x_{\text{clamped}}}$
   - Apply log: $\text{result} = \log(r)$

5. **Special Cases**
   - $x = 0.5 \Rightarrow \frac{0.5}{0.5} = 1 \Rightarrow \log(1) = 0$
   - $x$ near 0 $\Rightarrow$ ratio $\approx \frac{\epsilon}{1-\epsilon} \Rightarrow$ large negative
   - $x$ near 1 $\Rightarrow$ ratio $\approx \frac{1-\epsilon}{\epsilon} \Rightarrow$ large positive

## Tensor Shape Flow (Input → Intermediate → Output)

```
Input:
  x (B,) in [0, 1]

Clamping:
  x_clamped (B,) in [eps, 1-eps]

Intermediate:
  numerator (B,) = x_clamped
  denominator (B,) = 1 - x_clamped
  ratio (B,) = numerator / denominator

Output:
  result (B,) = log(ratio) in (-∞, +∞)
```

## Practical Interpretation

**Why Inverse Sigmoid?**
In detection models like D-FINE:
- References points are stored in [0,1]: normalized image coordinates
- When predicting offsets/corrections, we add deltas in unconstrained space then apply sigmoid
- This requires converting predictions: $x_{\text{next}} = \sigma(x_{\text{prev}} + \Delta)$
- Rearranging: $\Delta = \sigma^{-1}(x_{\text{next}}) - x_{\text{prev}}$

**Numerical Stability Critical**
- Without clamping: $\log(0)$ is undefined, causes NaN in gradients
- With clamping: $\log(\epsilon/(1-\epsilon))$ is finite but small
- Prevents gradient explosion during backpropagation

**Range of Output**
- $\text{logit}(\epsilon) \approx \log(\epsilon/(1-\epsilon)) \approx -10.6$ for $\epsilon=1e-5$
- $\text{logit}(0.5) = 0$  
- $\text{logit}(1-\epsilon) \approx 10.6$
- Total range: [-10.6, 10.6] for $\epsilon=1e-5$

**Symmetry Property**
$\text{logit}(x) = -\text{logit}(1-x)$: The function is antisymmetric around 0.5, making it ideal for balanced predictions
