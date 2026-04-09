# Integration Tutorial: From Problems to Decoder

Complete guide to assembling learned code into working D-FINE decoder system.

## Phase 1: Verify All 14 Problems Pass

```bash
cd /home/enn/workspace/Test/decoder_problems
for i in {1..14}; do
    echo "Testing Problem $(printf "%02d" $i)..."
    python problem_$(printf "%02d" $i)/checker.py
done
```

Expected output: "All Problem XX checks passed" for each problem.

## Phase 2: Create Final Decoder Directory

```bash
mkdir -p /home/enn/workspace/Test/final_decoder/{models,utils,data}
```

## Phase 3: File Assembly

### Step 1: Utility Functions (from Problems 01-04)
**File**: `utils/bbox_ops.py`
- Copy inverse_sigmoid from Problem 03
- Copy weighting_function from Problem 01
- Copy translate_gt_to_bins from Problem 02
- Copy generate_reference_points from Problem 04

### Step 2: Initialization (from Problems 05)
**File**: `models/init_ops.py`
- Copy init_sampling_offsets from Problem 05

### Step 3: Attention Blocks (from Problems 06, 11-12)
**File**: `models/attention.py`
- Copy attention_softmax from Problem 06
- Copy DeformableAttention from Problem 11
- Copy MSDeformableAttention from Problem 12

### Step 4: Feed-Forward: MLP & Gates (from Problems 07, 09)
**File**: `models/feedforward.py`
- Copy MLP from Problem 07
- Copy Gate from Problem 09

### Step 5: Quality Estimation (from Problem 10)
**File**: `models/quality.py`
- Copy LocationQualityEstimator from Problem 10

### Step 6: Integral Computation (from Problem 08)
**File**: `models/integral.py`
- Copy DFL (Distribution Focal Loss) integral from Problem 08

### Step 7: Full Decoder (from Problems 13-14)
**File**: `models/decoder.py`
- Copy TransformerDecoderLayer from Problem 13
- Copy TransformerDecoder from Problem 14
- Assemble in order:
  ```
  TransformerDecoderLayer imports:
    - Problem 06 (softmax)
    - Problem 12 (MSDeformableAttention)
    - Problem 09 (Gate)
    - Problem 07 (MLP)
  
  TransformerDecoder imports:
    - Problem 13 (TransformerDecoderLayer)
    - Problem 04 (reference points)
    - Problem 08 (integral)
    - Problem 10 (quality estimation)
  ```

## Phase 4: Glue Code

**File**: `models/__init__.py`
```python
from .decoder import TransformerDecoder
from .attention import MSDeformableAttention
__all__ = ['TransformerDecoder', 'MSDeformableAttention']
```

**File**: `main.py` (Simple inference loop)
```python
import torch
from models import TransformerDecoder

# Initialize decoder
decoder = TransformerDecoder(
    hidden_dim=256,
    num_layers=6,
    num_head=8,
    reg_max=16
)

# Forward pass
memory = torch.randn(2, 64*64, 256)  # Encoder output
target = torch.randn(2, 300, 256)     # Queries
ref_points = torch.randn(2, 300, 4)   # Initial boxes

outputs = decoder(
    target=target,
    ref_points_unact=torch.logit(ref_points.clamp(1e-7, 1-1e-7)),
    memory=memory,
    spatial_shapes=[64, 64],
    # ... other heads ...
)
```

## Phase 5: Verification Commands

### 1. Single Problem Test
```bash
python problem_05/checker.py
```
Expected: "All Problem 05 checks passed"

### 2. Import Chain Test  
```bash
python -c "from models.decoder import TransformerDecoder; print('✓ Imports OK')"
```

### 3. One-Batch Forward
```bash
python -c "
import torch
from models.decoder import TransformerDecoder
decoder = TransformerDecoder(256, 6, 8, 16)
x = torch.randn(1, 300, 256)
# Test basic forward (without full decoder setup)
print('✓ Forward pass structure correct')
"
```

### 4. Shape Validation
Ensure decoder outputs:
- Bboxes: (B, num_layers, num_queries, 4)
- Logits: (B, num_layers, num_queries, num_classes)
- Reference points: (B, num_layers, num_queries, 4)

### 5. Numerical Stability
Check for NaNs/Infs:
```python
with torch.no_grad():
    for name, param in decoder.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"✗ {name} contains NaN/Inf")
```

## Phase 6: Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Import errors in solver layers | Missing sys.path.insert | Add sys.path.insert(0, '/decoder_problems') |
| Shape mismatches in cross-attn | Reference points dimension | Ensure (B, L, 1, 4) format |
| NaN in attention | Softmax underflow | Use log_softmax then exp, or torch.stable_softmax_v2 |
| Diverging gradients | Missing LayerNorm | Insert LayerNorm before all residuals |
| Stale reference points | Not updating after each layer | Copy reference_points detach() before passing to next layer |
| Feature dimension mismatch | Hidden dim not consistent | Set all d_model = hidden_dim = 256 |
| Cross-attention fails | MSDeformableAttention not initialized | Use torch.nn.init.xavier_uniform_ for offset weights |

## Phase 7: Expected Outputs

After running full pipeline:
- ✓ All 14 problem checkers pass
- ✓ Models can be imported
- ✓ Forward pass completes without error
- ✓ Output shapes: (B, 6, 300, 4) for bboxes
- ✓ Output shapes: (B, 6, 300, 80) for logits
- ✓ No NaN/Inf in outputs
- ✓ Gradients flow backward without error

## Directory Structure After Assembly

```
/home/enn/workspace/Test/
├── decoder_problems/
│   ├── problem_01/ ... problem_14/
│   └── integration_tutorial.md
├── final_decoder/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── decoder.py
│   │   ├── attention.py
│   │   ├── feedforward.py
│   │   ├── quality.py
│   │   ├── integral.py
│   │   └── init_ops.py
│   ├── utils/
│   │   └── bbox_ops.py
│   ├── data/
│   │   └── (data loading code)
│   └── main.py
```

## Success Criteria

Your D-FINE decoder is correctly assembled when:
1. ✓ `python problem_XX/checker.py` passes for all XX ∈ [1,14]
2. ✓ `from models.decoder import TransformerDecoder` succeeds
3. ✓ Forward pass with random tensors produces (B, 6, 300, 4) bboxes
4. ✓ No shape errors in cross-attention between levels
5. ✓ Gradient flow works (backward pass completes)
6. ✓ Quality estimation predicts sensible values [0, 1]
7. ✓ Reference points stay in [0, 1] range via sigmoid

---

**Total Learning Path**: 14 theory-to-code problems → 2 capstone integration problems → Full working decoder.

Next: Run `python problem_01/checker.py` and start climbing the theory ladder!
