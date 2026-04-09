"""
Problem 01: Checker for Weighting Function

Tests correctness of weighting function implementation.
"""

import torch
import sys

# Import the solution (user will fill this in)
try:
    from solution import weighting_function
except ImportError:
    print("ERROR: solution.py not found")
    sys.exit(1)


def test_basic_shape_correctness():
    """Test that output shape is correct."""
    reg_max = 16
    up = torch.tensor([2.0])
    reg_scale = torch.tensor([1.0])
    
    output = weighting_function(reg_max, up, reg_scale)
    
    assert isinstance(output, torch.Tensor), "Output must be torch.Tensor"
    assert output.shape == (reg_max + 1,), f"Expected shape ({reg_max + 1},), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected dtype float32, got {output.dtype}"
    print("✓ Shape correctness test passed")


def test_center_bias():
    """Test that center bins have higher weights than edges."""
    reg_max = 16
    up = torch.tensor([2.0])
    reg_scale = torch.tensor([1.0])
    
    output = weighting_function(reg_max, up, reg_scale)
    
    center_idx = reg_max // 2
    edge_idx_left = 0
    edge_idx_right = reg_max
    
    center_weight = output[center_idx].item()
    edge_left_weight = output[edge_idx_left].item()
    edge_right_weight = output[edge_idx_right].item()
    
    assert center_weight > edge_left_weight, \
        f"Center weight {center_weight} should be > left edge {edge_left_weight}"
    assert center_weight > edge_right_weight, \
        f"Center weight {center_weight} should be > right edge {edge_right_weight}"
    print("✓ Center bias test passed")


def test_all_positive():
    """Test that all weights are non-negative."""
    reg_max = 16
    up = torch.tensor([2.0])
    reg_scale = torch.tensor([1.0])
    
    output = weighting_function(reg_max, up, reg_scale)
    
    assert (output >= 0).all(), "All weights must be non-negative"
    print("✓ Non-negativity test passed")


def test_symmetry():
    """Test that weights are symmetric around center."""
    reg_max = 16
    up = torch.tensor([2.0])
    reg_scale = torch.tensor([1.0])
    
    output = weighting_function(reg_max, up, reg_scale)
    
    # Check symmetry: w(center-k) ≈ w(center+k)
    center = reg_max / 2.0
    for k in range(1, 5):
        left_idx = int(center - k)
        right_idx = int(center + k)
        left_weight = output[left_idx].item()
        right_weight = output[right_idx].item()
        assert abs(left_weight - right_weight) < 1e-5, \
            f"Weights should be symmetric: left[{k}]={left_weight} vs right[{k}]={right_weight}"
    print("✓ Symmetry test passed")


def test_edge_case_small_reg_max():
    """Test with minimum reg_max value."""
    reg_max = 1
    up = torch.tensor([2.0])
    reg_scale = torch.tensor([1.0])
    
    output = weighting_function(reg_max, up, reg_scale)
    
    assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"
    assert len(output) == reg_max + 1
    print("✓ Edge case (reg_max=1) test passed")


def test_scaling_effect():
    """Test that reg_scale multiplies output correctly."""
    reg_max = 16
    up = torch.tensor([2.0])
    
    output1 = weighting_function(reg_max, up, torch.tensor([1.0]))
    output2 = weighting_function(reg_max, up, torch.tensor([2.0]))
    
    # output2 should be approximately 2x output1
    ratio = (output2 / (output1 + 1e-8)).abs().mean().item()
    assert abs(ratio - 2.0) < 0.1, f"Expected ~2x scaling, got {ratio}x"
    print("✓ Scaling effect test passed")


def run_all_checks():
    """Run all test cases."""
    tests = [
        test_basic_shape_correctness,
        test_center_bias,
        test_all_positive,
        test_symmetry,
        test_edge_case_small_reg_max,
        test_scaling_effect,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            return False
    
    return True


if __name__ == "__main__":
    if run_all_checks():
        print("\n✓ All Problem 01 checks passed")
    else:
        print("\n✗ Some checks failed")
        sys.exit(1)
