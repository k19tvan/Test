"""
Problem 02: Checker for Translate GT to Distribution

Tests correctness of GT translation implementation.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solution import translate_gt
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


def test_output_shapes():
    """Test that outputs have correct shapes."""
    gt = torch.tensor([0.5, 1.5, 2.5])
    reg_max = 5
    reg_scale = torch.tensor([4.0])
    up = torch.tensor([2.0])
    
    indices, weight_left, weight_right = translate_gt(gt, reg_max, reg_scale, up)
    
    B = gt.shape[0]
    assert indices.shape == (B,), f"indices shape {indices.shape} != ({B},)"
    assert weight_left.shape == (B,), f"weight_left shape {weight_left.shape} != ({B},)"
    assert weight_right.shape == (B,), f"weight_right shape {weight_right.shape} != ({B},)"
    print("✓ Output shapes test passed")


def test_weight_sum():
    """Test that weights sum to approximately 1.0."""
    gt = torch.tensor([0.5, 1.5, 2.5, 3.5])
    reg_max = 8
    reg_scale = torch.tensor([4.0])
    up = torch.tensor([2.0])
    
    indices, weight_left, weight_right = translate_gt(gt, reg_max, reg_scale, up)
    
    weight_sum = weight_left + weight_right
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), \
        f"Weights should sum to 1.0, got {weight_sum}"
    print("✓ Weight sum test passed")


def test_weights_in_valid_range():
    """Test that all weights are in [0, 1]."""
    gt = torch.tensor([0.0, 1.0, 2.0, 3.0])
    reg_max = 8
    reg_scale = torch.tensor([4.0])
    up = torch.tensor([2.0])
    
    indices, weight_left, weight_right = translate_gt(gt, reg_max, reg_scale, up)
    
    assert (weight_left >= 0).all() and (weight_left <= 1).all(), \
        f"weight_left values should be in [0,1], got min={weight_left.min()}, max={weight_left.max()}"
    assert (weight_right >= 0).all() and (weight_right <= 1).all(), \
        f"weight_right values should be in [0,1], got min={weight_right.min()}, max={weight_right.max()}"
    print("✓ Weight range test passed")


def test_indices_valid_range():
    """Test that indices are in valid range."""
    gt = torch.tensor([0.5, 1.5, 2.5])
    reg_max = 5
    reg_scale = torch.tensor([4.0])
    up = torch.tensor([2.0])
    
    indices, weight_left, weight_right = translate_gt(gt, reg_max, reg_scale, up)
    
    assert (indices >= 0).all() and (indices < reg_max).all(), \
        f"Indices should be in [0, {reg_max-1}], got min={indices.min()}, max={indices.max()}"
    print("✓ Indices range test passed")


def test_exact_bin_alignment():
    """Test behavior when GT exactly matches bin location."""
    reg_max = 8
    reg_scale = torch.tensor([4.0])
    up = torch.tensor([2.0])
    
    # Create a GT value that should get weight_left=1.0
    gt = torch.tensor([0.0])
    
    indices, weight_left, weight_right = translate_gt(gt, reg_max, reg_scale, up)
    
    # At exact bin location: should heavily favor one side
    assert weight_left[0].item() > 0.5 or weight_right[0].item() < 0.5, \
        "Should favor left weight at exact location"
    print("✓ Exact bin alignment test passed")


def test_batch_consistency():
    """Test consistent behavior for batched inputs."""
    reg_max = 8
    reg_scale = torch.tensor([4.0])
    up = torch.tensor([2.0])
    
    # Same GT value in batch
    gt = torch.tensor([1.5, 1.5, 1.5])
    
    indices, weight_left, weight_right = translate_gt(gt, reg_max, reg_scale, up)
    
    # All three should give same results
    assert torch.allclose(indices[0], indices[1]) and torch.allclose(indices[1], indices[2]), \
        "Same GT values should produce same indices"
    assert torch.allclose(weight_left[0], weight_left[1]) and torch.allclose(weight_left[1], weight_left[2]), \
        "Same GT values should produce same weights"
    print("✓ Batch consistency test passed")


def run_all_checks():
    """Run all test cases."""
    tests = [
        test_output_shapes,
        test_weight_sum,
        test_weights_in_valid_range,
        test_indices_valid_range,
        test_exact_bin_alignment,
        test_batch_consistency,
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
        print("\n✓ All Problem 02 checks passed")
    else:
        print("\n✗ Some checks failed")
        sys.exit(1)
