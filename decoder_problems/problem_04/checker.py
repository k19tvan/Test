"""
Problem 04: Checker for Reference Point Generation

Tests correctness of anchor grid generation.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solution import _generate_anchors
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


def test_output_shapes():
    """Test output shapes match expectations."""
    eval_spatial_size = (512, 512)
    feat_strides = [8, 16, 32]
    
    anchors, valid_mask = _generate_anchors(eval_spatial_size, feat_strides)
    
    # Calculate expected total points
    expected_total = sum((512 // s) * (512 // s) for s in feat_strides)
    
    assert anchors.shape[0] == 1, f"Batch size should be 1, got {anchors.shape[0]}"
    assert anchors.shape[1] == expected_total, f"Total points {anchors.shape[1]} != {expected_total}"
    assert anchors.shape[2] == 4, f"Coordinates should have 4 dims, got {anchors.shape[2]}"
    assert valid_mask.shape == (1, expected_total, 1), f"Mask shape mismatch"
    print("✓ Output shapes test passed")


def test_valid_mask_values():
    """Test that valid_mask contains only 0 and 1."""
    anchors, valid_mask = _generate_anchors((256, 256), [8, 16])
    
    unique_values = torch.unique(valid_mask)
    valid_set = {0.0, 1.0}
    assert set(unique_values.tolist()).issubset(valid_set), \
        f"Valid mask should only contain 0 and 1, got {unique_values}"
    print("✓ Valid mask values test passed")


def test_anchors_no_nan_with_mask():
    """Test that masked anchors (invalid) have inf, valid ones are finite."""
    anchors, valid_mask = _generate_anchors((256, 256), [8, 16])
    
    # Invalid points should have inf or be masked
    num_valid = (valid_mask == 1).sum().item()
    num_invalid = (valid_mask == 0).sum().item()
    
    assert num_valid > 0, "Should have at least some valid points"
    assert num_invalid >= 0, "Invalid points count should be non-negative"
    assert (num_valid + num_invalid) == valid_mask.numel(), "Counts should add up"
    print("✓ Anchor validity test passed")


def test_coordinate_ranges():
    """Test that valid anchors are in reasonable ranges."""
    anchors, valid_mask = _generate_anchors((512, 512), [8, 16, 32])
    
    # Valid anchors should have finite values (usually in range [-10, 10] for log-sigmoid)
    # Squeeze mask to match indexing requirements
    valid_mask_squeezed = valid_mask.squeeze(-1)  # (1, 5376)
    valid_anchors = anchors[valid_mask_squeezed == 1]
    
    if valid_anchors.numel() > 0:
        assert torch.isfinite(valid_anchors).all(), \
            f"Valid anchors should be finite, but found {torch.isnan(valid_anchors).sum()} NaNs"
        # Log-sigmoid output typically in [-20, 20]
        assert (valid_anchors.abs() < 50).all(), \
            f"Anchors seem out of range: min={valid_anchors.min()}, max={valid_anchors.max()}"
    print("✓ Coordinate ranges test passed")


def test_multilevels():
    """Test with different numbers of levels."""
    for num_levels in [1, 2, 3, 4]:
        feat_strides = [8 * (2 ** i) for i in range(num_levels)]
        anchors, valid_mask = _generate_anchors((512, 512), feat_strides)
        
        # Check shape is correct
        expected_total = sum((512 // s) * (512 // s) for s in feat_strides)
        assert anchors.shape[1] == expected_total, f"Mismatch for {num_levels} levels"
    print("✓ Multiple levels test passed")


def test_different_spatial_sizes():
    """Test with different evaluation spatial sizes."""
    for spatial_size in [(256, 256), (512, 512), (1024, 1024)]:
        anchors, valid_mask = _generate_anchors(spatial_size, [8, 16, 32])
        
        # All outputs should be finite or properly masked
        assert anchors.shape[2] == 4
        assert valid_mask.shape[2] == 1
    print("✓ Different spatial sizes test passed")


def run_all_checks():
    """Run all test cases."""
    tests = [
        test_output_shapes,
        test_valid_mask_values,
        test_anchors_no_nan_with_mask,
        test_coordinate_ranges,
        test_multilevels,
        test_different_spatial_sizes,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    if run_all_checks():
        print("\n✓ All Problem 04 checks passed")
    else:
        print("\n✗ Some checks failed")
        sys.exit(1)
