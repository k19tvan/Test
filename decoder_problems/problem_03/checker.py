"""
Problem 03: Checker for Inverse Sigmoid

Tests correctness of inverse sigmoid implementation.
"""

import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solution import inverse_sigmoid
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


def test_shape_preservation():
    """Test that output shape matches input shape."""
    x = torch.tensor([0.1, 0.5, 0.9])
    y = inverse_sigmoid(x)
    assert y.shape == x.shape, f"Output shape {y.shape} != input shape {x.shape}"
    print("✓ Shape preservation test passed")


def test_midpoint_is_zero():
    """Test that logit(0.5) = 0."""
    x = torch.tensor([0.5])
    y = inverse_sigmoid(x)
    assert torch.allclose(y, torch.tensor([0.0]), atol=1e-4), \
        f"logit(0.5) should be 0, got {y.item()}"
    print("✓ Midpoint test passed")


def test_symmetry():
    """Test antisymmetry: logit(x) = -logit(1-x)."""
    x_vals = torch.tensor([0.2, 0.3, 0.7, 0.8])
    y = inverse_sigmoid(x_vals)
    y_complement = inverse_sigmoid(1.0 - x_vals)
    
    # logit(x) should equal -logit(1-x)
    assert torch.allclose(y, -y_complement, atol=1e-4), \
        f"logit should be antisymmetric: {y} != {-y_complement}"
    print("✓ Symmetry test passed")


def test_no_nans():
    """Test that no NaN values are produced."""
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    y = inverse_sigmoid(x)
    assert not torch.isnan(y).any(), f"Output contains NaN: {y}"
    assert not torch.isinf(y).any(), f"Output contains Inf: {y}"
    print("✓ No NaN/Inf test passed")


def test_boundary_values():
    """Test behavior at boundary values."""
    x = torch.tensor([[0.0, 1.0]])
    y = inverse_sigmoid(x, eps=1e-5)
    
    # Should be approximately symmetric around 0
    assert torch.allclose(y[0, 0], -y[0, 1], atol=0.1), \
        f"Boundary values should be symmetric: {y[0, 0]} vs {y[0, 1]}"
    print("✓ Boundary values test passed")


def test_inverse_property():
    """Test that sigmoid(inverse_sigmoid(x)) ≈ x for valid x."""
    x_original = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    y = inverse_sigmoid(x_original)
    x_reconstructed = torch.sigmoid(y)
    
    assert torch.allclose(x_original, x_reconstructed, atol=1e-4), \
        f"Reconstruction error too high: {(x_original - x_reconstructed).abs().max()}"
    print("✓ Inverse property test passed")


def test_batch_processing():
    """Test with different batch sizes."""
    for batch_size in [1, 2, 8, 16]:
        x = torch.rand(batch_size)  # Random values in [0, 1]
        y = inverse_sigmoid(x)
        assert y.shape == (batch_size,), f"Batch size mismatch for B={batch_size}"
        assert not torch.isnan(y).any(), f"NaN in batch of size {batch_size}"
    print("✓ Batch processing test passed")


def run_all_checks():
    """Run all test cases."""
    tests = [
        test_shape_preservation,
        test_midpoint_is_zero,
        test_symmetry,
        test_no_nans,
        test_boundary_values,
        test_inverse_property,
        test_batch_processing,
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
        print("\n✓ All Problem 03 checks passed")
    else:
        print("\n✗ Some checks failed")
        sys.exit(1)
