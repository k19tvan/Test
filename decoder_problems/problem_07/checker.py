"""Problem 07: MLP Forward Pass - Checker"""
import torch
import sys
try:
    from solution import MLP
except ImportError:
    print("ERROR: solution.py not found")
    sys.exit(1)

def test_mlp_forward():
    mlp = MLP(256, 512, 128, 3, "relu")
    x = torch.randn(2, 300, 256)
    y = mlp(x)
    assert y.shape == (2, 300, 128), f"Expected (2,300,128), got {y.shape}"
    assert not torch.isnan(y).any(), "Output contains NaN"
    print("✓ MLP forward test passed")

def test_output_size():
    mlp = MLP(64, 256, 32, 2, "gelu")
    x = torch.randn(4, 200, 64)
    y = mlp(x)
    assert y.shape[-1] == 32, f"Output dim should be 32, got {y.shape[-1]}"
    print("✓ Output size test passed")

if __name__ == "__main__":
    try:
        test_mlp_forward()
        test_output_size()
        print("\n✓ All Problem 07 checks passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)
