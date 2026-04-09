"""Problem 08: Checker"""
import torch
import sys
try:
    from solution import Integral
except ImportError:
    print("ERROR"); sys.exit(1)

def test():
    integral = Integral(16)
    dist = torch.randn(2, 300, 4*17)
    weights = torch.randn(17)
    out = integral(dist, weights)
    assert out.shape == (2, 300, 4)
    assert not torch.isnan(out).any()
    print("✓ All tests passed")

if __name__ == "__main__":
    try:
        test()
        print("\n✓ All Problem 08 checks passed")
    except Exception as e:
        print(f"✗ {e}"); sys.exit(1)
