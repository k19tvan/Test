"""Problem 09: Checker"""
import torch
import sys
try:
    from solution import Gate
except ImportError:
    print("ERROR"); sys.exit(1)

def test():
    gate = Gate(256)
    x1 = torch.randn(2, 300, 256)
    x2 = torch.randn(2, 300, 256)
    out = gate(x1, x2)
    assert out.shape == x1.shape
    assert not torch.isnan(out).any()
    print("✓ All tests passed")

if __name__ == "__main__":
    try:
        test()
        print("\n✓ All Problem 09 checks passed")
    except Exception as e:
        print(f"✗ {e}"); sys.exit(1)
