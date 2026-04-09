"""Problem 10: Checker"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solution import LQE
except ImportError as e:
    print(f"ERROR: {e}"); sys.exit(1)

def test():
    lqe = LQE(4, 64, 2, 16)
    scores = torch.randn(2, 300, 80)
    # corners shape: (B, L, 4, reg_max+1) where reg_max=16, so 17 bins per coordinate
    corners = torch.randn(2, 300, 4, 17)
    out = lqe(scores, corners)
    assert out.shape == scores.shape, f"Expected {scores.shape}, got {out.shape}"
    print("✓ Output shape correct")
    print("✓ All tests passed")

if __name__ == "__main__":
    try:
        test()
        print("\n✓ All Problem 10 checks passed")
    except Exception as e:
        print(f"✗ {e}"); sys.exit(1)
