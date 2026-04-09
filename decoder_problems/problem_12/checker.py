"""Problem 12: Checker"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solution import MSDeformableAttention
except ImportError as e:
    print(f"ERROR: {e}"); sys.exit(1)

def test():
    attn = MSDeformableAttention(256, 8, 2, 4)
    query = torch.randn(2, 300, 256)
    ref_pts = torch.rand(2, 300, 2, 4)
    value = (torch.randn(2, 64*64, 256), torch.randn(2, 32*32, 256))
    spatial_shapes = [[64, 64], [32, 32]]
    out = attn(query, ref_pts, value, spatial_shapes)
    # Should output same shape as query
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == query.shape, f"Expected {query.shape}, got {out.shape}"
    print("✓ All tests passed")

if __name__ == "__main__":
    try:
        test()
        print("\n✓ All Problem 12 checks passed")
    except Exception as e:
        print(f"✗ {e}"); sys.exit(1)
