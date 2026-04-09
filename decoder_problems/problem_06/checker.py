"""Problem 06: Checker"""
import torch
import sys
try:
    from solution import attention_weights_softmax
except ImportError:
    print("ERROR"); sys.exit(1)

def test():
    logits = torch.randn(2, 10, 8, 4)
    weights = attention_weights_softmax(logits)
    assert weights.shape == logits.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2,10,8), atol=1e-5)
    assert (weights >= 0).all() and (weights <= 1).all()
    print("✓ All tests passed")

if __name__ == "__main__":
    try:
        test()
        print("\n✓ All Problem 06 checks passed")
    except Exception as e:
        print(f"✗ {e}"); sys.exit(1)
