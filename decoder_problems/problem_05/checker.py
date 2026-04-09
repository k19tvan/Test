"""Problem 05: Sampling Offsets - Checker"""
import torch
import sys
try:
    from solution import init_sampling_offsets
except ImportError:
    print("ERROR: solution.py not found")
    sys.exit(1)

def test_layer_creation():
    layer = init_sampling_offsets(8, 3, [4,4,4], 256)
    assert isinstance(layer, torch.nn.Linear)
    assert layer.in_features == 256
    assert layer.out_features == 192  # 8*12*2
    print("✓ Layer creation test passed")

def test_bias_not_zero():
    layer = init_sampling_offsets(8, 3, [4,4,4], 256)
    assert (layer.bias != 0).any(), "Bias should be non-zero after initialization"
    print("✓ Bias initialization test passed")

def test_weight_zero():
    layer = init_sampling_offsets(8, 3, [4,4,4], 256)
    assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
    print("✓ Weight zero test passed")

if __name__ == "__main__":
    try:
        test_layer_creation()
        test_bias_not_zero()
        test_weight_zero()
        print("\n✓ All Problem 05 checks passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)
