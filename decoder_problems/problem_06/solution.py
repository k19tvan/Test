"""Problem 06: Solution"""
import torch
import torch.nn.functional as F

def attention_weights_softmax(logits):
    """Apply softmax normalization to attention logits."""
    return F.softmax(logits, dim=-1)
