"""Reduce resident weight memory for the pillow shadow policy using PyTorch only."""

import torch
from torch import nn
from torch.nn import functional as F


class Int8WeightLinear(nn.Module):
    """Symmetric per-output-channel INT8 storage with on-demand BF16 weights."""

    def __init__(self, linear):
        super().__init__()
        weight = linear.weight.detach()
        if weight.device.type != 'cpu':
            raise ValueError('Quantize the merged weights on CPU before CUDA placement')
        weight = weight.float()
        low, high = torch.aminmax(weight, dim=1, keepdim=True)
        scale = (torch.maximum(-low, high) / 127).to(torch.bfloat16)
        if not torch.isfinite(scale).all():
            raise ValueError('Cannot quantize non-finite weights')
        scale.masked_fill_(scale == 0, 1)
        packed = (weight / scale.float()).round_().clamp_(-127, 127).to(torch.int8)
        self.register_buffer('packed', packed)
        self.register_buffer('scale', scale)
        self.register_buffer('bias', None if linear.bias is None else linear.bias.detach().to(torch.bfloat16))

    def forward(self, x):
        weight = self.packed.to(self.scale.dtype).mul_(self.scale)
        return F.linear(x, weight, self.bias)


def _quantize_linears(module):
    elements = 0
    for name, child in module.named_children():
        if type(child) is nn.Linear:
            packed = Int8WeightLinear(child)
            packed.train(child.training)
            setattr(module, name, packed)
            elements += packed.packed.numel()
        else:
            elements += _quantize_linears(child)
    return elements


def quantize_pillow_head(head):
    # Keep the state and action projections, the T5 embedding, CLIP and VAE in BF16.
    dit = _quantize_linears(head.model.blocks)
    text = _quantize_linears(head.text_encoder.blocks)
    if not dit or not text:
        raise ValueError('Expected merged DiT and T5 Transformer blocks for INT8 storage')
    print(f'Packed {dit:,} DiT and {text:,} T5 weight elements into INT8 on CPU', flush=True)
