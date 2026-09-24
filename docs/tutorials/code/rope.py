"""Minimal RoPE: adjacent pairs and split-half pairs, for [..., L, d] tensors.

Rotate Q and K after splitting attention heads; leave V unchanged.
Use the checkpoint's layout and base. The default base=10000 is illustrative.
Angles and rotation are computed in FP32, then cast back to the input dtype.
"""

import torch


def rope_angles(x, start_pos=0, base=10000.0):
    L, d = x.shape[-2:]
    assert d > 0 and d % 2 == 0
    freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    pos = torch.arange(start_pos, start_pos + L, device=x.device, dtype=torch.float32)
    angle = pos[:, None] * freq[None, :]  # [L, d/2]; broadcasts over B and H
    return angle.cos(), angle.sin()


def rope_interleaved(x, start_pos=0, base=10000.0):
    cos, sin = rope_angles(x, start_pos, base)
    a, b = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


def rope_split_half(x, start_pos=0, base=10000.0):
    cos, sin = rope_angles(x, start_pos, base)
    a, b = x.float().chunk(2, dim=-1)
    out = torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return out.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 8)
    # Reorder adjacent coordinates into the split-half layout.
    to_half = lambda t: torch.cat((t[..., 0::2], t[..., 1::2]), dim=-1)
    torch.testing.assert_close(to_half(rope_interleaved(x)), rope_split_half(to_half(x)))
    for rope in (rope_interleaved, rope_split_half):
        out = rope(x)
        torch.testing.assert_close(out[..., 0, :], x[..., 0, :])
        torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))
    print("RoPE checks passed: layout equivalence, position zero, and norm preservation.")
