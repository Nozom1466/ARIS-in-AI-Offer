"""Numerical and documentation checks for the interview RoPE implementations."""

import inspect
from pathlib import Path
import re
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "tutorials" / "code"))
from rope import rope_angles, rope_interleaved, rope_split_half


def to_half(x):
    return torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)


class RoPETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, 5, 8)

    def test_complex_reference(self):
        # Independent complex-number reference, in float64.
        for d in (2, 8, 64):
            x = torch.randn(2, 3, 5, d)
            for offset in (0, 17, 1024):
                for base in (10000.0, 500000.0):
                    freq = torch.tensor([base ** (-2 * j / d) for j in range(d // 2)], dtype=torch.float64)
                    angle = torch.arange(offset, offset + 5, dtype=torch.float64)[:, None] * freq
                    z = torch.view_as_complex(x.double().reshape(2, 3, 5, d // 2, 2))
                    ref = torch.view_as_real(z * torch.polar(torch.ones_like(angle), angle)).flatten(-2).float()
                    torch.testing.assert_close(rope_interleaved(x, offset, base), ref, atol=3e-4, rtol=3e-4)
                    torch.testing.assert_close(rope_split_half(to_half(x), offset, base), to_half(ref), atol=3e-4, rtol=3e-4)

    def test_layout_equivalence_and_not_interchangeable(self):
        x = self.x
        torch.testing.assert_close(to_half(rope_interleaved(x, 7)), rope_split_half(to_half(x), 7))
        self.assertFalse(torch.allclose(rope_interleaved(x, 7), rope_split_half(x, 7)))

    def test_identity_norm_and_dtype(self):
        for rope in (rope_interleaved, rope_split_half):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = self.x.to(dtype)
                y = rope(x)
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, dtype)
                torch.testing.assert_close(y[..., 0, :], x[..., 0, :])
                torch.testing.assert_close(y.float().norm(dim=-1), x.float().norm(dim=-1), atol=0.02, rtol=0.005)

    def test_full_sequence_matches_cached_positions(self):
        for rope in (rope_interleaved, rope_split_half):
            full = rope(self.x, start_pos=7)
            pieces = [rope(self.x[..., i:i + 1, :], start_pos=7 + i) for i in range(5)]
            torch.testing.assert_close(full, torch.cat(pieces, dim=-2))

    def test_relative_position_scores(self):
        q = self.x
        k = torch.randn_like(q)
        for rope in (rope_interleaved, rope_split_half):
            scores = rope(q) @ rope(k).transpose(-1, -2)
            shifted = rope(q, 19) @ rope(k, 19).transpose(-1, -2)
            torch.testing.assert_close(scores, shifted, atol=1e-5, rtol=1e-5)

    def test_gradients(self):
        for rope in (rope_interleaved, rope_split_half):
            x = self.x.clone().requires_grad_()
            rope(x, 3).square().sum().backward()
            torch.testing.assert_close(x.grad, 2 * x)

    def test_odd_head_dimension_rejected(self):
        for rope in (rope_interleaved, rope_split_half):
            with self.assertRaises(AssertionError):
                rope(torch.zeros(2, 3, 5, 7))

    def test_bilingual_snippets_match_executable_source(self):
        for name in ("attention_tutorial.md", "attention_tutorial_en.md"):
            text = (ROOT / "docs" / "tutorials" / name).read_text(encoding="utf-8")
            blocks = re.findall(r"```python[^\n]*\n(.*?)\n```", text, re.S)
            block = next(b for b in blocks if "def rope_interleaved(" in b)
            for fn in (rope_angles, rope_interleaved, rope_split_half):
                self.assertIn(inspect.getsource(fn).strip(), block)


if __name__ == "__main__":
    unittest.main()
