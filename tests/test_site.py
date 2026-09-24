"""Cross-platform rendering and fail-closed checks for the publishing workflow."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from render_html import _repo_relative, sha256_of
from verify_reviews import hash_matches, sha256_file


class SiteTest(unittest.TestCase):
    def test_source_hash_agrees_with_renderer_for_lf_and_crlf(self):
        text = "# RoPE\n\n位置与旋转\n"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            for newline in ("\n", "\r\n"):
                path = Path(tmp) / "source.md"
                path.write_text(text, encoding="utf-8", newline=newline)
                self.assertEqual(sha256_file(path), expected)
                self.assertEqual(sha256_file(path), sha256_of(path.read_text(encoding="utf-8")))

    def test_source_paths_use_url_separators(self):
        path = Path.cwd() / "docs" / "tutorials" / "attention_tutorial.md"
        self.assertEqual(_repo_relative(path), "docs/tutorials/attention_tutorial.md")

    def test_source_edits_still_invalidate_hash(self):
        original = sha256_of("original")
        changed = sha256_of("changed")
        self.assertFalse(hash_matches(original, changed))
        self.assertFalse(hash_matches(original[:1], original))
        self.assertTrue(hash_matches(original[:16], original))


if __name__ == "__main__":
    unittest.main()
