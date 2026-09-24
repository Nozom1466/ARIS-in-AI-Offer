#!/usr/bin/env python3
"""Build Markdown pages, code viewers and the index without altering review records."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "tools/tutorials_render_manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Render manifest is empty")
    # Code viewers must exist before the renderer rewrites code links.
    subprocess.run([sys.executable, "tools/build_code_pages.py"], cwd=ROOT, check=True)
    for output, entry in manifest.items():
        source = entry["md"]
        for path in (source, output):
            if not (ROOT / path).resolve().is_relative_to(ROOT / "docs"):
                raise ValueError(f"Manifest path must stay inside docs/: {path}")
        cmd = [sys.executable, "tools/render_html.py", source, "--out", output, "--template", "academic"]
        for key in ("title", "subtitle", "eyebrow", "author", "lang"):
            if entry.get(key):
                cmd.extend(["--" + key, entry[key]])
        if entry.get("blog_mode"):
            cmd.append("--blog-mode")
        subprocess.run(cmd, cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tools/build_index.py"], cwd=ROOT, check=True)
    print(f"Built {len(manifest)} Markdown pages plus the index and code viewers.")


if __name__ == "__main__":
    main()
