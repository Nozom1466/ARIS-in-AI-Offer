"""Shared deployment identity. Original author and external citations stay intact."""

import os

REPOSITORY = os.environ.get("ARIS_REPOSITORY", os.environ.get("GITHUB_REPOSITORY", "Nozom1466/ARIS-in-AI-Offer"))
OWNER, NAME = REPOSITORY.split("/", 1)
REPO = f"https://github.com/{REPOSITORY}"
SITE = os.environ.get("ARIS_SITE_URL", f"https://{OWNER.lower()}.github.io/{NAME}").rstrip("/")
SHORT = SITE  # This fork has no separate redirect domain.
SITE_LABEL = SITE.removeprefix("https://").removeprefix("http://")
