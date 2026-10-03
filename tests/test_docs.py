"""Every relative link in the README, model card and docs/ points at a file that exists."""

import re
from pathlib import Path

import pytest

DOCS = [Path("README.md"), Path("MODEL_CARD.md"), *sorted(Path("docs").glob("*.md"))]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize("doc", DOCS, ids=str)
def test_relative_links_resolve(doc):
    missing = []
    for target in LINK.findall(doc.read_text(encoding="utf-8")):
        if re.match(r"[a-z]+://|mailto:", target):
            continue
        if not (doc.parent / target).exists():
            missing.append(target)
    assert not missing, f"{doc}: broken links {missing}"
