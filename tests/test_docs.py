"""Every relative link in the README, model card and docs/ points at a file that exists, and
every #anchor into a markdown file matches one of its headings (GitHub's slug rules)."""

import re
from pathlib import Path

import pytest

DOCS = [Path("README.md"), Path("MODEL_CARD.md"), *sorted(Path("docs").glob("*.md"))]
LINK = re.compile(r"\]\(([^)\s]+)\)|(?:src|href)=\"([^\"]+)\"")


def slugs(md: Path) -> set[str]:
    """GitHub heading anchors: lowercase, drop punctuation, spaces to hyphens."""
    out, fence = set(), False
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fence = not fence
        elif not fence and (m := re.match(r"#{1,6} (.+)", line)):
            out.add(re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-"))
    return out


@pytest.mark.parametrize("doc", DOCS, ids=str)
def test_relative_links_resolve(doc):
    missing = []
    for md_target, html_target in LINK.findall(doc.read_text(encoding="utf-8")):
        target = md_target or html_target
        if re.match(r"[a-z]+://|mailto:", target):
            continue
        path, _, anchor = target.partition("#")
        file = doc.parent / path if path else doc
        bad_anchor = file.exists() and anchor and file.suffix == ".md" and anchor not in slugs(file)
        if not file.exists() or bad_anchor:
            missing.append(target)
    assert not missing, f"{doc}: broken links {missing}"


def test_slugs_follow_github_rules(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("## 4.1 First model (weeks 1–3)\n```\n# not a heading\n```\n"
                  "## Appendix B. Reproducing\n")  # fmt: skip
    assert slugs(md) == {"41-first-model-weeks-13", "appendix-b-reproducing"}
