"""Recall-first union of spans from the LLM and the validators.

Anything either detector flags is redacted. Overlapping spans, whatever their labels, merge
into one span covering their outer extent, so a partial hit never leaves a fragment of a
value in clear text. The merged span takes the label of the longest validator span in the
group when a validator fired (validators are checked: Luhn, mod-97, libphonenumber), else
the label of the longest model span. Adjacent spans (GIVENNAME then SURNAME) stay separate.
"""

from __future__ import annotations

from pii_gateway.spans import Span


def recall_first_union(text: str, *groups: list[Span]) -> list[Span]:
    spans = sorted((s for g in groups for s in g), key=lambda s: (s.start, -s.end))
    clusters: list[list[Span]] = []
    for s in spans:
        if clusters and s.start < max(x.end for x in clusters[-1]):
            clusters[-1].append(s)
        else:
            clusters.append([s])
    out = []
    for c in clusters:
        start, end = min(s.start for s in c), max(s.end for s in c)
        validators = [s for s in c if s.source == "validator"]
        pick = max(validators or c, key=lambda s: s.end - s.start)
        sources = {s.source for s in c}
        source = sources.pop() if len(sources) == 1 else "merged"
        score = max((s.score for s in c if s.score is not None), default=None)
        out.append(Span(start, end, pick.label, text[start:end], source, score))
    return out
