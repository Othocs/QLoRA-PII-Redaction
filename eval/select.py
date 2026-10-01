"""Pick the winner of a sweep with the team's selection rules (HPO plan).

    python -m eval.select --name phase1_lr --tags p1_lr5e-5,p1_lr1e-4,... \
        --primary gretel_dev --in-dist dev
    python -m eval.select --name phase2_m4 --prefix p2_ \
        --primary support_desk_val,val_in_region --in-dist dev,nemotron_dev,gretel_dev --sanity

Rules:
  1. Primary metric = leakage on the primary sets; several sets are averaged with equal
     weight (a pooled character count would let the larger set decide alone).
  2. Candidates within TIE (1.0) percentage points of the best are tied; ties are broken
     by over-redaction on the primary sets, then by mean leakage on the in-distribution sets.
  3. --sanity (phase 2 on): a candidate whose share of dropped (invented) values exceeds 5%
     or whose share of outputs hitting the token limit exceeds 2% on the primary sets is
     discarded. If every candidate fails, the best is still reported, flagged.

Writes results/sweeps/<name>.md (a table + the decision) and prints it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

TIE = 1.0  # percentage points
MAX_DROPPED = 0.05
MAX_TRUNCATED = 0.02


@dataclass
class Candidate:
    tag: str
    leak: float | None  # mean leakage % over primary sets
    over: float | None  # mean over-redaction % over primary sets
    in_dist: float | None  # mean leakage % over in-distribution sets
    dropped: float | None  # share of returned values not found in the text
    truncated: float | None  # share of outputs that hit the token limit
    missing: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)

    @property
    def sane(self) -> bool:
        return (self.dropped is None or self.dropped <= MAX_DROPPED) and (
            self.truncated is None or self.truncated <= MAX_TRUNCATED
        )


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def load_candidate(results: Path, tag: str, primary: list[str], in_dist: list[str]) -> Candidate:
    leaks, overs, ins, missing = [], [], [], []
    items = dropped = outputs = truncated = 0
    for ts in primary:
        p = results / f"{tag}__{ts}.json"
        if not p.exists():
            missing.append(ts)
            continue
        m = json.loads(p.read_text())["metrics"]
        leaks.append(100 * m["leakage_chars"])
        overs.append(100 * m["over_redaction"])
        o = m.get("llm_output") or {}
        items += o.get("items", 0)
        dropped += o.get("dropped_items", 0)
        outputs += o.get("outputs", 0)
        truncated += o.get("truncated", 0)
    for ts in in_dist:
        p = results / f"{tag}__{ts}.json"
        if p.exists():
            ins.append(100 * json.loads(p.read_text())["metrics"]["leakage_chars"])
        else:
            missing.append(ts)
    info = Path("outputs/hpo") / tag / "run_info.json"
    overrides = json.loads(info.read_text()).get("overrides", []) if info.exists() else []
    return Candidate(
        tag,
        _mean(leaks) if len(leaks) == len(primary) else None,
        _mean(overs) if len(overs) == len(primary) else None,
        _mean(ins),
        dropped / items if items else None,
        truncated / outputs if outputs else None,
        missing,
        overrides,
    )


def choose(cands: list[Candidate], sanity: bool) -> tuple[Candidate | None, str]:
    scored = [c for c in cands if c.leak is not None]
    if not scored:
        return None, "no candidate has results on every primary set"
    pool, note = scored, ""
    if sanity:
        sane = [c for c in scored if c.sane]
        if sane:
            pool = sane
        else:
            note = " Every candidate failed the sanity rule; best reported anyway (flagged)."
    best = min(c.leak for c in pool)
    tied = [c for c in pool if c.leak - best < TIE]
    winner = min(tied, key=lambda c: (c.over, c.in_dist if c.in_dist is not None else float("inf")))
    if len(tied) > 1:
        why = (
            f"{len(tied)} candidates within {TIE} pt of the best leakage ({best:.2f}%); "
            f"tie broken by over-redaction, then in-distribution leakage."
        )
    else:
        why = f"lowest primary leakage ({best:.2f}%), no other candidate within {TIE} pt."
    return winner, why + note


def report(name: str, cands: list[Candidate], winner, why, primary, in_dist, sanity) -> str:
    def f(x, pct=False):
        if x is None:
            return ""
        return f"{100 * x:.1f}" if pct else f"{x:.2f}"

    lines = [
        f"## Sweep: {name}",
        "",
        f"Primary: leakage on {' + '.join(primary)} (mean). Tie band {TIE} pt; tie-breaks: "
        f"over-redaction, then leakage on {' + '.join(in_dist) or '—'}."
        + (
            f" Sanity rule on: dropped ≤ {100 * MAX_DROPPED:.0f}%, "
            f"token-limit ≤ {100 * MAX_TRUNCATED:.0f}%."
            if sanity
            else ""
        ),
        "",
        "| Run | Overrides | Primary leakage (%) | Over-redaction (%) | In-dist leakage (%) "
        "| Dropped values (%) | Hit token limit (%) | Sane | Missing |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for c in sorted(cands, key=lambda c: (c.leak is None, c.leak or 0)):
        mark = " **(winner)**" if winner is not None and c.tag == winner.tag else ""
        lines.append(
            f"| {c.tag}{mark} | {' '.join(c.overrides)} | {f(c.leak)} | {f(c.over)} | "
            f"{f(c.in_dist)} | {f(c.dropped, True)} | {f(c.truncated, True)} | "
            f"{'yes' if c.sane else 'no'} | {', '.join(c.missing)} |"
        )
    lines += ["", f"**Decision:** {winner.tag if winner else 'none'}: {why}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Candidate | None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tags", help="comma-separated result tags")
    g.add_argument("--prefix", help="every tag in results/ starting with this prefix")
    ap.add_argument("--primary", required=True, help="comma-separated selection sets")
    ap.add_argument("--in-dist", default="", help="comma-separated in-distribution sets")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)

    results = Path(args.results)
    primary = [s for s in args.primary.split(",") if s]
    in_dist = [s for s in args.in_dist.split(",") if s]
    if args.tags:
        tags = [t for t in args.tags.split(",") if t]
    else:
        tags = sorted({p.name.split("__")[0] for p in results.glob(f"{args.prefix}*__*.json")})
    cands = [load_candidate(results, t, primary, in_dist) for t in tags]
    winner, why = choose(cands, args.sanity)
    md = report(args.name, cands, winner, why, primary, in_dist, args.sanity)
    (results / "sweeps").mkdir(parents=True, exist_ok=True)
    (results / "sweeps" / f"{args.name}.md").write_text(md)
    print(md)
    return winner


if __name__ == "__main__":
    main()
