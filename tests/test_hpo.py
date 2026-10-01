"""HPO tooling: config overrides and the sweep selection rules."""

import json

import pytest

from eval.select import choose, load_candidate, main
from training.train_lora import apply_overrides


def test_overrides_types_and_nesting():
    cfg = {"learning_rate": 2e-4, "lora": {"r": 16, "alpha": 32}}
    apply_overrides(
        cfg, ["learning_rate=1e-4", "lora.r=32", "lora.alpha=64", "seed=7",
              "output_dir=outputs/hpo/x", "lora.target_modules=[q_proj, v_proj]"]
    )  # fmt: skip
    assert cfg["learning_rate"] == pytest.approx(1e-4) and isinstance(cfg["learning_rate"], float)
    assert cfg["lora"] == {"r": 32, "alpha": 64, "target_modules": ["q_proj", "v_proj"]}
    assert cfg["seed"] == 7 and cfg["output_dir"] == "outputs/hpo/x"


def test_overrides_reject_bad_input():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])
    with pytest.raises(ValueError):
        apply_overrides({"lora": 3}, ["lora.r=8"])


def write(results, tag, ts, leak, over, items=100, dropped=0, outputs=50, truncated=0):
    m = {"leakage_chars": leak / 100, "over_redaction": over / 100,
         "llm_output": {"items": items, "dropped_items": dropped,
                        "outputs": outputs, "truncated": truncated}}  # fmt: skip
    (results / f"{tag}__{ts}.json").write_text(json.dumps({"metrics": m}))


def test_lowest_leakage_wins_outside_tie_band(tmp_path):
    write(tmp_path, "a", "g", 30.0, 40.0)
    write(tmp_path, "b", "g", 28.5, 50.0)  # 1.5 pt better: no tie
    cands = [load_candidate(tmp_path, t, ["g"], []) for t in "ab"]
    w, why = choose(cands, sanity=False)
    assert w.tag == "b" and "no other candidate" in why


def test_tie_broken_by_over_redaction_then_in_dist(tmp_path):
    write(tmp_path, "a", "g", 29.0, 35.0)
    write(tmp_path, "b", "g", 29.6, 31.0)  # within 1 pt, less over-redaction -> wins
    write(tmp_path, "c", "g", 29.9, 31.0)  # same over-redaction as b, worse in-dist
    for t, v in (("a", 1.0), ("b", 0.8), ("c", 0.9)):
        write(tmp_path, t, "dev", v, 0.5)
    cands = [load_candidate(tmp_path, t, ["g"], ["dev"]) for t in "abc"]
    w, why = choose(cands, sanity=False)
    assert w.tag == "b" and "3 candidates" in why


def test_primary_is_equal_weight_mean(tmp_path):
    write(tmp_path, "a", "s1", 10.0, 5.0)
    write(tmp_path, "a", "s2", 2.0, 1.0)
    c = load_candidate(tmp_path, "a", ["s1", "s2"], [])
    assert c.leak == pytest.approx(6.0) and c.over == pytest.approx(3.0)


def test_sanity_rule_and_fallback(tmp_path):
    write(tmp_path, "loopy", "g", 20.0, 30.0, items=100, dropped=30)  # 30% invented
    write(tmp_path, "clean", "g", 25.0, 30.0, items=100, dropped=2, outputs=100, truncated=1)
    cands = [load_candidate(tmp_path, t, ["g"], []) for t in ("loopy", "clean")]
    assert choose(cands, sanity=True)[0].tag == "clean"
    assert choose(cands, sanity=False)[0].tag == "loopy"
    only = [load_candidate(tmp_path, "loopy", ["g"], [])]
    w, why = choose(only, sanity=True)
    assert w.tag == "loopy" and "flagged" in why


def test_missing_results_and_report(tmp_path):
    write(tmp_path, "p1_a", "g", 30.0, 40.0)
    write(tmp_path, "p1_b", "dev", 1.0, 1.0)  # no primary result
    w = main(["--name", "t", "--prefix", "p1_", "--primary", "g", "--in-dist", "dev",
              "--results", str(tmp_path)])  # fmt: skip
    assert w.tag == "p1_a"
    md = (tmp_path / "sweeps" / "t.md").read_text()
    assert "p1_a **(winner)**" in md
    assert "| p1_b |" in md and md.rstrip().splitlines()[-3].endswith("| g |")  # missing primary


def test_bootstrap_ratios_and_interval():
    import numpy as np

    from eval.bootstrap import bootstrap, ci, ratios

    counts = np.array([[10, 2, 12, 4], [20, 0, 18, 0], [10, 5, 6, 1]], dtype=float)
    leak, over = ratios(counts)  # sums: gold 40, leaked 7, pred 36, over 5
    assert leak == pytest.approx(7 / 40) and over == pytest.approx(5 / 36)
    lo, hi = ci(bootstrap(counts, 500)[:, 0])
    assert lo <= leak <= hi
    same = np.array([[10, 1, 10, 0]] * 5, dtype=float)
    assert ci(bootstrap(same, 200)[:, 0]) == pytest.approx((0.1, 0.1))


def _write_runs(results, tag, perfect):
    from eval.run_eval import load_testset

    _, examples = load_testset("fixture", None)
    (results / "runs").mkdir(parents=True, exist_ok=True)
    with open(results / "runs" / f"{tag}__fixture.jsonl", "w") as f:
        for ex in examples:
            pred = [s.to_dict() for s in ex.spans] if perfect else []
            f.write(json.dumps({"id": ex.id, "pred": pred}) + "\n")


def test_bootstrap_cli_seeds_and_paired(tmp_path):
    from eval.bootstrap import main as boot

    _write_runs(tmp_path, "good", perfect=True)
    _write_runs(tmp_path, "none", perfect=False)
    out = boot(["--testset", "fixture", "--tags", "good", "--n", "200", "--results", str(tmp_path)])
    assert out["leakage_pct"] == 0 and out["leakage_ci95_pct"] == [0, 0] and out["docs"] == 20
    pair = boot(["--testset", "fixture", "--tags", "good,none", "--paired", "--n", "200",
                 "--results", str(tmp_path)])  # fmt: skip
    assert pair["leakage_diff_pct"] == pytest.approx(-100) and pair["a_better_share"] == 1.0
    seeds = boot(["--testset", "fixture", "--tags", "good,none", "--n", "200",
                  "--results", str(tmp_path)])  # fmt: skip
    assert seeds["leakage_pct"] == pytest.approx(50) and seeds["seed_spread_pct"] == 100
