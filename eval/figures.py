"""Regenerate the figures in docs/figures/ from the saved results (no model, no GPU).

    python -m eval.figures            # or: make figures

Reads results/phase4/{phase4,final_coverage}.json (seed-averaged M5 + baselines with CIs) and
the per-run results/*.json files, and writes six PNGs:
  1_leak_vs_over.png     leakage vs over-redaction per test set, M5 against the baselines
  2_coverage.png         M5 vs the best baseline on every test set (in/out of distribution)
  3_data_ablation.png    M0 -> M5: what the training data changed
  4_hpo.png              phase 1 learning-rate sweep and phase 3 rank comparison
  5_m5_gate.png          milestone 1 gate: M4 vs M5 on the fresh and hard support sets
  6_per_label.png        M5 leakage by label and test set (seed mean)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("results")
OUT = Path("docs/figures")
SYSTEMS = {
    "m5": ("M5", "#1f6feb"),
    "openmed": ("OpenMed", "#8957e5"),
    "gliner_nvidia": ("GLiNER-PII", "#2da44e"),
    "presidio": ("Presidio", "#bf8700"),
}
SETS = {  # name -> (title, out-of-distribution for M5?)
    "support_desk_300": ("Support desk (LLM-drafted)", True),
    "abcd": ("ABCD (real human chats)", True),
    "tab": ("TAB (real court cases)", True),
    "test_holdout_regions": ("OpenPII held-out region", True),
    "nemotron": ("Nemotron-PII test", False),
    "gretel_en": ("Gretel EN test", False),
}
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})  # fmt: skip


def load_sets() -> dict:
    sets = {}
    for name in ("phase4", "final_coverage"):
        p = R / "phase4" / f"{name}.json"
        if p.exists():
            sets.update(json.loads(p.read_text())["sets"])
    return sets


def metric(tag: str, ts: str, key: str = "leakage_chars") -> float | None:
    p = R / f"{tag}__{ts}.json"
    return 100 * json.loads(p.read_text())["metrics"][key] if p.exists() else None


def rows(entry: dict) -> dict[str, dict]:
    out = {"m5": entry["m5_model"]}
    out.update({b: s for b, s in entry["baselines"].items() if b in SYSTEMS})
    return out


def fig_leak_vs_over(sets: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, (ts, (title, ood)) in zip(axes.flat, SETS.items(), strict=True):
        for sys_, s in rows(sets[ts]).items():
            label, color = SYSTEMS[sys_]
            xerr = [[s["over"] - s["over_ci"][0]], [s["over_ci"][1] - s["over"]]]
            yerr = [[s["leak"] - s["leak_ci"][0]], [s["leak_ci"][1] - s["leak"]]]
            ax.errorbar(s["over"], s["leak"], xerr=xerr, yerr=yerr, fmt="o", color=color,
                        ms=7 if sys_ == "m5" else 5, capsize=2, label=label)  # fmt: skip
        ax.set_title(f"{title}{'' if ood else '  (in-dist. for M5)'}", fontsize=9)
        ax.set_xlabel("over-redaction (%)")
        ax.set_ylabel("leakage (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Leakage vs over-redaction (lower-left is better); 95% CIs, M5 = 3-seed mean",
                 y=0.99, fontsize=10)  # fmt: skip
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4,
               frameon=False)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "1_leak_vs_over.png")
    plt.close(fig)


def fig_coverage(sets: dict) -> None:
    names = list(SETS)
    m5 = [sets[t]["m5_model"]["leak"] for t in names]
    ci = [sets[t]["m5_model"]["leak_ci"] for t in names]
    best = [min((s["leak"], b) for b, s in sets[t]["baselines"].items() if b in SYSTEMS)
            for t in names]  # fmt: skip
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(x - 0.2, m5, 0.4, color=SYSTEMS["m5"][1], label="M5 (3-seed mean)",
           yerr=[[m - c[0] for m, c in zip(m5, ci, strict=True)],
                 [c[1] - m for m, c in zip(m5, ci, strict=True)]], capsize=3)  # fmt: skip
    ax.bar(x + 0.2, [b[0] for b in best], 0.4, color="#8c959f", label="best baseline")
    for i, (v, name) in enumerate(best):
        ax.text(x[i] + 0.2, v * 1.12, SYSTEMS[name][0], ha="center", fontsize=7, rotation=90,
                va="bottom")  # fmt: skip
    ax.set_xticks(x, [SETS[t][0].replace(" (", "\n(") + ("" if SETS[t][1] else "\n[in-dist.]")
                      for t in names], fontsize=7.5)  # fmt: skip
    ax.set_ylabel("leakage (%, log scale)")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 60)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("M5 vs the best baseline on each final test set (lower is better)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "2_coverage.png")
    plt.close(fig)


def fig_data_ablation() -> None:
    models = [("m0_openpii10k", "M0\nOpenPII 10k"), ("m1_clean10k", "M1\ncleaned"),
              ("m2_mix10k", "M2\n+Nemotron 5k"), ("m3_mix20k", "M3\n+Nemotron 10k"),
              ("p2_m4_lr4e-4", "M4\n+Gretel 10k"), ("m5_targeted", "M5\n+targeted 2k")]  # fmt: skip
    series = [("gretel_dev", "Gretel dev"), ("test_holdout_regions", "OpenPII held-out region"),
              ("tab", "TAB")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for ts, label in series:
        ys = [metric(t, ts) for t, _ in models]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ax.plot(xs, [ys[i] for i in xs], "o-", label=label)
    ax.set_xticks(range(len(models)), [m[1] for m in models], fontsize=7.5)
    ax.set_ylabel("leakage (%), seed 13")
    ax.legend(frameon=False)
    ax.set_title("What the training data changed (M4 was not scored on the held-out region or TAB)",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "3_data_ablation.png")
    plt.close(fig)


def fig_hpo() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    lrs = ["5e-5", "1e-4", "2e-4", "4e-4", "6e-4"]
    a.plot(range(5), [metric(f"p1_lr{lr}", "gretel_dev") for lr in lrs], "o-", label="leakage")
    a.plot(range(5), [metric(f"p1_lr{lr}", "gretel_dev", "over_redaction") for lr in lrs], "s--",
           label="over-redaction")  # fmt: skip
    a.set_xticks(range(5), lrs)
    a.set_xlabel("learning rate")
    a.set_ylabel("% on Gretel dev")
    a.set_title("Phase 1: learning rate (M3 data)", fontsize=9)
    a.legend(frameon=False)
    runs = [("p2_m4_lr4e-4", "r16\n4e-4"), ("p3_r32_lr1e-4", "r32\n1e-4"),
            ("p3_r32_lr1.4e-4", "r32\n1.4e-4"), ("p3_r32_lr2e-4", "r32\n2e-4")]  # fmt: skip
    sets3 = ["support_desk_val", "support_desk_hard", "val_in_region"]
    vals = [np.mean([metric(t, s) for s in sets3]) for t, _ in runs]
    b.bar(range(4), vals, color=["#1f6feb"] + ["#8c959f"] * 3)
    b.set_xticks(range(4), [r[1] for r in runs])
    b.set_ylabel("val_ood leakage (%), mean of 3 sets")
    b.set_title("Phase 3: rank 32 vs 16 (all within noise)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "4_hpo.png")
    plt.close(fig)


def fig_gate() -> None:
    sets_ = [("support_desk_fresh", "support_desk_fresh\n(gate, unseen)"),
             ("support_desk_hard", "support_desk_hard\n(diagnostic)")]  # fmt: skip
    m4 = [metric("p2_m4_lr4e-4", s) for s, _ in sets_]
    m5 = [metric("m5_targeted", s) for s, _ in sets_]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(x - 0.2, m4, 0.4, color="#8c959f", label="M4")
    ax.bar(x + 0.2, m5, 0.4, color="#1f6feb", label="M5 (+2k targeted)")
    ax.axhline(5, color="#cf222e", lw=1, ls=":", label="gate: < 5%")
    for i in range(2):
        ax.text(x[i] - 0.2, m4[i] + 0.4, f"{m4[i]:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, m5[i] + 0.4, f"{m5[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, [s[1] for s in sets_])
    ax.set_ylabel("leakage (%)")
    ax.legend(frameon=False)
    ax.set_title("Milestone 1 gate: targeted data (seed 13)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "5_m5_gate.png")
    plt.close(fig)


def fig_per_label(sets: dict) -> None:
    labels = ["GIVENNAME", "SURNAME", "TITLE", "EMAIL", "TELEPHONENUM", "STREET", "BUILDINGNUM",
              "CITY", "ZIPCODE", "DATE", "AGE", "CREDITCARDNUMBER", "SOCIALNUM",
              "IDCARDNUM"]  # fmt: skip
    cols = [t for t in SETS if t != "tab"]  # TAB uses its own label scheme
    grid = np.full((len(labels), len(cols)), np.nan)
    for j, t in enumerate(cols):
        per = sets[t].get("m5_per_label_leak", {})
        for i, lab in enumerate(labels):
            if lab in per:
                grid[i, j] = per[lab]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.clip(grid, 0, 30), cmap="Reds", vmin=0, vmax=30, aspect="auto")
    for i in range(len(labels)):
        for j in range(len(cols)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] > 18 else "black")  # fmt: skip
    ax.set_xticks(range(len(cols)), [textwrap.fill(SETS[t][0], 14) for t in cols], fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    fig.colorbar(im, ax=ax, label="leakage (%), clipped at 30")
    ax.set_title("M5 leakage by label (3-seed mean); blank = label absent", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "6_per_label.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = load_sets()
    fig_leak_vs_over(sets)
    fig_coverage(sets)
    fig_data_ablation()
    fig_hpo()
    fig_gate()
    fig_per_label(sets)
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
