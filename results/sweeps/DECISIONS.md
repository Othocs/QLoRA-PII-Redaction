# HPO plan: decisions log

Branch `hyperparameter_optimization`. Each decision is written down before the next phase starts.

## Phase 1: learning rate on M3 data (2026-10-02)
- **Rule:** lowest `gretel_dev` leakage; a difference under 1.0 pt is a tie, broken by over-redaction.
- **Result:** gretel_dev leakage is flat at 28.7–29.8% over η 5e-5 to 6e-4; dev leakage is 1.6% at 5e-5 and 0.6–0.7% from 2e-4 up.
- **Decision: η\* = 4e-4.** Four LRs tie; 4e-4 has the lowest over-redaction (`phase1_lr.md`).

## Phase 2: M4 (+ Gretel EN, 30k) vs M3 (2026-10-02)
- **Rule:** lowest `val_ood` leakage (mean of support_desk_val and val_in_region), with the 1-pt tie band.
- **Result:** `val_ood` is saturated. All candidates leak 0.9–1.5%, and M4@4e-4 (0.91) vs M3@4e-4 (0.99) is noise. The literal tie-break on over-redaction would pick M3.
- **Decision (user, with the team's rationale): M4 @ 4e-4 goes forward.** When scaling capacity, a tie is broken in favour of the broader data mix (higher ranks need more data and diversity). In phase 4, a tie is broken in favour of lower over-redaction.
- **Fix for saturation:** `support_desk_hard` was added (100 deliberately difficult messages). `val_ood` is now the mean leakage of support_desk_val, support_desk_hard and val_in_region.

## Support-desk labels (2026-10-02)
- **Decision (user):** the LLM-drafted labels in support_desk_val, support_desk_hard and support_desk_300 are accepted as correct without a human spot-check. They are validated by `check_support_desk.py` only. Reports on these sets carry this caveat.

## Phase 3: rank 32/64 (rules set before running)
- **Grid:** r ∈ {32, 64}, α = 2r, η ∈ {1e-4, 1.4e-4, 2e-4} (`phase3_rank.txt`). Baseline: M4 r=16 @ 4e-4.
- **Budget cut (user, 2026-10-02, mid-run): r=64 dropped.** Only r=32 is run, at all three LRs (Phase 3 ≈ $4 instead of ≈ $8). r=64 is run later only if r=32 earns verdict A or B. If r=32 doesn't beat r=16, a further doubling is unlikely to pay off.
- **Phase 4 is not automatic.** The user decides whether to run it after seeing the phase 3 results.
- **Best LR per rank:** chosen with the general rule (val_ood leakage, then the tie band, then over-redaction).
- **Decision matrix against the baseline**, on the val_ood mean, with paired document bootstraps:
  - **A, adopt:** relative leakage reduction of at least 20%, with the CI excluding 0.
  - **B, adopt:** leakage within ±0.15 pt, and over-redaction down at least 1.5 pt with the CI excluding 0.
  - **C, keep r=16:** parity or noise.
  - **D, reject:** leakage worse by more than 0.15 pt, or more invented values (+2 pt) or token-limit hits (+1 pt).
- **Parsimony:** r=16 is the default. If two ranks qualify, the smaller wins. Seed variance is checked in phase 4 (3 seeds).
- **Also reported:** leakage on the hard slice, invented values, token-limit hits, and p95 latency on support_desk_val.

### Phase 3 result (2026-10-02)
- **Runs:** r=32 at η ∈ {1e-4, 1.4e-4, 2e-4} (34.9M trainable parameters, ~128 min per run on an A40); r=16 baseline re-scored on the hardened `val_ood`. Cost about $3.80.
- **Hardened `val_ood`:** baseline mean 4.04% (support_desk_val 1.1%, support_desk_hard 10.3%, val_in_region 0.7%). The hard slice falls in the team's 5–10% target band; the 3-set mean is just below it.
- **Best r=32:** η = 1.4e-4 (4.24%). All four runs are within the 1-pt tie band; the table's "winner" row is the best-LR pick among them, not the phase decision.
- **Decision matrix:**
  - **Verdict C for every r=32 run.** Leakage Δ is +0.20 to +0.52 pt, with every 95% CI spanning 0. Over-redaction Δ is −0.16 to +0.31 pt, also within noise.
  - Hard-slice leakage: r=32 is 10.2–10.9% vs 10.3% at r=16.
  - No change in invented values (≤0.2%) or token-limit hits (0%).
  - Latency p95 is unchanged: 9.4–10.0 s vs 9.2 s per 1k chars, for single requests on support_desk_val.
- **Rule fix made while scoring:** D now also needs the leakage CI to exclude 0, symmetric with A. Before the fix, the point estimates alone labelled these runs D; either way r=16 is kept.
- **Decision: keep r=16 (M4 @ 4e-4).** Per the gate set before running, r=64 is not run: r=32 earned neither A nor B.
- **What still leaks:** the remaining errors look like data and convention gaps, not capacity. On support_desk_hard (baseline), the worst labels by character leakage are:
  - TELEPHONENUM 43% (spelled-out and split numbers);
  - TITLE 45%;
  - SOCIALNUM 28% (3 spans);
  - CREDITCARDNUMBER 20% (5 spans).

  STREET/BUILDINGNUM have strict F1 0 but no leakage: the model folds the house number into STREET, a convention difference.
- **Phase 4:** awaiting the user's decision.
