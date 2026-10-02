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

## Phase 3: rank 32/64 (rules set before running)
- **Grid:** r ∈ {32, 64}, α = 2r, η ∈ {1e-4, 1.4e-4, 2e-4} (`phase3_rank.txt`). Baseline: M4 r=16 @ 4e-4.
- **Best LR per rank:** chosen with the general rule (val_ood leakage, then the tie band, then over-redaction).
- **Decision matrix against the baseline**, on the val_ood mean, with paired document bootstraps:
  - **A, adopt:** relative leakage reduction of at least 20%, with the CI excluding 0.
  - **B, adopt:** leakage within ±0.15 pt, and over-redaction down at least 1.5 pt with the CI excluding 0.
  - **C, keep r=16:** parity or noise.
  - **D, reject:** leakage worse by more than 0.15 pt, or more invented values (+2 pt) or token-limit hits (+1 pt).
- **Parsimony:** r=16 is the default. If two ranks qualify, the smaller wins. Seed variance is checked in phase 4 (3 seeds).
- **Also reported:** leakage on the hard slice, invented values, token-limit hits, and p95 latency on support_desk_val.
