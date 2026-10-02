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

## Milestone 1: targeted failure-mode data (M5), rules set before generating data (2026-10-02)
- **Candidate.** M5 = M4's recipe (r=16, α=32, learning rate 4e-4, 1 epoch, seed 13) on `train_mix_32k`: `train_mix_30k` + 2,000 targeted synthetic messages generated with the DeepSeek API (`data/synthetic/`).
- **Gate set.** `support_desk_fresh`: 100 unseen hard messages, committed before any targeted data was generated. It was written from the pattern list by a different generator (Claude) than the training data.
- **Promotion gate, all on support_desk_fresh, M5 vs M4:**
  1. M5 character leakage < 5%;
  2. M5 leakage lower than M4's, with the paired document-bootstrap 95% CI excluding 0;
  3. no rise in over-redaction: point Δ ≤ +1.0 pt, and the CI not entirely above 0;
  4. no extra loops: invented values ≤ M4 + 2 pt, and token-limit hits ≤ M4 + 1 pt.

  If M4 is already under 5% on fresh, the gate is (2)–(4) only, flagged as weak.
- **Diagnostic, reported but not gating.** On support_desk_hard: how many of the spans M4 missed entirely does M5 recover, and the overall hard-slice leakage.
- **Guard rails.** These must hold or M5 is not promoted:
  - support_desk_val and val_in_region leakage within 1.0 pt of M4;
  - val_in_dist mean (dev, nemotron_dev, gretel_dev) within 1.0 pt;
  - sanity rule (invented ≤ 5%, token limit ≤ 2%).
- **Outcome.** A pass makes M5 the phase 4 candidate. A fail keeps M4 and records which patterns didn't move.

### Milestone 1 result (2026-10-02): M5 PASSES the gate
- **Run:** one A40, 137 min of training ($1.12), about $1.30 for the pod in total. Seed 13, 2,000 steps on `train_mix_32k`. All data checksums matched the local files.
- **Gate on support_desk_fresh**, M5 vs M4:

  | Criterion | M4 | M5 | Result |
  | --- | ---: | ---: | --- |
  | Leakage < 5% | 27.65% | **4.57%** | pass |
  | Leakage below M4, paired 95% CI excluding 0 | — | Δ −23.1 pt, CI [−32.5, −14.1] | pass |
  | Over-redaction Δ ≤ +1.0 pt, CI not entirely above 0 | 8.71% | 3.12%: Δ −5.6 pt, CI [−9.5, −2.4] | pass (lower) |
  | Loops: invented ≤ M4 + 2 pt, token limit ≤ M4 + 1 pt | 1/270 dropped, 0 truncated | 1/285 dropped, 0 truncated | pass |

- **Guard rails, all pass:**

  | Set | M4 leakage | M5 leakage | Δ | Limit |
  | --- | ---: | ---: | ---: | ---: |
  | support_desk_val | 1.12% | 1.58% | +0.46 pt | 1.0 pt |
  | val_in_region | 0.68% | 0.62% | −0.06 pt | 1.0 pt |
  | val_in_dist mean | 6.86% | 6.94% | +0.08 pt | 1.0 pt |

  Over-redaction also dropped on support_desk_val (14.5% → 7.5%) and support_desk_hard (11.8% → 8.9%).
- **Diagnostic on support_desk_hard, not gating:**
  - Leakage 10.33% → **5.28%**.
  - **M5 recovered 10 of the 21 spans M4 missed entirely.** These include spoken phone numbers ("plus four four seven seven double oh…", "zero two zero, seven nine four six…"), all four titles (Dr., Pvt., Mrs., Prof. Dr.) and the word-like name "will".
  - **Still missed:** the line-split digit phone, the unspaced UK mobile, the compact NINO "QQ123456C", the 4-8-4 card, lowercase "ms", "100" as an age, "dot", "mark", "Lagos", and an Aadhaar and PAN.
  - **7 new misses** that M4 had caught: CITY 3, ZIPCODE, CREDITCARDNUMBER, SEX, SURNAME.

  The team's diagnostic target ("near 0% on the known misses") is **not met**. About half of the known failure formats are still missed, even though the targeted data covers them.
- **Caveats:**
  - Single seed.
  - The fresh set is hard: M4 leaks 27.6% on it, above the team's 5–10% band for val_ood. Its leakage is dominated by spoken phone numbers and titles, where M5 gained most.
  - Both the fresh set and the training data are LLM-written, by different models.
- **Decision: M5 is the phase 4 candidate.** Phase 4 is waiting for the user's go (the rule set before the run).

## Phase 4: final blind evaluation of M5 (2026-10-02)
- **Run:** two A40 pods, about $3.30. Seeds 42 and 3407 were trained (135 min each); seed 13 is the milestone 1 adapter. Each of TAB, support_desk_300 and test_id was scored **once** per system.
- **Baselines:** GLiNER-PII (NVIDIA), OpenMed and Presidio on support_desk_300. Their TAB numbers are from week 3, and they were not run on test_id. Validators alone ran on all three sets.
- **Statistics:** 95% CIs from 1,000 document resamples, with per-document counts averaged over the seeds (`eval/phase4_report.py` → `results/phase4/phase4.md`).
- **Support desk (300).** M5 leaks **1.70% [0.82, 2.78]**, with document leakage 4.6% and over-redaction 5.8%. The full gateway leaks **1.25% [0.50, 2.14]** under the gateway scope, against 4.21% for the model alone in that scope (Δ −2.96 [−5.30, −1.02]). Paired against each baseline:

  | M5 − baseline | Leakage Δ (pt) | Over-redaction Δ (pt) |
  | --- | --- | --- |
  | OpenMed (3.54%) | −1.84 [−3.76, −0.03] | −10.3 |
  | GLiNER-PII (8.11%) | −6.41 [−9.46, −3.39] | −17.5 |
  | Presidio (18.91%) | −17.21 [−20.78, −13.36] | −17.3 |

- **TAB (127).** M5 leaks 15.91% [13.92, 18.42], with over-redaction 5.2%. Against the baselines:
  - OpenMed (14.91%): +1.01 [−0.85, +3.31], a tie;
  - GLiNER-PII (19.29%): −3.38 [−4.54, −2.22], better;
  - Presidio (11.23%): +4.69 [+2.92, +6.46], worse on leakage, though Presidio over-redacts 34.1%.

  The validators add nothing on TAB, and every system leaks some PII in every document.
- **test_id (5,000).** M5 leaks 0.60% [0.54, 0.69], with document leakage 13.8%; the gateway leaks 0.55%.
- **Seed spread:**

  | Test set | Seed 13 | Seed 42 | Seed 3407 |
  | --- | ---: | ---: | ---: |
  | support_desk_300 | 0.68% | 2.22% | 2.20% |
  | TAB | 15.05% | 17.52% | 15.17% |
  | test_id | 0.61% | 0.61% | 0.60% |

  Seed 13 is the run that passed the milestone 1 gate, so single-seed numbers on support text vary by about 1.5 pt. This is why seed-averaged numbers are reported.
- **Live gateway** (M5 + validators, uvicorn, one A40): 100 support_desk_val requests, all returned 200.
  - Latency was 0.38 s at the median and 1.08 s at p95 per request.
  - All 100 restores were exact, and the server log contained 0 values.
  - 31 of 205 gold values remained in the output, but 29 of those are AGE, SEX or CITY, which the analytics policy deliberately keeps.
  - Offline, the gateway leaks 1.51% on support_desk_val.
- **Docker:** both images build in CI. The CPU image ran locally and in CI; the GPU image was built and its imports checked, but it was not run.
- **Final model: M5** (r=16, learning rate 4e-4, `train_mix_32k`).

## Final coverage run: rules set before running (2026-10-02)
- **Model:** M5 is frozen and nothing is selected. The three seeds' adapters (13, 42, 3407) are scored once each on four sets:
  - `nemotron` (3,000) and `gretel_en` (1,000): the official test splits of two training sources. They are **in-distribution** for M5, which trained on their train splits, so they show what it learned, not how it generalises.
  - `test_holdout_regions` (2,000): OpenPII region IN, excluded from training. A clean unseen test.
  - `abcd` (1,002): **real human-typed support chats** from ASAPP's ABCD test split (MIT licence), labelled from each conversation's fictional customer card (`data/abcd/README.md`). A clean unseen test.
- **Baselines:** GLiNER-PII (NVIDIA), OpenMed and Presidio are re-scored on `abcd` and `nemotron`. The week-3 Nemotron predictions predate the duplicate-ID fix. The week-3 predictions on `test_holdout_regions` and `gretel_en` are still valid and are reused.
- **ABCD labelling rules:**
  - customer name → GIVENNAME / SURNAME; email, phone (with digit variants), street (BUILDINGNUM + STREET), city and zip → their labels;
  - username, account ID, PIN, password and security answer → IGNORE;
  - order ID, state, membership level, products and amounts → not labelled;
  - extra rules: a house number before a labelled street, an agent's self-introduction → GIVENNAME, a typed username → IGNORE.
- **ABCD filter:** drop a conversation whose delexicalised tokens show a value the matching missed. This dropped 2 of 1,004.
- **ABCD caveat:** only 10 distinct customer names and 9 cities, so results are reported by label as well as overall.
- **Statistics:** 95% CIs from 1,000 document resamples, seeds averaged. The model alone and the gateway (with validators) are both reported.
