# Errata

## 2026-10-02 — Nemotron-PII test set had 49 duplicate document ids

**What:** `data/processed/nemotron.jsonl` used `nemotron-<uid>` as the document id, but
Nemotron-PII reuses each `uid` for a document's `us` and `intl` versions. 49 ids appeared
twice (98 of 3,000 documents, 3.3%). Predictions are keyed by id, so each colliding pair
was scored with one document's predictions applied to both texts.

**Effect:** every system's Nemotron leakage was overstated by about 0.8–1.2 points.
Rankings and conclusions are unchanged. Corrected values below are re-scored from the
saved predictions on the 2,902 unaffected documents (no new inference).

| System | Reported leakage (%) | Corrected, 2,902 docs (%) |
| --- | ---: | ---: |
| GLiNER-PII (NVIDIA) | 6.2 | 5.0 |
| OpenMed privacy filter v2 | 2.4 | 1.3 |
| GLiNER-PII (Knowledgator) | 15.0 | 14.1 |
| Presidio | 13.9 | 12.9 |
| Qwen3-1.7B zero-shot | 39.4 | 38.6 |
| M0 (OpenPII 10k), 2,000-char chunks | 21.7 | 20.9 |
| M0, 1,200-char chunks | 19.0 | 18.2 |
| M1 (cleaned) | 18.7 | 17.8 |
| M2 (+ Nemotron 5k) | 3.7 | 2.5 |
| M3 (+ Nemotron 10k) | 3.7 | 2.5 |

**Fix:** ids are now `nemotron-<uid>-<locale>` (unique), the set was rebuilt with the same
documents in the same order, `nemotron_dev` likewise, and `eval/run_eval.py` now refuses a
test set with duplicate ids. A test asserts unique ids for every eval set. Training files
are unaffected (ids are not used in training). Nemotron numbers from phase 4 onwards use
the fixed set.
