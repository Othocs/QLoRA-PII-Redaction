# PII Redaction Gateway

A self-hosted gateway that finds personal data in English customer-support text and masks or pseudonymises it before the text is logged, analysed or sent to an external LLM. The detector is a small open LLM fine-tuned with QLoRA, backed by deterministic validators. It's benchmarked honestly against Presidio, GLiNER-PII and OpenMed's privacy filter.

> **Status: week 1 of 6.** Data splits, audit tooling, metrics and baseline evaluation work. The LoRA model, gateway, demo and Docker images are stubs (see [Roadmap](#roadmap)).

## Why a small fine-tuned model

You can't send customer PII to a third party in order to remove the PII. A redactor has to run on-premises, on every message, with exact spans, and a missed phone number is a data leak. That makes it a narrow, high-volume task that a small model can learn, where cost and latency per call matter. GDPR names pseudonymisation as a safeguard (Art. 4(5), 25 and 32).

## Results

Headline metric: **leakage**, the share of gold PII characters left unmasked (lower is better). The out-of-distribution columns (held-out region, Nemotron-PII, support desk) are the headline; the in-distribution OpenPII column is not. Full per-test-set numbers are in [`results/SUMMARY.md`](results/SUMMARY.md).

<!-- RESULTS_TABLE:BEGIN -->
| System | Leakage, OpenPII (%) | Leakage, held-out region IN (%) | Leakage, Nemotron-PII (%) | Leakage, support desk (%) | Strict F1, support desk | p95 latency CPU (ms / 1k chars) | Cost ($ / 1M chars) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Presidio |  |  |  |  |  | 65 |  |
| GLiNER-PII (Knowledgator) |  |  |  |  |  | 725 |  |
| GLiNER-PII (NVIDIA) |  |  |  |  |  | 7763 |  |
| OpenMed privacy filter v2 |  |  |  |  |  |  |  |
| Base LLM, zero-shot |  |  |  |  |  |  |  |
| Your LoRA model |  |  |  |  |  |  |  |
| Your LoRA model + validators |  |  |  |  |  |  |  |
<!-- RESULTS_TABLE:END -->

Caveats:
- OpenMed's privacy filter was trained partly on OpenPII, so its OpenPII score is a ceiling, not a fair fight.
- NVIDIA's GLiNER-PII was trained on Nemotron-PII's train split.
- The week 1 numbers are on 200 dev examples ([`results/SUMMARY.md`](results/SUMMARY.md)). CPU latencies come from an 8 GB M1 under heavy memory pressure and will be re-measured in week 4.
- OpenPII card and phone numbers are mostly not valid numbers (only 10% of cards pass Luhn), which penalises validator-based systems such as Presidio in-distribution. See [`data/audit/AUDIT.md`](data/audit/AUDIT.md).

## Scope

**In scope:**
- English text
- The 19 OpenPII labels, plus IBAN and IP address (validators)
- Reversible pseudonymisation

**Out of scope:**
- Non-English text
- Images, PDFs and audio (OCR or speech-to-text can feed the gateway later)
- Health data (PHI) and free-text sensitive categories such as religion or political opinion
- Any real customer data: every example, including demo text, is synthetic or hand-written

## Quickstart

```bash
make setup     # uv sync with the data, presidio, gliner and serve extras
make data      # download OpenPII (4.6 GB), keep English, build splits + dedup
make test      # unit tests
make eval      # Presidio + both GLiNER-PII models on 200 dev examples (CPU)
```

`make help` lists every target. Evaluate other systems or test sets like this:

```bash
make eval SYSTEMS=openmed TESTSETS=dev,test_id,test_holdout_regions EVAL_LIMIT=
```

## Data

**Training data:** [Ai4Privacy OpenPII 1M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m) (CC-BY-4.0, credit "Ai4Privacy / Ai Suisse SA"), English rows only.

- The English split has 143,515 train and 35,904 validation rows, drawn from four regions (CA, GB, IN, US) of about 25% each. See `data/processed/stats.json` after `make data`.

| Split | Size | Source |
| --- | ---: | --- |
| `train_2k` ⊂ `train_10k` ⊂ `train_50k` | 2k / 10k / 50k | train, regions CA/GB/US, stratified by region × rarest label |
| `dev` | 2,000 | validation, CA/GB/US |
| `test_id` | 5,000 | validation, CA/GB/US |
| `test_holdout_regions` | 2,000 | validation, **IN only** (never in training) |

- **Deduplication:** dev and test items that are near-duplicates of the training pool are dropped. The check is MinHash on masked text (PII replaced by `[LABEL]`) at 5-gram Jaccard ≥ 0.8, and only about 0.2% were dropped. The generator paraphrases more than it reuses templates; see [`data/audit/AUDIT.md`](data/audit/AUDIT.md).
- **Audit:** 200 training documents are hand-checked for label noise ([`data/audit/`](data/audit/)).
- **Support-desk test set:** 300 hand-written English support messages, including business keys that must *not* be masked ([`data/support_desk/`](data/support_desk/)). It's in progress.
- **Nemotron-PII:** 3,000 records from the test split, with labels mapped to ours. This arrives in week 3.

## Metrics

Defined in [`eval/metrics.py`](eval/metrics.py):

- **Leakage:** share of gold PII characters left unmasked. Also reported as the share of documents where at least one value survives. It ignores labels ("was it masked?").
- **Over-redaction:** share of masked characters that weren't PII.
- **Strict F1:** exact boundaries and label.
- **Partial F1:** any overlap, label ignored.
- **Latency:** p50 and p95 per 1,000 characters.

All metrics are also broken down per label and per region.

## Roadmap

1. **Week 1 (this pass): data and baselines.** Splits, MinHash dedup, audit sheets, metrics, Presidio / GLiNER / OpenMed wrappers and `make eval`.
2. **Week 2: first LoRA model.** Prompt format, span alignment, JSON-constrained decoding, then r=16 on 10k examples with Qwen3-1.7B.
3. **Week 3: ablations.** Rank 8/16/32, data 2k/10k/50k, 1.7B vs 4B, attention-only vs all linear layers; Nemotron-PII evaluation.
4. **Week 4: gateway.** Validators, normalisation, recall-first merge, policies, encrypted vault, FastAPI `/redact` `/restore` `/proxy`, and the canary log-leak test.
5. **Week 5: demo and hard tests.** Gradio (Redact, Safe LLM, Compare), GPU and CPU Docker images, the finished support-desk and adversarial sets.
6. **Week 6: release.** Model card, adapters on the Hugging Face Hub, write-up.

### Demo example (week 5)

Every value here is fake: 555 numbers, example.com, the published example IBAN and a test card.

```text
Hi, this is Sarah Mitchell (customer since 2019). My order #ORD-88213 never arrived at
42 Elm Street, Springfield, IL 62704. Call me on (217) 555-0143 or sarah.mitchell@example.com.
Please refund to IBAN GB82 WEST 1234 5698 7654 32, not the card 4111 1111 1111 1111.
```

In mask mode, the order number and "2019" are kept, and the IBAN and card number are caught by the validators (mod-97 and Luhn):

```text
Hi, this is [GIVENNAME_1] [SURNAME_1] (customer since 2019). My order #ORD-88213 never arrived at
[BUILDINGNUM_1] [STREET_1], [CITY_1], IL [ZIPCODE_1]. Call me on [TELEPHONENUM_1] or [EMAIL_1].
Please refund to IBAN [IBAN_1], not the card [CREDITCARDNUMBER_1].
```

## Repository layout

```text
configs/        data, label maps for baselines, training runs, redaction policies
data/           prepare_openpii.py, audit/, support_desk/ (raw + processed data are gitignored)
src/pii_gateway spans.py (shared types), detectors/ (baselines, LLM, validators), gateway modules
eval/           metrics.py, run_eval.py, summarize.py
training/       train_lora.py (week 2)
demo/, docker/  week 5
tests/          unit tests + an end-to-end eval smoke test
```
