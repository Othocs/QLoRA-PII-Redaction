# PII Redaction Gateway

A self-hosted gateway that finds personal data in English customer-support text and masks or pseudonymises it before the text is logged, analysed or sent to an external LLM. The detector is a small open LLM fine-tuned with QLoRA, backed by deterministic validators. It's benchmarked honestly against Presidio, GLiNER-PII and OpenMed's privacy filter.

> **Status: week 3 of 6.** Data, label audit, baselines, out-of-distribution tests and a training-data ablation are done; the current model is M3 (cleaned OpenPII + Nemotron, 20k). The gateway, demo and Docker images are stubs (see [Roadmap](#roadmap)).

## Why a small fine-tuned model

You can't send customer PII to a third party in order to remove the PII. A redactor has to run on-premises, on every message, with exact spans, and a missed phone number is a data leak. That makes it a narrow, high-volume task that a small model can learn, where cost and latency per call matter. GDPR names pseudonymisation as a safeguard (Art. 4(5), 25 and 32).

## Results

Headline metric: **leakage**, the share of gold PII characters left unmasked (lower is better). The out-of-distribution columns (held-out region, Nemotron-PII, support desk) are the headline; the in-distribution OpenPII column is not. Full per-test-set numbers are in [`results/SUMMARY.md`](results/SUMMARY.md).

<!-- RESULTS_TABLE:BEGIN -->
| System | Leakage, OpenPII (%) | Leakage, held-out region IN (%) | Leakage, Nemotron-PII (%) | Leakage, support desk (%) | Strict F1, support desk | p95 latency CPU (ms / 1k chars) | Cost ($ / 1M chars) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Presidio |  | 35.7 | 13.9 |  |  | 65 |  |
| GLiNER-PII (Knowledgator) |  | 12.7 | 15.0 |  |  | 725 |  |
| GLiNER-PII (NVIDIA) |  | 6.3 | 6.2 |  |  | 7763 |  |
| OpenMed privacy filter v2 |  | 0.8 | 2.4 |  |  |  |  |
| Base LLM, zero-shot |  | 46.8 | 39.4 |  |  |  |  |
| Your LoRA model |  | 0.8 | 3.7 |  |  |  |  |
| Your LoRA model + validators |  |  |  |  |  |  |  |
| LoRA r16, 10k (Qwen3-1.7B) |  | 0.9 | 21.7 |  |  |  |  |
| M0 OpenPII 10k |  | 0.8 | 19.0 |  |  |  |  |
| M1 cleaned OpenPII 10k |  | 0.8 | 18.7 |  |  |  |  |
| M2 cleaned OpenPII 5k + Nemotron 5k |  | 0.9 | 3.7 |  |  |  |  |
| M3 cleaned OpenPII 10k + Nemotron 10k |  | 0.8 | 3.7 |  |  |  |  |
<!-- RESULTS_TABLE:END -->

### First LoRA run (week 2, OpenPII dev, in-distribution)

All five systems below are scored on the same first 200 dev examples. The two LLM runs were also scored on all 2,000 dev examples, with near-identical results: LoRA leakage 0.74%, strict F1 0.938; zero-shot leakage 45.2%.

| System | Leakage (%) | Docs leaking (%) | Over-redaction (%) | Strict F1 | Partial F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Presidio | 33.63 | 89.0 | 22.36 | 0.232 | 0.721 |
| GLiNER-PII (Knowledgator) | 11.67 | 57.0 | 11.53 | 0.641 | 0.897 |
| GLiNER-PII (NVIDIA) | 4.64 | 51.5 | 6.92 | 0.701 | 0.953 |
| Qwen3-1.7B, zero-shot | 46.44 | 92.0 | 4.58 | 0.482 | 0.697 |
| **Qwen3-1.7B + LoRA (r=16, 10k examples)** | **0.67** | **13.0** | **0.52** | **0.941** | **0.994** |

**Training:**
- Hardware and cost: 55 minutes on an A40, 11.3 GB peak VRAM, **$0.45**.
- Size: 17.4M trainable parameters (1.0%), and a 33 MB adapter.
- Loss: dev loss fell from 0.019 to 0.008 over training.
- Output reliability: valid JSON on 100% of 2,000 outputs, with 37 of 15,255 returned values not found in the text and dropped.

**How to read this:**
- It's **in-distribution**. The model trained on the same generator it's tested on, so a big win is expected and proves little on its own. The headline comparison is the held-out region, Nemotron-PII and support-desk tests (weeks 3–5).
- A strict F1 of 0.94 sits right at the label-noise ceiling. The [audit](data/audit/AUDIT.md) found 8.9% of gold spans wrong, mostly AGE and CREDITCARDNUMBER, so the model has learned those mistakes too.
- **Latency is the LLM's weak point.** p50 is about 4.7 s per 1,000 characters for a single request on an A40 (batched throughput: 10.8k chars/s). GLiNER takes 0.3–2 s on a laptop CPU. Week 4 measures this properly.
- Zero-shot reached only 97.2% valid JSON despite constrained decoding, probably from long repetitive outputs hitting the 1,024-token limit. This is to be checked.

### Out-of-distribution tests (week 3)

The same LoRA model (trained only on English OpenPII from the CA, GB and US regions) and every baseline were run on six sets the model never trained on. Each cell is leakage in % (lower is better). Partial F1, per-language and per-label results are in [`results/SUMMARY.md`](results/SUMMARY.md).

| System | OpenPII, region IN | OpenPII, 6 other languages | Nemotron-PII | TAB (real court cases) | Gretel EN | Gretel, 6 other languages |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Presidio | 35.7 | n/a | 13.9 | **11.2** | 29.0 | n/a |
| GLiNER-PII (Knowledgator) | 12.7 | 11.9 | 15.0 | 58.1 | 46.7 | 44.9 |
| GLiNER-PII (NVIDIA) | 6.3 | 8.4 | 6.2 ¹ | 19.3 | **25.5** | **29.5** |
| OpenMed privacy filter v2 | **0.8** ² | 3.0 ² | **2.4** | 14.9 | 29.2 | 39.1 |
| Qwen3-1.7B, zero-shot | 46.8 | 51.3 | 39.4 | 82.7 | 62.0 | 65.5 |
| **Qwen3-1.7B + LoRA (r16, 10k)** | 0.9 | **2.4** | 21.7 | 23.2 | 44.1 | 45.0 |

¹ NVIDIA's GLiNER was trained on Nemotron-PII. ² OpenMed was trained partly on OpenPII.

**What it shows:**
- **Same generator, new region or language: LoRA is as good as the best baseline.** It was trained on English only but leaks 1.3–2.2% in French, German, Spanish, Italian and Dutch, and 5.2% in Bulgarian, which uses Cyrillic script.
- **Different generator or real text: LoRA falls behind NVIDIA's GLiNER and OpenMed.** On Nemotron it misses 40% of first names, against about 1% for GLiNER. It also misses ordinary English names, full addresses written in one piece, and date formats OpenPII never uses ("Jan-21"). It has learned OpenPII's style more than the task.
- **Fine-tuning still helps everywhere**: it cuts zero-shot leakage by 30–98%.
- **Presidio wins on the real court cases (TAB).** Its rules for names and dates hold up on real legal text, though it over-redacts a lot (34%).
- **Diagnostic (raw outputs logged, then re-scored offline):** the LoRA model's out-of-distribution gap is mostly the model, not the pipeline.
  - **Alignment isn't the cause.** Loose matching (case, whitespace, quotes) recovers under 1% of dropped values. The dropped values are **invented**: on 1–5% of chunks the model loops, repeating a made-up value (one card number 14 times, "98A" as AGE), until it hits the output limit. Alignment correctly discards these, but real PII after the loop is lost.
  - **Chunk size matters a little.** Splitting inputs into 1,200-character chunks (the training length) instead of 2,000 lowers leakage by 3–4 points on every OOD set: Nemotron 18.8%, TAB 19.0%, Gretel EN 40.5%, Gretel non-EN 42.4%. Dev is unchanged at 0.7%. This is now the default.
  - **Over-redaction of about 50% on Gretel finance documents** comes from the training-label noise the audit found: amounts and codes get tagged AGE or CREDITCARDNUMBER, as in OpenPII.
  - **Takeaway:** more varied, cleaner training data is the lever, not decoding tricks. The OOD table above uses the earlier 2,000-character setting; `results/diagB_*` holds the 1,200-character runs.
- **Label mappings:** see `configs/labels/eval/`. Labels outside our scope (company, IBAN, time, and so on) are ignored, so they count neither as leaks nor as over-redaction.

### Training-data ablation (week 3, part B)

Same model and recipe (Qwen3-1.7B, QLoRA r=16, 1 epoch, 1,200-character chunks); only the training data changes. Each cell is leakage % / over-redaction %.

| Model | OpenPII dev | **Gretel dev** (selection) | TAB | Gretel EN | Gretel non-EN | OpenPII IN | OpenPII non-EN | Nemotron ¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 OpenPII 10k | 0.7 / 0.4 | 40.0 / 51.5 | 18.5 / 3.5 | 40.7 / 47.2 | 42.4 / 54.3 | 0.8 / 0.5 | 2.3 / 1.6 | 19.0 / 6.0 |
| M1 cleaned OpenPII 10k | 0.8 / 0.5 | 40.2 / 51.2 | 19.5 / 3.3 | 40.4 / 47.2 | 41.7 / 52.9 | 0.8 / 0.5 | 2.3 / 1.4 | 18.7 / 5.8 |
| M2 cleaned OpenPII 5k + Nemotron 5k | 0.9 / 0.7 | **28.8** / 33.6 | 21.5 / 2.7 | 27.6 / 30.8 | 32.7 / 38.3 | 0.9 / 0.8 | 2.3 / 2.1 | 3.7 / 2.7 |
| **M3 cleaned OpenPII 10k + Nemotron 10k** | **0.7** / 0.5 | 29.3 / **30.8** | **18.3** / 2.0 | 28.7 / 29.0 | 33.1 / 33.6 | 0.8 / 0.5 | 2.1 / 1.8 | 3.7 / 2.6 |
| *GLiNER-PII (NVIDIA), reference* | | | 19.3 | 25.5 | 29.5 | 6.3 | 8.4 | 6.2 ² |

¹ Nemotron's train split is in M2 and M3's training data, so Nemotron test is in-distribution for them. ² NVIDIA's GLiNER was trained on Nemotron.

- **A second data source is what helps.** Adding Nemotron cut leakage on Gretel, a generator neither model saw, from about 40% to about 29% and roughly halved over-redaction. OpenPII results held.
- **Label cleaning alone barely moved anything** (M1 vs M0), even though it removed 32% of the card-number predictions on Gretel. Over-redaction there comes mostly from Gretel's own unlabelled dates and cities, EDI codes and blank form fields.
- **Doubling the mixed data** (M3 vs M2) left leakage about the same and reduced over-redaction further.
- **Selection rule, using dev sets only:** lowest Gretel-dev leakage, with differences under 1 point counted as a tie, broken by over-redaction and then OpenPII dev. M2 and M3 tie on leakage (28.8 vs 29.3), and M3 wins the tie-break, so **M3 is the current model.** It's now within 3–4 points of NVIDIA's GLiNER on Gretel and slightly better on TAB (18.3 vs 19.3).
- **Still open:** invented values remain frequent. On Gretel dev, 36% of M3's returned values weren't in the text and 2.8% of outputs hit the length limit. Alignment removes them, but the loops cost recall.
- **Cost:** M1 52 min ($0.43), M2 49 min ($0.40), M3 97 min ($0.80) on an A40; about $2.40 for the whole session with evaluation.

Caveats:
- "Your LoRA model" is M3 (see the training-data ablation below). Its Nemotron column is **in-distribution**, because Nemotron's train split is in its training data. Its out-of-distribution numbers are TAB, Gretel and the India region.
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

## Training and LLM evaluation (GPU)

Training and vLLM evaluation run on a rented GPU (RunPod), never on a laptop. Pods clone this private repo over `ssh -A` (agent forwarding), so no credential is copied to the pod:

```bash
ssh -A root@<pod-ip> -p <port> 'bash -s' < scripts/pod_setup.sh
```

The setup script clones the repo, installs uv and the `data`, `train` and `llm` extras, and runs `make data`. Then, on the pod:

```bash
uv run --no-sync python training/train_lora.py configs/train/r16_10k_1.7b.yaml
ADAPTER=outputs/r16_10k_1.7b TAG=lora_r16_10k_1.7b bash scripts/pod_eval.sh
```

- **Training:** QLoRA (4-bit NF4 base, bf16 compute) with TRL's `SFTTrainer`. Loss is computed on the JSON answer only. Each run writes `run_info.json` with the GPU, peak VRAM, wall time, tokens/s, trainable parameters, adapter size and cost.
- **Evaluation:** vLLM with JSON-schema-constrained decoding, so every output parses. The valid-JSON rate is still recorded in each result file.

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
- **Audit:** all 1,490 gold spans in 200 training documents were reviewed for label noise by an LLM (Claude), with a 50-row human spot-check pending. 8.9% are wrong, concentrated in AGE, CREDITCARDNUMBER, GENDER and TAXNUM ([`data/audit/AUDIT.md`](data/audit/AUDIT.md)).
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
