# Evaluation protocol

How every number in this repository was produced. The code is in [`eval/`](../eval); the results are in [`results/`](../results).

## Metrics

All metrics are computed per document from character offsets ([`eval/metrics.py`](../eval/metrics.py)), then summed over documents. Predicted labels don't matter for leakage, because a masked value is protected whatever it is called. Labels do matter for F1.

| Metric | Definition | Direction |
| --- | --- | --- |
| **Character leakage** (headline) | Gold PII characters not covered by any predicted span ÷ all gold PII characters | lower is better |
| Document leakage | Share of documents with gold PII that leak at least one gold character | lower is better |
| Over-redaction | Predicted characters outside every gold span ÷ all predicted characters | lower is better |
| Strict P / R / F1 | Exact span and label matches | higher is better |
| Partial P / R / F1 | Any overlap with the same label | higher is better |
| Invented values | LLM only: share of returned values that could not be aligned to the text, which signals repetition loops | lower is better |
| Hit token limit | LLM only: share of outputs truncated at the generation limit | lower is better |

**Scope.**

- **IGNORE.** Gold spans of types outside our 19 labels are marked `IGNORE` and excluded from both leakage and over-redaction. Examples are TAB's quasi-identifiers, Nemotron's account numbers, and ABCD's usernames and account IDs.
- **Model-only scope** (the default). IBAN and IPADDRESS are `IGNORE`, because the LLM is not asked to find them; the gateway's validators are.
- **Gateway scope** (`--gateway-labels`, used for every "M5 + validators" row). IBAN and IPADDRESS count as gold. In the gateway tables, M5 alone is re-scored under the same scope, so the comparison is like for like.

## Statistics

- **95% confidence intervals** come from a document bootstrap: documents are resampled with replacement, 1,000 times ([`eval/bootstrap.py`](../eval/bootstrap.py)). Leakage and over-redaction are ratios of sums, so each resample re-adds per-document counts.
- **Seed averaging.** For the final model, per-document counts are averaged over its 3 training seeds (13, 42, 3407) before resampling. The interval therefore covers document sampling around the seed mean, and the spread between seeds is reported next to it.
- **Paired differences** (M5 − baseline, gateway − model) use the same resampled documents for both systems. A difference is called significant only when its CI excludes 0.

## Rules fixed before each run

Every phase's selection rule, pass/fail gate and test-set status was written to [`results/sweeps/DECISIONS.md`](../results/sweeps/DECISIONS.md) and committed **before** the run it governs. Git history shows the order.

- **Model selection** used only validation sets (dev, gretel_dev, nemotron_dev, val_in_region and the support_desk val, hard and fresh sets). The ties rule: differences under 1 pt are ties, broken by over-redaction.
- **The final test sets** (TAB, support_desk_300, test_id) were scored **once per system**, after the model was frozen. The later coverage sets (held-out region, Nemotron and Gretel test splits, ABCD) were also scored once, with M5 already frozen, and were never used to choose anything.
- **Results that went against the project**, such as the ABCD real-chat result, are reported as measured. The labelling rules are not changed after seeing results.

## Test sets

**Status** is relative to the final model, M5. *In-distribution* means M5 trained on the same source or generator, though never on these documents.

| Set | Docs | Source | Licence | Status for M5 | Role |
| --- | ---: | --- | --- | --- | --- |
| `support_desk_300` | 300 | LLM-drafted support messages ([`data/support_desk`](../data/support_desk/README.md)) | this repo | unseen (different author from the training data) | final test |
| `tab` | 127 | Text Anonymization Benchmark, ECHR court cases, human-annotated | MIT | unseen | final test |
| `test_id` | 5,000 | OpenPII 1M, English, deduplicated against training | CC-BY-4.0 | in-distribution | final test |
| `test_holdout_regions` | 2,000 | OpenPII 1M, region IN (excluded from training) | CC-BY-4.0 | unseen region | coverage |
| `abcd` | 1,002 | ASAPP ABCD test split: real human-typed support chats, labelled from scenario cards ([`data/abcd`](../data/abcd/README.md)) | MIT | unseen | coverage |
| `nemotron` | 3,000 | NVIDIA Nemotron-PII test split | CC-BY-4.0 | in-distribution | coverage |
| `gretel_en` | 1,000 | Gretel synthetic PII finance, English test split | Apache-2.0 | in-distribution | coverage |
| `gretel_xx`, `openpii_xx` | 2,629 / 3,000 | Non-English Gretel and OpenPII | as above | cross-lingual | weeks 2–3 only |

Validation sets, used for model selection only: `dev` (2,000), `gretel_dev` (1,000), `nemotron_dev` (1,000), `val_in_region` (1,000), and `support_desk_val`, `support_desk_hard` and `support_desk_fresh` (100 each). See [`DATASETS.md`](DATASETS.md).

## Systems compared

| System | What it is | How it was run |
| --- | --- | --- |
| **M5** | Qwen3-1.7B + QLoRA r=16 ([`MODEL_CARD.md`](../MODEL_CARD.md)) | vLLM, JSON-schema-constrained decoding, windows of 1,200 characters with 150 overlap |
| M5 + validators | The full gateway detector: M5 ∪ the deterministic validators, recall-first merge | [`eval/gateway_eval.py`](../eval/gateway_eval.py), re-scored from M5's saved predictions |
| OpenMed privacy filter v2 | Encoder PII model | GPU, label map in [`configs/labels/openmed.yaml`](../configs/labels/openmed.yaml) |
| GLiNER-PII (NVIDIA) | Zero-shot encoder NER for PII | GPU, label map in [`configs/labels/gliner.yaml`](../configs/labels/gliner.yaml) |
| Presidio 2.2 | Rules + spaCy `en_core_web_lg` | CPU |
| Validators alone | Email, Luhn-checked cards, IBAN mod-97, libphonenumber, IPs | CPU |
| Base LLM (zero-shot) | Qwen3-1.7B with the same prompt, no adapter | weeks 2–3 only |

The gateway rows are an approximation in one respect: offline, M5's predictions come from raw text, whereas the live gateway feeds it normalised text.

## Errata

[`results/ERRATA.md`](../results/ERRATA.md) records one correction. The Nemotron test set had 49 duplicated document IDs, so 98 of 3,000 documents were mis-scored and every system's Nemotron leakage was overstated by 0.8–1.2 pt. The set was rebuilt with unique IDs, the evaluator now refuses duplicates, and a test checks every set.

- The M5 and baseline Nemotron numbers in the final tables come from the fixed set.
- The week-3 M0–M3 Nemotron numbers in the report predate the fix, and the report says so.

## Reproducing a number

```bash
# a result file: results/<system>__<set>.json; per-document predictions: results/runs/ (gitignored)
uv run python -m eval.run_eval --systems validators --testsets abcd           # CPU systems run locally
uv run python -m eval.bootstrap --testset abcd --tags m5_targeted,m5_s42,m5_s3407 --n 1000
uv run python -m eval.phase4_report --name final_coverage --sets test_holdout_regions,abcd,nemotron,gretel_en
uv run python -m eval.figures
```

LLM and GPU-encoder predictions are produced on a rented GPU ([`scripts/pod_sweep.sh`](../scripts/pod_sweep.sh) and [`scripts/pod_eval_baselines.sh`](../scripts/pod_eval_baselines.sh)), never on a laptop.
