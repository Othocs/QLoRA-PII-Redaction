# Datasets

Every dataset used for training, validation or testing: where it comes from, how we processed it, its known caveats, and its licence. The processed files live in `data/processed/` (gitignored, rebuilt by the scripts below). The support-desk and targeted sets we created are committed.

## Training data (M5's `train_mix_32k`)

### Ai4Privacy OpenPII 1M, English
- **Source:** [`ai4privacy/pii-masking-openpii-1m`](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m), 1.43M synthetic rows, 23 languages, 19 labels. **CC-BY-4.0**, attribution "Ai4Privacy / Ai Suisse SA".
- **Processing** ([`data/prepare_openpii.py`](../data/prepare_openpii.py), [`configs/data/openpii_en.yaml`](../configs/data/openpii_en.yaml)):
  - English only;
  - region IN removed from every training and in-distribution split, for the held-out region test;
  - MinHash near-duplicate removal over the masked text (128 permutations, threshold 0.8, 5-gram), so dev and test share no templates with training;
  - stratified sampling.
- **Label audit** ([`data/audit/AUDIT.md`](../data/audit/AUDIT.md)): 1,490 gold spans in 200 documents were reviewed. **8.9% are wrong**, concentrated in AGE (51%) and CREDITCARDNUMBER (45%).
- **Cleaning** ([`data/clean_labels.py`](../data/clean_labels.py)) removed about 12% of documents with noisy labels (`train_clean_10k`). On its own it barely changed results.
- **Used:** 10,000 cleaned training examples. dev (2,000), test_id (5,000), test_holdout_regions (2,000) and val_in_region (1,000) are disjoint from training.

### NVIDIA Nemotron-PII
- **Source:** [`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII), synthetic documents in US and international locales. **CC-BY-4.0**.
- **Processing** ([`data/prepare_train_mix.py`](../data/prepare_train_mix.py), [`configs/labels/train/nemotron.yaml`](../configs/labels/train/nemotron.yaml)):
  - the train split is cut into windows of at most 1,200 characters, with no span crossing a boundary;
  - labels are mapped to ours;
  - documents with PII our labels can't name (account and customer IDs and similar) are excluded.
- **Used:** 10,000 training windows. `nemotron_dev` (1,000) comes from train documents not used for training; the `nemotron` test set (3,000) comes from the official test split.

### Gretel synthetic PII (finance, multilingual)
- **Source:** [`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual), synthetic financial documents. **Apache-2.0**.
- **Processing:**
  - the English train split is cut into windows of at most 1,200 characters;
  - full-name spans are split into TITLE, GIVENNAME and SURNAME;
  - documents with unnameable PII or placeholder names are dropped;
  - the 1,000 `gretel_dev` documents are excluded.
- **Used:** 10,000 training windows. `gretel_dev` (1,000) comes from the train split; `gretel_en` (1,000) and `gretel_xx` (2,629) come from the test split.

### Targeted synthetic support messages (this repo)
- **What:** 2,000 support messages aimed at the formats the previous model (M4) missed: spoken, split and unspaced phone numbers; stacked or lowercase titles; SSN, NINO and SIN numbers and card layouts; compact dates of birth; shorthand ages; and look-alike business numbers to keep. See [`data/synthetic/README.md`](../data/synthetic/README.md).
- **How:**
  - Python generates every PII value, using reserved and test ranges only;
  - the DeepSeek API writes the message around the values;
  - labels are exact by construction.
- **Rejection rules:** messages are dropped for missing values, ambiguous word-like names, unlabelled digits, emails or title-plus-name sequences, a phone given as the company's, a date of birth used as a header date, or a look-alike presented as a card.
- **Contamination:** none. No shared 8-word sequence or value with any support-desk evaluation set.
- **File:** [`data/synthetic/targeted_2k.jsonl`](../data/synthetic/targeted_2k.jsonl), 7,859 spans.

## Test and validation sets

### Support-desk sets (this repo, LLM-drafted)
Short English customer-support messages: chat, email and agent notes from US, UK, Canadian and Indian customers. All values are fake. Drafted by Claude; **not human-checked**. Conventions: [`data/support_desk/README.md`](../data/support_desk/README.md).

| File | Msgs | Spans | Role |
| --- | ---: | ---: | --- |
| `support_desk_val.jsonl` | 100 | 205 | validation |
| `support_desk_hard.jsonl` | 100 | 371 | hard slice, added after `val_ood` saturated; the milestone 1 diagnostic |
| `support_desk_fresh.jsonl` | 100 | 280 | milestone 1 gate, committed before the targeted data existed |
| `support_desk_300.jsonl` | 300 | 715 | **final test** |

### TAB: Text Anonymization Benchmark
- **Source:** [`ildpil/text-anonymization-benchmark`](https://huggingface.co/datasets/ildpil/text-anonymization-benchmark) (Pilán et al., 2022). European Court of Human Rights judgements, annotated by people. **MIT**.
- **Used:** 127 test documents, about 5,000 characters each. TAB's own entity types are mapped to ours ([`configs/labels/eval/tab.yaml`](../configs/labels/eval/tab.yaml)); quasi-identifiers outside our scope are `IGNORE`.

### ABCD: real human-typed support chats
- **Source:** [ASAPP ABCD](https://github.com/asappresearch/abcd), test split. Trained agents chat with crowd workers who play a customer from a scenario card. **MIT**.
- **Processing** ([`data/prepare_abcd.py`](../data/prepare_abcd.py)):
  - labels come from matching the card's fictional values in the text (name, email, phone, street, city, zip); usernames and account IDs are `IGNORE`;
  - a consistency check against ABCD's delexicalised copy drops conversations where a value was retyped differently, 2 of 1,004.
- **Caveats** ([`data/abcd/README.md`](../data/abcd/README.md)): only 10 distinct customer names and 9 cities; order IDs and purchase dates are unlabelled, which inflates every system's over-redaction.

### Week-3 cross-lingual sets
`gretel_xx` (de, nl, es, it, sv, fr) and `openpii_xx` (fr, de, es, it, nl, bg, 500 each) measured cross-lingual transfer of the English-only models. The project is English-only, so M5 was not evaluated on them.

## Attribution

[`NOTICE`](../NOTICE) lists the attribution each licence requires. If you redistribute processed data derived from these sources, keep their licences: CC-BY-4.0 requires attribution, while Apache-2.0 and MIT require keeping the licence notice.
