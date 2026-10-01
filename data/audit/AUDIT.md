# OpenPII (English) data audit

OpenPII is synthetic. If the model learns the generator's habits instead of the task, in-distribution scores go up for the wrong reason. This file records how noisy the gold labels are, so the results can be read with that in mind.

## How to review

1. `make audit` samples 200 training documents (equal shares of CA, GB and US) and writes `audit_spans.csv` and `audit_docs.csv`.
2. In `audit_spans.csv`, set `verdict` on every gold span:
   - `ok`
   - `wrong_label` (put the right one in `correct_label`)
   - `not_pii` (for example a statistic tagged SEX)
   - `bad_boundary` (the span cuts a value or includes extra text)
3. In `audit_docs.csv`, list any PII the gold labels **missed** in `missed_pii`, for example `TELEPHONENUM: 555 0100; SURNAME: Moreau`.
4. Run `python data/audit/score_audit.py`. It fills the results block below.

## Hypotheses to confirm or reject

These come from the dataset card's preview rows, which are mostly non-English:

- **Label noise:** statistics tagged SEX ("Female – 55%"), rating values tagged AGE, and common nouns tagged TITLE.
- **Injected values:** PII dropped into sentences where it makes no sense (a driver's licence number in a menu announcement).
- **Implausible values:** ages like "(2)" for a survey respondent, and dates such as "October/89".
- **Every date is PII:** event dates carry the same DATE label as birthdates. Whether a meeting date is personal data is a policy decision (see `configs/policy/`), not a model decision.
- **Placement bias:** most documents are letters that open with TITLE, GIVENNAME and SURNAME, which a model can learn as a position habit.

## Findings so far (automatic)

- **English split:** 143,515 train and 35,904 validation rows. Every `privacy_mask` offset matches its value (0 rows dropped).
- **Regions:** there are only four English regions (CA, GB, IN, US), about 25% each. IN is held out as the shift test set.
- **Text length:** p50 is about 280 characters and p95 about 876, with a maximum of 1,200.
- **Near-duplicates are rare.** MinHash is computed on masked text (PII replaced by `[LABEL]`), comparing dev and held-out items with the 50k training pool:

  | Word n-gram | Jaccard ≥ 0.3 | ≥ 0.5 | ≥ 0.7 | ≥ 0.8 (used) |
  | --- | ---: | ---: | ---: | ---: |
  | 5 | ~11% | ~0.5% | ~0% | 0.15–0.2% dropped |
  | 3 | ~30% | ~3–4% | ~0.2% | |

  The generator paraphrases rather than copying templates word for word. So the risk to in-distribution scores is shared *style* (letter openings, value formats), not leaked templates, and the out-of-distribution sets (Nemotron-PII and the support desk) remain the headline.

- **Card numbers mostly fail the Luhn check.** Only 10.1% of the 11,399 CREDITCARDNUMBER spans in `train_50k` pass, which is what random digits give. This has two consequences:
  - Luhn-gated detectors miss about 90% of OpenPII cards. Presidio leaks 63% of card characters on dev, so that result says more about the data than about Presidio.
  - The week 4 validators will use Luhn, as real cards pass it. Their recall has to be measured on the support-desk set, not on OpenPII.
- **Phone numbers use invented formats**, for example `+8-10.565 1155`, `+72.78390.9318` and `0133 67 337-8736`. Most aren't valid numbers in any country, so `phonenumbers` validation will reject them. This has the same consequence for the validators and for reading Presidio's 58% phone leakage.
- **Literal `\n` escapes.** 1.1% of training docs (568 of 50k) contain a backslash followed by `n` where a newline was meant. 247 gold spans start right after one. Every baseline then predicts `nbesulzbacher@gmail.com` for gold `besulzbacher@gmail.com`: that's a strict-F1 miss, but leakage is 0. The data is left as is for now; replacing the escapes with real newlines before training is a week 2 decision.

## Audit method (2026-10-01)

Claude (an LLM) reviewed all 1,490 gold spans in the 200 sampled documents, each read in full context. The verdicts are in `llm_audit.tsv`, and `apply_llm_audit.py` writes them into the sheets. A human spot-check is pending: `audit_spotcheck.csv` holds 25 flagged and 25 unflagged spans. Until it's done, report these numbers as "LLM-audited, human spot-check pending".

The rubric separates **errors** from **policy**:

- **Verdicts mark annotation errors:**
  - `not_pii`: the span isn't a person's data in context. Examples: "increased by 81 %", "average age 60", "rent of £6581…", a software version.
  - `wrong_label`: an employee or case number labelled as an ID card number.
  - `bad_boundary`: two postcodes in one span, or a trailing space.
- **Note tags mark spans that follow the dataset's convention but aren't really personal:**
  - `event_date`: a meeting, deadline or message date.
  - `place`: a venue, office or region address.
  - `injected`: a value dropped into a sentence where it makes no sense.

  These stay `ok` because masking them is a policy choice, not a labelling mistake.

## What the audit shows

- **Overall, 8.9% of gold spans are wrong** (95% CI 7.6–10.5%). This is almost all `not_pii` (111 of 133).
- **Errors are concentrated in a few labels:**
  - **AGE 51% wrong:** the generator fills AGE slots with percentages, durations ("64 years of experience"), counts and group statistics.
  - **CREDITCARDNUMBER 45% wrong:** money amounts ("total funds collected: $…", "rent of £…").
  - **GENDER 30%, TAXNUM 29%, SEX 17%:** group statistics ("gender ratio (F)") and regulation or budget references.
  - **ZIPCODE 14%:** mostly spans holding two or three postcodes.
  - **Names, emails and phone numbers: 0–1%.**
- **What this means for scores:**
  - Per-label OpenPII scores for AGE, CREDITCARDNUMBER, GENDER and TAXNUM have a low ceiling. A model that "beats" the gold there may be right, and one that matches it is learning to mask money and percentages.
  - Report these labels separately, and read the support-desk and Nemotron results for them.
- **Policy-level findings:**
  - **70% of DATE spans are event dates** (133 of 189), not dates of a person.
  - **58% of address spans** (178 of 308 CITY, STREET, BUILDINGNUM and ZIPCODE) are business, venue or region addresses.
  - A model trained on OpenPII will therefore mask all dates and all addresses. That's the right default for a recall-first redactor, but `configs/policy/` is where an analytics policy can keep them.
- **Missed PII is rare:** only 4 values, all "Dr." never being labelled TITLE.
- **Training implication (week 2/3 decision):** the noise is systematic, not random, so the model *will* learn it. Options:
  - Leave it and report it (the current choice for the first run).
  - Drop training examples whose AGE or CREDITCARDNUMBER spans follow money or percent cues.
  - Compare both as an ablation.

## Cleaning rules (week 3, part B)

`data/clean_labels.py` turns the audit's noise patterns into four context rules:
- `money_as_card`: a CREDITCARDNUMBER right after a currency sign or an amount word.
- `age_not_a_person`: an AGE that's a percentage, duration, range or group threshold.
- `group_attribute`: GENDER or SEX describing a group.
- `reference_not_id`: a number followed by "regulations", "compliance", "entries" and the like.

A training example is **dropped**, never relabelled, if any of its spans is flagged.

| Check | Result |
| --- | --- |
| Against the 1,490 audited spans | precision **1.00** (61/61); recall 0.67 on the noisy AGE / CREDITCARDNUMBER / GENDER / SEX / TAXNUM spans |
| 40 random flags from unseen training docs, before tightening | 36–37 of 40 correct |
| 40 fresh random flags after tightening | 40 of 40 correct |
| Effect on `train_50k` | 6,034 examples dropped (12%), 43,966 kept |

The audit sample was used to write the rules, so its precision is optimistic. The unseen samples are the honest check. Two fixes came from them: "fee payable via ⟦card⟧" is no longer flagged, and "aged" only counts as a group cue after a plural noun. Both cases are in `tests/test_clean_labels.py`.

## Results

<!-- RESULTS:BEGIN -->
Spans reviewed: 1490. Documents with a missed-PII or note entry: 200.

| Label | Reviewed | Wrong | Error rate | 95% CI | wrong_label | not_pii | bad_boundary |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| GIVENNAME | 191 | 0 | 0.0% | 0.0%–2.0% | 0 | 0 | 0 |
| DATE | 189 | 5 | 2.6% | 1.1%–6.0% | 0 | 5 | 0 |
| SURNAME | 152 | 1 | 0.7% | 0.1%–3.6% | 0 | 1 | 0 |
| CITY | 103 | 2 | 1.9% | 0.5%–6.8% | 0 | 2 | 0 |
| TITLE | 96 | 1 | 1.0% | 0.2%–5.7% | 0 | 1 | 0 |
| EMAIL | 90 | 0 | 0.0% | 0.0%–4.1% | 0 | 0 | 0 |
| TELEPHONENUM | 83 | 1 | 1.2% | 0.2%–6.5% | 0 | 0 | 1 |
| AGE | 81 | 41 | 50.6% | 40.0%–61.2% | 0 | 41 | 0 |
| BUILDINGNUM | 72 | 3 | 4.2% | 1.4%–11.5% | 0 | 3 | 0 |
| STREET | 69 | 1 | 1.4% | 0.3%–7.8% | 0 | 1 | 0 |
| ZIPCODE | 64 | 9 | 14.1% | 7.6%–24.6% | 0 | 1 | 8 |
| IDCARDNUM | 48 | 8 | 16.7% | 8.7%–29.6% | 6 | 2 | 0 |
| GENDER | 46 | 14 | 30.4% | 19.1%–44.8% | 0 | 14 | 0 |
| TAXNUM | 42 | 12 | 28.6% | 17.2%–43.6% | 1 | 11 | 0 |
| CREDITCARDNUMBER | 40 | 18 | 45.0% | 30.7%–60.2% | 0 | 18 | 0 |
| DRIVERLICENSENUM | 34 | 4 | 11.8% | 4.7%–26.6% | 3 | 1 | 0 |
| SOCIALNUM | 33 | 5 | 15.2% | 6.7%–30.9% | 2 | 3 | 0 |
| SEX | 29 | 5 | 17.2% | 7.6%–34.5% | 0 | 5 | 0 |
| PASSPORTNUM | 28 | 3 | 10.7% | 3.7%–27.2% | 1 | 2 | 0 |
| **All** | 1490 | 133 | 8.9% | 7.6%–10.5% | 13 | 111 | 9 |

Spans judged correct by the dataset's convention but tagged in notes:
- `event_date`: 133 (DATE 133)
- `place`: 178 (CITY 67, BUILDINGNUM 40, STREET 39, ZIPCODE 32)
- `injected`: 40 (DRIVERLICENSENUM 7, SOCIALNUM 5, IDCARDNUM 4, TAXNUM 4, AGE 4, CREDITCARDNUMBER 3, TELEPHONENUM 3, SEX 2, EMAIL 2, GENDER 2, GIVENNAME 1, SURNAME 1, PASSPORTNUM 1, TITLE 1)

PII missed by the gold labels, by label: TITLE 4
<!-- RESULTS:END -->
