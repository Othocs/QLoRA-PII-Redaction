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

## Results

<!-- RESULTS:BEGIN -->
Spans reviewed: 0. Documents with a missed-PII or note entry: 0.

No verdicts filled in yet.
<!-- RESULTS:END -->
