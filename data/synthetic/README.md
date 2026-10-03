# Targeted synthetic data (milestone 1)

M4 still leaks on a few formats: spelled-out or split phone numbers, titles, SSN, NINO and card numbers in unusual layouts, compact dates of birth and shorthand ages. This folder builds about 2,000 extra training messages around those formats. Mixed into the 30k training mix, they train M5.

## How the data is made

1. **`values.py`** generates the values in Python, so the labels are exact.
   - All values are fake. Numbers come from reserved or test ranges: Ofcom drama numbers, NANP 555-01xx, published test cards or random Luhn-valid numbers, SSNs in the never-issued 9xx area, and NINOs with the `QQ` prefix.
   - Names come from multicultural pools, including names that are also ordinary words.
2. **`generate_targeted.py`** asks DeepSeek (`deepseek-chat`, JSON mode) to write the message around those values. The model writes the text but never assigns labels.
   - Every occurrence of a supplied value is labelled, longest value first.
3. **A message is rejected** if:
   - a supplied value is missing;
   - a word-like name or a bare number appears more than once;
   - an unlabelled run of 7 or more digits, an email, or a title followed by a name appears outside the spans and the declared look-alikes (`meta.keep`);
   - it is longer than 1,200 characters;
   - it shares an 8-word sequence, or any non-name value, with a support-desk evaluation set.

## Running it

The key goes in `DEEPSEEK_API_KEY`, either in the environment or in `.env`, which is gitignored. Raw responses are cached in `raw/`, also gitignored, so a re-run resumes where it stopped.

```bash
uv run python data/synthetic/generate_targeted.py --n 50      # pilot
uv run python data/synthetic/generate_targeted.py --n 2000
uv run python data/prepare_train_mix.py targeted              # -> data/processed/train_mix_32k.jsonl
```

## Gate

The promotion gate is `data/support_desk/support_desk_fresh.jsonl`. It was committed before any message here was generated and was written by a different author (Claude). The rules are in `results/sweeps/DECISIONS.md`.
