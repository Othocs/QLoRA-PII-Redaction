# Support-desk sets (English)

These sets are the closest thing we have to production text: short English support messages (chats, emails, agent notes) with clearly fake values only. **Never paste real customer text here.**

| File | Messages | PII spans | Role (HPO plan) |
| --- | ---: | ---: | --- |
| `support_desk_val.jsonl` | 100 | 205 | `val_ood`: model selection from phase 2 on |
| `support_desk_300.jsonl` | 300 | 715 | `test_final`: touched once, in phase 4 |

**How they were made.** Claude drafted all 395 new messages; the 5 original hand-written seeds are in `support_desk_300` as `sd-0001` to `sd-0005`. The messages were then split 100 / 295 by a seeded shuffle. A human spot-check of 50 messages (`spotcheck_50.csv`: 30 val, 20 test) is pending. Until it's done, report these sets as "LLM-written and LLM-labelled".

**Coverage (400 messages):**
- Channels: 52% chat, 26% email, 22% agent notes (more chat than the 40/40/20 target).
- About 30% have no PII at all, to measure over-redaction.
- About 30% contain business keys that must be kept.
- 16 contain a full card number or IBAN (fewer than the ~10% target).
- Conventions are mostly US and UK, with some Canadian and Indian.
- Difficult cases include typos, lowercase names, informal tone, short non-English phrases, spelled-out emails, phone numbers written as words, email headers and forwarded threads.

## Format

There is one JSON object per line: `id`, `text`, `spans` (`label`, `text`, `start`, `end`), and `meta` (`channel`, `keep`, `note`). When writing, give spans as `{"label", "text"}` in order of appearance, then run:

```bash
uv run python data/support_desk/check_support_desk.py --path <file> --fill
```

`--fill` finds each value **as a whole word**, so a house number "9" is never matched inside "ORD-51290". The checker then validates labels, offsets, word boundaries, overlaps, and that every `meta.keep` string is present and unlabelled.

## Labelling conventions

- **Every person's name**, including support staff, third parties and family members. `GIVENNAME` and `SURNAME` are separate spans. A lone initial before a surname is `GIVENNAME` ("W. Ashworth"); a surname initial after a first name is not labelled ("Marcus T."). `TITLE` covers honorifics (Mr, Mrs, Ms, Mx, Dr, Prof., Sir).
- **The person's own address parts:** `BUILDINGNUM`, `STREET` (a named house counts as the street line), `CITY` and `ZIPCODE`. Not labelled: flat, unit, room and floor numbers, localities, states and countries, and the addresses of stores, warehouses and companies.
- **Dates and ages:**
    - `DATE` covers dates tied to the person: date of birth, their appointment, a delivery to them, work done at their home.
    - Not labelled: order dates, statement and letter dates, relative dates, and a year on its own (including a birth year).
    - `AGE` is a person's age; product age ratings and "18th" are not ages.
- **Kept (`meta.keep`), never labelled:** order, invoice, ticket, tracking, SKU, serial and gift-card numbers, including 16-digit numbers that look like cards; card last-four digits; promo codes; company names.
- **IBAN and IPADDRESS are labelled**, but model-only scoring treats them as `IGNORE` (`eval/run_eval.py`; `--gateway-labels` scores them), because the week 4 validators cover them.
