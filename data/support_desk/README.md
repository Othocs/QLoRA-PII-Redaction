# Support-desk test set (English)

This is the test set closest to production: 300 short English support messages (chats, emails and agent notes) written by hand. Every value is fake. **Never paste real customer text here.**

Five seed messages are in `support_desk.jsonl`, and the week 1 target is 100. It's scored as the `support_desk` test set in `eval/run_eval.py`.

## Format

There is one JSON object per line (see `schema.json`). When writing, give spans as `{"label", "text"}` in order of appearance. Then run:

```bash
uv run python data/support_desk/check_support_desk.py --fill
```

It fills in `start`/`end` and checks the file. `meta.keep` lists strings that look like identifiers but must **not** be masked.

## What counts as PII

Use our labels: the 19 OpenPII labels, plus `IBAN` and `IPADDRESS`.

- **Names:** `GIVENNAME` and `SURNAME` are separate spans; middle names count as GIVENNAME. `TITLE` is only an honorific (Mr, Ms, Dr). Label a name even when it's lowercase or misspelt.
- **Contact:** `EMAIL` and `TELEPHONENUM`. Write phone numbers the way customers do: spaced, dotted, with or without +44/+1, and occasionally as words.
- **Address:** `BUILDINGNUM`, `STREET`, `CITY` and `ZIPCODE`. States, counties and countries are **not** labelled; there's no label for them, and alone they aren't identifying.
- **Identifiers:** `IDCARDNUM`, `PASSPORTNUM`, `DRIVERLICENSENUM`, `SOCIALNUM`, `TAXNUM`, `CREDITCARDNUMBER` (the full number only), `IBAN` and `IPADDRESS`.
- **Personal attributes:** `AGE`, `SEX`, `GENDER`, and `DATE` for birthdates and other dates tied to the person.

## What must stay (put these in `meta.keep`)

- Order numbers, ticket IDs, invoice numbers, SKUs and product codes (`#ORD-88213`, `T-40912`, `BX-2210-M`).
- A year on its own ("customer since 2019").
- The last four digits of a card on their own ("card ending 1111").
- Company, product and shop names.

## Mix to aim for (300 total)

- About 40% chat (typos, lowercase, no punctuation), 40% email, 20% agent notes.
- US and UK conventions roughly equally, plus some Canadian and Indian ones.
- At least 60 messages with a business key that must be kept, and at least 30 with an IBAN or card number.
- At least 30 messages with **no** PII at all, to measure over-redaction.

An LLM may draft messages, but check every label by hand.

## Safe fake values

- **Phone numbers:** US 555-01xx numbers and UK Ofcom drama ranges (07700 900xxx, 020 7946 0xxx).
- **Email domains:** example.com, example.org and example.net.
- **IP addresses:** the documentation ranges 192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24.
- **Test cards:** 4111 1111 1111 1111 and 5555 5555 5555 4444.
- **Example IBAN:** GB82 WEST 1234 5698 7654 32.
