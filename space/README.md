---
title: PII Redaction Gateway (M5 demo)
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Find and redact personal data with a fine-tuned 1.7B LLM
---

# PII Redaction Gateway: M5 demo

Paste English customer-support text and see which personal data is found and how each value is masked, pseudonymised or kept.

## How it works

The page runs the gateway's redaction pipeline:
- **Validators:** deterministic checks (Luhn for cards, IBAN mod-97, phone numbers, email, IP addresses), on this Space's CPU.
- **M5:** Qwen3-1.7B fine-tuned with QLoRA, served by a GPU endpoint on RunPod that scales to zero when idle.
- **Merge and policy:** anything either one finds is redacted, then a policy masks, pseudonymises or keeps each value.

## Using it

- **Use fictional data only.** Submitted text goes to the GPU endpoint to run the model. This app doesn't store or log it.
- **Wait on the first request.** After a quiet period it can take up to about 3 minutes while the GPU wakes up; later requests take a second or two.
- **Limits:**
  - inputs up to 2,000 characters;
  - a per-visitor rate limit and a daily cap on requests.
- **If the model is unavailable,** the page says so and shows validator-only results, labelled as such.

## Results

M5 is best or tied-best on synthetic support messages, an unseen OpenPII region and Gretel-style documents. On real chat transcripts, encoder models such as GLiNER-PII still miss less. M5 is English only.
