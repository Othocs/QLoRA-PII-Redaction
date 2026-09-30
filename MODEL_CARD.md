# Model card: PII span detector (QLoRA adapter)

> Placeholder. It will be filled in during week 6, when the adapters are pushed to the Hugging Face Hub.

- **Base model:** to be decided in week 2 (Qwen3-1.7B / Qwen3-4B candidates; licence to be checked).
- **Task:** list every PII span in English text as JSON `[{"label", "text"}]`. Code aligns the spans to the source text; the model never rewrites the text.
- **Training data:** Ai4Privacy OpenPII 1M, English rows (CC-BY-4.0, "Ai4Privacy / Ai Suisse SA"), with region IN held out.
- **Evaluation:** see `README.md` and `results/SUMMARY.md`.
- **Limitations:** trained on synthetic data with known label noise (`data/audit/AUDIT.md`). English only. No PHI or sensitive categories.
