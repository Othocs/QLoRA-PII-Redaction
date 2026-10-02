## Sweep: m5_targeted

Primary: leakage on support_desk_fresh (mean). Tie band 1.0 pt; tie-breaks: over-redaction, then leakage on dev + nemotron_dev + gretel_dev. Sanity rule on: dropped ≤ 5%, token-limit ≤ 2%.

| Run | Overrides | Primary leakage (%) | Over-redaction (%) | In-dist leakage (%) | Dropped values (%) | Hit token limit (%) | Sane | Missing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| m5_targeted **(winner)** | output_dir=outputs/hpo/m5_targeted learning_rate=4e-4 train_file=data/processed/train_mix_32k.jsonl | 4.57 | 3.12 | 6.94 | 0.4 | 0.0 | yes |  |
| p2_m4_lr4e-4 | output_dir=outputs/hpo/p2_m4_lr4e-4 learning_rate=4e-4 | 27.65 | 8.71 | 6.86 | 0.4 | 0.0 | yes |  |

**Decision:** m5_targeted: lowest primary leakage (4.57%), no other candidate within 1.0 pt.
