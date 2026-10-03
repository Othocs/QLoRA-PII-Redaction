## Sweep: phase2_m4

Primary: leakage on support_desk_val + val_in_region (mean). Tie band 1.0 pt; tie-breaks: over-redaction, then leakage on dev + nemotron_dev + gretel_dev. Sanity rule on: dropped ≤ 5%, token-limit ≤ 2%.

| Run | Overrides | Primary leakage (%) | Over-redaction (%) | In-dist leakage (%) | Dropped values (%) | Hit token limit (%) | Sane | Missing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| p2_m4_lr4e-4 | output_dir=outputs/hpo/p2_m4_lr4e-4 learning_rate=4e-4 | 0.91 | 7.26 | 6.89 | 0.1 | 0.0 | yes |  |
| p1_lr4e-4 **(winner)** | output_dir=outputs/hpo/p1_lr4e-4 learning_rate=4e-4 | 0.99 | 6.21 | 10.64 | 0.1 | 0.0 | yes |  |
| p2_m4_lr2e-4 | output_dir=outputs/hpo/p2_m4_lr2e-4 learning_rate=2e-4 | 1.09 | 7.06 | 7.30 | 0.2 | 0.0 | yes |  |
| p2_m4_lr2.83e-4 | output_dir=outputs/hpo/p2_m4_lr2.83e-4 learning_rate=2.83e-4 | 1.53 | 7.83 | 7.15 | 0.1 | 0.0 | yes |  |

**Decision:** p1_lr4e-4: 4 candidates within 1.0 pt of the best leakage (0.91%); tie broken by over-redaction, then in-distribution leakage.
