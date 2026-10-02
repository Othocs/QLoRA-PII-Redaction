## Sweep: phase1_lr

Primary: leakage on gretel_dev (mean). Tie band 1.0 pt; tie-breaks: over-redaction, then leakage on dev.

| Run | Overrides | Primary leakage (%) | Over-redaction (%) | In-dist leakage (%) | Dropped values (%) | Hit token limit (%) | Sane | Missing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| p1_lr6e-4 | output_dir=outputs/hpo/p1_lr6e-4 learning_rate=6e-4 | 28.70 | 30.23 | 0.68 | 30.5 | 2.5 | no |  |
| p1_lr4e-4 **(winner)** | output_dir=outputs/hpo/p1_lr4e-4 learning_rate=4e-4 | 29.13 | 28.74 | 0.58 | 41.2 | 3.4 | no |  |
| p1_lr2e-4 | output_dir=outputs/hpo/p1_lr2e-4 learning_rate=2e-4 | 29.54 | 29.76 | 0.67 | 36.5 | 3.1 | no |  |
| p1_lr5e-5 | output_dir=outputs/hpo/p1_lr5e-5 learning_rate=5e-5 | 29.63 | 37.75 | 1.58 | 42.7 | 4.1 | no |  |
| p1_lr1e-4 | output_dir=outputs/hpo/p1_lr1e-4 learning_rate=1e-4 | 29.80 | 31.82 | 0.99 | 38.0 | 3.4 | no |  |

**Decision:** p1_lr4e-4: 4 candidates within 1.0 pt of the best leakage (28.70%); tie broken by over-redaction, then in-distribution leakage.
