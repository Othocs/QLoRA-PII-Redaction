## Sweep: phase3_rank

Primary: leakage on support_desk_val + support_desk_hard + val_in_region (mean). Tie band 1.0 pt; tie-breaks: over-redaction, then leakage on dev + nemotron_dev + gretel_dev. Sanity rule on: dropped ≤ 5%, token-limit ≤ 2%.

| Run | Overrides | Primary leakage (%) | Over-redaction (%) | In-dist leakage (%) | Dropped values (%) | Hit token limit (%) | Sane | Missing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| p2_m4_lr4e-4 | output_dir=outputs/hpo/p2_m4_lr4e-4 learning_rate=4e-4 | 4.04 | 8.83 | 6.86 | 0.1 | 0.0 | yes |  |
| p3_r32_lr1.4e-4 **(winner)** | output_dir=outputs/hpo/p3_r32_lr1.4e-4 learning_rate=1.4e-4 lora.r=32 lora.alpha=64 | 4.24 | 8.67 | 7.41 | 0.2 | 0.0 | yes |  |
| p3_r32_lr2e-4 | output_dir=outputs/hpo/p3_r32_lr2e-4 learning_rate=2e-4 lora.r=32 lora.alpha=64 | 4.54 | 8.99 | 6.93 | 0.1 | 0.0 | yes |  |
| p3_r32_lr1e-4 | output_dir=outputs/hpo/p3_r32_lr1e-4 learning_rate=1e-4 lora.r=32 lora.alpha=64 | 4.56 | 9.14 | 7.50 | 0.2 | 0.0 | yes |  |

**Decision:** p3_r32_lr1.4e-4: 4 candidates within 1.0 pt of the best leakage (4.04%); tie broken by over-redaction, then in-distribution leakage.

### Phase 3 decision matrix vs baseline p2_m4_lr4e-4

| Candidate | Rank | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI | Hard slice leakage (%) | Verdict |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| p3_r32_lr1e-4 | 32 | +0.52 | [-0.30, +1.59] | +0.31 | [-0.63, +1.42] | 10.9 | C: parity/noise (rel -13%, over +0.31 pt) |
| p3_r32_lr1.4e-4 | 32 | +0.20 | [-0.44, +0.91] | -0.16 | [-1.14, +0.89] | 10.6 | C: parity/noise (rel -5%, over -0.16 pt) |
| p3_r32_lr2e-4 | 32 | +0.49 | [-0.30, +1.44] | +0.16 | [-0.63, +1.00] | 10.2 | C: parity/noise (rel -12%, over +0.16 pt) |

Baseline hard-slice leakage: 10.3%.
**Phase 3 decision:** p2_m4_lr4e-4 (rank 16).
