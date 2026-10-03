## Final coverage: held-out region, training sources' test splits, ABCD real chats

M5 = Qwen3-1.7B + QLoRA r=16, lr 4e-4, train_mix_32k; seeds 13, 42, 3407 (per-document counts averaged over seeds). 95% CIs: 1000 document resamples. Each test set was scored once per system.

### test_holdout_regions (2000 documents)

| System | Char leakage (%) | 95% CI | Doc leakage (%) | 95% CI | Over-redaction (%) | 95% CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| **M5, model alone** | 0.70 | [0.60, 0.82] | 14.02 | [12.60, 15.40] | 0.56 | [0.47, 0.65] |
| gliner_nvidia | 6.31 | [5.87, 6.75] | 53.30 | [51.20, 55.60] | 5.70 | [5.23, 6.15] |
| openmed | 0.80 | [0.71, 0.90] | 27.50 | [25.55, 29.40] | 1.35 | [1.18, 1.55] |
| presidio | 35.67 | [34.90, 36.45] | 89.90 | [88.45, 91.15] | 18.99 | [18.13, 19.83] |
| validators | 84.04 | [83.45, 84.60] | 98.85 | [98.35, 99.25] | 0.74 | [0.40, 1.13] |

Seed leakage (13, 42, 3407): 0.72, 0.71, 0.68.

M5 leakage by label (%, seed mean): AGE 6.3, BUILDINGNUM 0.8, CITY 1.8, CREDITCARDNUMBER 0.9, DATE 0.1, DRIVERLICENSENUM 1.4, EMAIL 0.3, GENDER 0.0, GIVENNAME 1.2, IDCARDNUM 0.9, PASSPORTNUM 0.0, SEX 0.6, SOCIALNUM 0.4, STREET 0.9, SURNAME 0.9, TAXNUM 0.3, TELEPHONENUM 0.1, TITLE 0.5, ZIPCODE 0.1.

| M5 − baseline | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI |
| --- | ---: | --- | ---: | --- |
| gliner_nvidia | -5.60 | [-6.05, -5.13] | -5.14 | [-5.59, -4.69] |
| openmed | -0.10 | [-0.24, 0.04] | -0.79 | [-0.98, -0.61] |
| presidio | -34.97 | [-35.75, -34.15] | -18.43 | [-19.27, -17.57] |
| validators | -83.34 | [-83.87, -82.75] | -0.18 | [-0.54, 0.15] |

Gateway label scope (IBAN, IPADDRESS counted):

| System | Char leakage (%) | 95% CI | Doc leakage (%) | Over-redaction (%) |
| --- | ---: | --- | ---: | ---: |
| M5 model alone | 0.70 | [0.60, 0.82] | 14.02 | 0.56 |
| **M5 + validators (gateway)** | 0.66 | [0.56, 0.77] | 13.80 | 0.65 |

Gateway − model: leakage -0.04 pt [-0.08, -0.02], over-redaction +0.09 pt [0.04, 0.14].

### abcd (1002 documents)

| System | Char leakage (%) | 95% CI | Doc leakage (%) | 95% CI | Over-redaction (%) | 95% CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| **M5, model alone** | 2.55 | [2.04, 3.15] | 6.27 | [5.03, 7.58] | 48.40 | [46.70, 50.17] |
| gliner_nvidia | 0.26 | [0.05, 0.47] | 0.76 | [0.13, 1.40] | 32.36 | [30.81, 33.94] |
| openmed | 0.73 | [0.48, 1.04] | 4.57 | [3.22, 6.11] | 42.45 | [41.13, 43.87] |
| presidio | 8.76 | [7.59, 9.93] | 23.25 | [20.43, 26.26] | 59.64 | [58.27, 61.12] |
| validators | 74.22 | [72.43, 76.20] | 99.87 | [99.61, 100.00] | 44.01 | [40.98, 46.86] |

Seed leakage (13, 42, 3407): 2.98, 1.55, 3.13.

M5 leakage by label (%, seed mean): BUILDINGNUM 5.8, CITY 11.6, EMAIL 0.5, GIVENNAME 1.3, STREET 6.2, SURNAME 1.3, TELEPHONENUM 0.9, ZIPCODE 15.2.

| M5 − baseline | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI |
| --- | ---: | --- | ---: | --- |
| gliner_nvidia | +2.30 | [1.74, 2.94] | +16.04 | [14.60, 17.45] |
| openmed | +1.82 | [1.20, 2.45] | +5.95 | [4.48, 7.28] |
| presidio | -6.21 | [-7.46, -5.04] | -11.24 | [-12.25, -10.23] |
| validators | -71.66 | [-73.77, -69.74] | +4.39 | [1.91, 6.90] |

Gateway label scope (IBAN, IPADDRESS counted):

| System | Char leakage (%) | 95% CI | Doc leakage (%) | Over-redaction (%) |
| --- | ---: | --- | ---: | ---: |
| M5 model alone | 2.55 | [2.04, 3.15] | 6.27 | 48.40 |
| **M5 + validators (gateway)** | 2.34 | [1.83, 2.92] | 5.89 | 51.40 |

Gateway − model: leakage -0.21 pt [-0.40, -0.07], over-redaction +3.00 pt [2.69, 3.34].

### nemotron (3000 documents)

| System | Char leakage (%) | 95% CI | Doc leakage (%) | 95% CI | Over-redaction (%) | 95% CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| **M5, model alone** | 2.64 | [2.25, 3.09] | 8.58 | [7.72, 9.49] | 3.51 | [3.07, 3.96] |
| gliner_nvidia | 5.02 | [4.43, 5.65] | 11.81 | [10.63, 13.03] | 11.85 | [10.95, 12.76] |
| openmed | 1.33 | [1.02, 1.66] | 6.83 | [5.90, 7.72] | 4.22 | [3.82, 4.69] |
| presidio | 12.87 | [12.11, 13.62] | 43.17 | [41.28, 44.99] | 31.64 | [30.35, 32.86] |
| validators | 63.69 | [62.30, 64.96] | 93.41 | [92.48, 94.33] | 0.74 | [0.44, 1.10] |

Seed leakage (13, 42, 3407): 2.52, 2.67, 2.73.

M5 leakage by label (%, seed mean): ADDRESS 1.0, AGE 7.3, CITY 8.8, CREDITCARDNUMBER 3.0, DATE 5.2, DRIVERLICENSENUM 4.1, EMAIL 0.9, GENDER 24.1, GIVENNAME 1.3, SOCIALNUM 0.4, SURNAME 1.1, TAXNUM 9.2, TELEPHONENUM 0.2, ZIPCODE 3.9.

| M5 − baseline | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI |
| --- | ---: | --- | ---: | --- |
| gliner_nvidia | -2.38 | [-3.00, -1.79] | -8.34 | [-9.27, -7.43] |
| openmed | +1.32 | [1.03, 1.64] | -0.72 | [-1.30, -0.19] |
| presidio | -10.23 | [-10.96, -9.48] | -28.13 | [-29.26, -26.93] |
| validators | -61.05 | [-62.33, -59.75] | +2.77 | [2.22, 3.35] |

Gateway label scope (IBAN, IPADDRESS counted):

| System | Char leakage (%) | 95% CI | Doc leakage (%) | Over-redaction (%) |
| --- | ---: | --- | ---: | ---: |
| M5 model alone | 2.64 | [2.25, 3.09] | 8.58 | 3.51 |
| **M5 + validators (gateway)** | 2.41 | [2.02, 2.84] | 8.19 | 3.70 |

Gateway − model: leakage -0.23 pt [-0.33, -0.15], over-redaction +0.19 pt [0.10, 0.31].

### gretel_en (1000 documents)

| System | Char leakage (%) | 95% CI | Doc leakage (%) | 95% CI | Over-redaction (%) | 95% CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| **M5, model alone** | 11.43 | [10.13, 12.83] | 85.16 | [82.76, 87.31] | 9.88 | [8.70, 10.99] |
| gliner_nvidia | 25.51 | [23.90, 27.14] | 94.59 | [93.05, 96.01] | 35.70 | [33.49, 37.93] |
| openmed | 29.23 | [27.45, 31.33] | 93.80 | [92.19, 95.30] | 34.39 | [32.19, 36.73] |
| presidio | 29.05 | [27.65, 30.69] | 85.23 | [82.77, 87.71] | 58.50 | [56.47, 60.63] |
| validators | 88.52 | [86.97, 90.07] | 99.55 | [98.99, 99.89] | 23.94 | [19.38, 29.02] |

Seed leakage (13, 42, 3407): 11.37, 11.86, 11.05.

M5 leakage by label (%, seed mean): ADDRESS 8.4, CREDITCARDNUMBER 27.5, DATE 13.1, DRIVERLICENSENUM 13.7, EMAIL 3.2, GIVENNAME 2.6, NAME 16.1, PASSPORTNUM 10.4, SOCIALNUM 5.7, SURNAME 41.7, TELEPHONENUM 4.9.

| M5 − baseline | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI |
| --- | ---: | --- | ---: | --- |
| gliner_nvidia | -14.08 | [-15.76, -12.45] | -25.82 | [-28.00, -23.81] |
| openmed | -17.80 | [-19.58, -16.09] | -24.51 | [-26.79, -22.54] |
| presidio | -17.62 | [-19.23, -16.26] | -48.62 | [-50.79, -46.50] |
| validators | -77.09 | [-78.92, -75.28] | -14.06 | [-19.29, -9.48] |

Gateway label scope (IBAN, IPADDRESS counted):

| System | Char leakage (%) | 95% CI | Doc leakage (%) | Over-redaction (%) |
| --- | ---: | --- | ---: | ---: |
| M5 model alone | 11.43 | [10.13, 12.83] | 85.16 | 9.88 |
| **M5 + validators (gateway)** | 11.24 | [9.94, 12.67] | 85.08 | 12.70 |

Gateway − model: leakage -0.19 pt [-0.29, -0.09], over-redaction +2.82 pt [2.18, 3.54].

### Live gateway (M5 + validators, one A40, single requests)

100 support_desk_val messages, HTTP {'200': 100}: latency p50 0.38 s, p95 1.08 s per request; restore exact 100/100.
