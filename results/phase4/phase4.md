## Phase 4: final blind evaluation

M5 = Qwen3-1.7B + QLoRA r=16, lr 4e-4, train_mix_32k; seeds 13, 42, 3407 (per-document counts averaged over seeds). 95% CIs: 1000 document resamples. Each test set was scored once per system.

### tab (127 documents)

| System | Char leakage (%) | 95% CI | Doc leakage (%) | 95% CI | Over-redaction (%) | 95% CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| **M5, model alone** | 15.91 | [13.92, 18.42] | 100.00 | [100.00, 100.00] | 5.24 | [4.30, 6.32] |
| gliner_nvidia | 19.29 | [17.02, 21.72] | 100.00 | [100.00, 100.00] | 17.55 | [15.18, 20.10] |
| openmed | 14.91 | [13.43, 16.44] | 100.00 | [100.00, 100.00] | 6.75 | [5.80, 7.81] |
| presidio | 11.23 | [9.32, 13.70] | 100.00 | [100.00, 100.00] | 34.12 | [31.74, 36.36] |
| validators | 99.95 | [99.88, 100.00] | 100.00 | [100.00, 100.00] | 0.00 | [0.00, 0.00] |

Seed leakage (13, 42, 3407): 15.05, 17.52, 15.17.

M5 leakage by label (%, seed mean): CODE 94.3, DATE 5.3, LOC 58.0, NAME 16.8.

| M5 − baseline | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI |
| --- | ---: | --- | ---: | --- |
| gliner_nvidia | -3.38 | [-4.54, -2.22] | -12.32 | [-15.02, -9.77] |
| openmed | +1.01 | [-0.85, 3.31] | -1.51 | [-2.71, -0.25] |
| presidio | +4.69 | [2.92, 6.46] | -28.88 | [-31.02, -26.77] |
| validators | -84.04 | [-86.04, -81.55] | +5.24 | [4.30, 6.32] |

Gateway label scope (IBAN, IPADDRESS counted):

| System | Char leakage (%) | 95% CI | Doc leakage (%) | Over-redaction (%) |
| --- | ---: | --- | ---: | ---: |
| M5 model alone | 15.91 | [13.92, 18.42] | 100.00 | 5.24 |
| **M5 + validators (gateway)** | 15.87 | [13.86, 18.39] | 100.00 | 5.24 |

Gateway − model: leakage -0.04 pt [-0.10, 0.00], over-redaction -0.00 pt [-0.01, 0.00].

### support_desk_300 (300 documents)

| System | Char leakage (%) | 95% CI | Doc leakage (%) | 95% CI | Over-redaction (%) | 95% CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| **M5, model alone** | 1.70 | [0.82, 2.78] | 4.58 | [2.70, 7.15] | 5.75 | [4.02, 7.77] |
| gliner_nvidia | 8.11 | [5.18, 11.03] | 16.59 | [11.68, 22.01] | 23.23 | [20.13, 26.40] |
| openmed | 3.54 | [2.09, 5.27] | 16.11 | [11.00, 21.17] | 16.05 | [13.47, 18.65] |
| presidio | 18.91 | [15.23, 22.37] | 45.02 | [38.28, 51.42] | 23.10 | [19.68, 26.37] |
| validators | 69.31 | [64.17, 74.46] | 87.68 | [83.09, 91.98] | 1.32 | [0.00, 4.02] |

Seed leakage (13, 42, 3407): 0.68, 2.22, 2.20.

M5 leakage by label (%, seed mean): AGE 1.8, BUILDINGNUM 1.3, CITY 5.7, CREDITCARDNUMBER 3.4, DATE 0.0, DRIVERLICENSENUM 0.0, EMAIL 2.7, GENDER 5.6, GIVENNAME 0.1, IDCARDNUM 0.0, PASSPORTNUM 0.0, SEX 16.7, SOCIALNUM 0.0, STREET 0.5, SURNAME 1.1, TAXNUM 32.3, TELEPHONENUM 0.0, TITLE 0.0, ZIPCODE 3.5.

| M5 − baseline | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI |
| --- | ---: | --- | ---: | --- |
| gliner_nvidia | -6.41 | [-9.46, -3.39] | -17.48 | [-20.77, -14.49] |
| openmed | -1.84 | [-3.76, -0.03] | -10.30 | [-12.82, -7.92] |
| presidio | -17.21 | [-20.78, -13.36] | -17.34 | [-20.71, -14.29] |
| validators | -67.61 | [-72.93, -62.37] | +4.43 | [1.91, 6.94] |

Gateway label scope (IBAN, IPADDRESS counted):

| System | Char leakage (%) | 95% CI | Doc leakage (%) | Over-redaction (%) |
| --- | ---: | --- | ---: | ---: |
| M5 model alone | 4.21 | [2.12, 6.53] | 7.51 | 5.74 |
| **M5 + validators (gateway)** | 1.25 | [0.50, 2.14] | 4.23 | 5.72 |

Gateway − model: leakage -2.96 pt [-5.30, -1.02], over-redaction -0.02 pt [-0.26, 0.24].

### test_id (5000 documents)

| System | Char leakage (%) | 95% CI | Doc leakage (%) | 95% CI | Over-redaction (%) | 95% CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| **M5, model alone** | 0.60 | [0.54, 0.69] | 13.84 | [13.00, 14.77] | 0.48 | [0.42, 0.55] |
| validators | 83.68 | [83.33, 84.03] | 99.10 | [98.82, 99.36] | 0.96 | [0.69, 1.27] |

Seed leakage (13, 42, 3407): 0.61, 0.61, 0.60.

M5 leakage by label (%, seed mean): AGE 8.4, BUILDINGNUM 0.8, CITY 1.1, CREDITCARDNUMBER 0.7, DATE 0.1, DRIVERLICENSENUM 0.9, EMAIL 0.5, GENDER 0.2, GIVENNAME 0.9, IDCARDNUM 0.3, PASSPORTNUM 0.0, SEX 0.8, SOCIALNUM 0.1, STREET 0.4, SURNAME 1.0, TAXNUM 0.4, TELEPHONENUM 0.1, TITLE 0.5, ZIPCODE 0.2.

| M5 − baseline | Leakage Δ (pt) | 95% CI | Over-redaction Δ (pt) | 95% CI |
| --- | ---: | --- | ---: | --- |
| validators | -83.08 | [-83.44, -82.72] | -0.48 | [-0.77, -0.21] |

Gateway label scope (IBAN, IPADDRESS counted):

| System | Char leakage (%) | 95% CI | Doc leakage (%) | Over-redaction (%) |
| --- | ---: | --- | ---: | ---: |
| M5 model alone | 0.60 | [0.54, 0.69] | 13.84 | 0.48 |
| **M5 + validators (gateway)** | 0.55 | [0.49, 0.63] | 13.70 | 0.59 |

Gateway − model: leakage -0.05 pt [-0.08, -0.03], over-redaction +0.11 pt [0.08, 0.14].

### Live gateway (M5 + validators, one A40, single requests)

100 support_desk_val messages, HTTP {'200': 100}: latency p50 0.38 s, p95 1.08 s per request; restore exact 100/100.
