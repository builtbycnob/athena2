# ATHENA2 — Exploratory Data Analysis Report

Generated: 2026-03-15 15:48


============================================================
Dataset: rcds/swiss_judgment_prediction_xl
============================================================
Total rows: 328,825
Splits: {'train': 197432, 'validation': 37485, 'test': 93908}
Languages: {'de': 160292, 'fr': 128008, 'it': 40525}
Labels: {'unknown_dismissal': 228822, 'unknown_approval': 100003}
  unknown_dismissal: 228,822 (69.6%)
  unknown_approval: 100,003 (30.4%)
Law areas: {None: 90516, 'social_law': 89413, 'public_law': 80907, 'civil_law': 37756, 'penal_law': 30233}
Year range: 1954–2022
Facts length (chars): avg=6265, median=4019, p95=18702
Considerations length (chars): avg=18071, median=14531, p95=42896


### Language Distribution — rcds/swiss_judgment_prediction_xl
| Language | Count | Percentage |
|----------|-------|------------|
| de | 160,292 | 48.7% |
| fr | 128,008 | 38.9% |
| it | 40,525 | 12.3% |

### Law Area Distribution — rcds/swiss_judgment_prediction_xl
| Law Area | Count | Percentage |
|----------|-------|------------|
| None | 90,516 | 27.5% |
| social_law | 89,413 | 27.2% |
| public_law | 80,907 | 24.6% |
| civil_law | 37,756 | 11.5% |
| penal_law | 30,233 | 9.2% |

### Label Distribution — rcds/swiss_judgment_prediction_xl
| Label | Count | Percentage |
|-------|-------|------------|
| unknown_dismissal | 228,822 | 69.6% |
| unknown_approval | 100,003 | 30.4% |

### Text Length Distribution — rcds/swiss_judgment_prediction_xl
- Facts: min=385, p25=2,460, median=4,019, p75=7,175, p95=18,702, max=338,568
- Considerations: min=39, p25=9,192, median=14,531, p75=22,771, p95=42,896, max=1,190,805

============================================================
Dataset: rcds/swiss_judgment_prediction
============================================================
Total rows: 264,383
Splits: {'train': 238818, 'validation': 8208, 'test': 17357}
Languages: {'de': 74139, 'fr': 69618, 'it': 60923, 'en': 59703}
Labels: {'dismissal': 202613, 'approval': 61770}
  dismissal: 202,613 (76.6%)
  approval: 61,770 (23.4%)
Law areas: {'unknown': 264383}
Year range: 2000–2020
Facts length (chars): avg=3365, median=2750, p95=7752


### Language Distribution — rcds/swiss_judgment_prediction
| Language | Count | Percentage |
|----------|-------|------------|
| de | 74,139 | 28.0% |
| fr | 69,618 | 26.3% |
| it | 60,923 | 23.0% |
| en | 59,703 | 22.6% |

### Law Area Distribution — rcds/swiss_judgment_prediction
| Law Area | Count | Percentage |
|----------|-------|------------|
| unknown | 264,383 | 100.0% |

### Label Distribution — rcds/swiss_judgment_prediction
| Label | Count | Percentage |
|-------|-------|------------|
| dismissal | 202,613 | 76.6% |
| approval | 61,770 | 23.4% |

### Text Length Distribution — rcds/swiss_judgment_prediction
- Facts: min=7, p25=1,815, median=2,750, p75=4,168, p95=7,752, max=78,751

============================================================
Dataset: rcds/swiss_criticality_prediction
============================================================
Total rows: 138,531
Splits: {'train': 91075, 'validation': 14837, 'test': 32619}
Languages: {'de': 85167, 'fr': 45451, 'it': 7913}
Labels: {'bge=non-critical': 134440, 'bge=critical': 4091}
  bge=non-critical: 134,440 (97.0%)
  bge=critical: 4,091 (3.0%)
Year range: 2002–2022


### Language Distribution — rcds/swiss_criticality_prediction
| Language | Count | Percentage |
|----------|-------|------------|
| de | 85,167 | 61.5% |
| fr | 45,451 | 32.8% |
| it | 7,913 | 5.7% |

### Label Distribution — rcds/swiss_criticality_prediction
| Label | Count | Percentage |
|-------|-------|------------|
| bge=non-critical | 134,440 | 97.0% |
| bge=critical | 4,091 | 3.0% |

============================================================
Dataset: rcds/swiss_citation_extraction
============================================================
Total rows: 127,483
Splits: {'train': 87760, 'validation': 12359, 'test': 27364}
Languages: {'de': 83082, 'fr': 36326, 'it': 8075}
Year range: 1996–2022


### Language Distribution — rcds/swiss_citation_extraction
| Language | Count | Percentage |
|----------|-------|------------|
| de | 83,082 | 65.2% |
| fr | 36,326 | 28.5% |
| it | 8,075 | 6.3% |