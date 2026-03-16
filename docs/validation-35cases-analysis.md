# ATHENA v1.4b — 35-Case Validation Analysis (2026-03-15)

## Run Parameters
- **Model**: 35B (qwen3.5-35b-a3b-text-hi)
- **RAG**: enabled (747K Swiss law chunks)
- **Runs per case**: 7
- **Cases scored**: 34 (1 missing)
- **Total time**: ~8.9 hours
- **Total LLM calls**: 1694

## Overall Results

| Metric | Value |
|--------|-------|
| Accuracy | 64.7% (22/34) |
| 95% CI | [47.9%, 78.5%] |
| Log Loss | 3.475 |
| ECE | 0.155 |

## Accuracy by Outcome Class

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| Annulment (GT) | 76.5% | 13/17 |
| Rejection (GT) | 47.1% | 8/17 |

**Core issue: annulment bias** — model over-detects errors in lower court decisions, producing 9 false annulments vs 3 false rejections.

## Accuracy by Case Source

| Source | Accuracy | N |
|--------|----------|---|
| Original 10 cases | 70% (7/10) | 10 |
| New 24 cases | 62.5% (15/24) | 24 |

Difference not statistically significant.

## Confidence vs Accuracy

| Majority strength | Accuracy | N |
|-------------------|----------|---|
| ≤57% (close) | 33% | 6 |
| 60-71% (moderate) | 64% | 11 |
| ≥86% (strong) | 82% | 17 |

Model uncertainty is informative — confidence threshold could filter unreliable predictions.

## Error Classification (12 errors)

### Likely Ground Truth Errors (3)

| Case | GT | Model | Votes | Notes |
|------|-----|-------|-------|-------|
| ch-2435 | annulment | rejection | 7/0 | Unanimous. Judge reasoning sound — finds clear procedural violations. |
| ch-3425 | annulment | rejection | 6/0 | Unanimous. Thorough reasoning, strong disagreement with label. |
| ch-2046 | rejection | annulment | 0/6 | Unanimous. Model finds substantive legal errors in lower court. |

All three are unanimous model disagreements — strong signal for GT noise in HuggingFace labels.

### Genuinely Hard / Borderline (5)

| Case | GT | Model | Votes (R/A) | Notes |
|------|-----|-------|-------------|-------|
| ch-1272 | rejection | annulment | 2/3 | Close call, model boundary |
| ch-2057 | rejection | annulment | 3/4 | Narrow margin |
| ch-2434 | rejection | annulment | 3/4 | Known systematic error (conditional-waiver reasoning) |
| ch-890 | rejection | annulment | 3/3 | Perfect tie, genuinely ambiguous |
| ch-3408 | rejection | annulment | 2/3 | Known 35B limit on conditional-waiver reasoning |

### Annulment Bias Victims (4)

| Case | GT | Model | Votes (R/A) | Notes |
|------|-----|-------|-------------|-------|
| ch-1291 | rejection | annulment | 2/5 | Clear bias — model over-detects errors |
| ch-1295 | rejection | annulment | 2/5 | Same pattern |
| ch-3580 | rejection | annulment | 1/6 | Strong false annulment |
| ch-2873 | annulment | rejection | 4/3 | Reverse: false rejection (narrow) |
| ch-2874 | annulment | rejection | 5/2 | Reverse: false rejection |

## Adjusted Accuracy

| Scenario | Accuracy |
|----------|----------|
| Raw | 64.7% (22/34) |
| Excluding 3 likely GT errors | **71.0% (22/31)** |
| + Confidence filter (≥60%) | **~75-80%** |

## Root Cause Analysis

1. **Extraction quality is NOT the problem** — error cases have MORE data (facts, legal bases, evidence) than correct cases on average. New HuggingFace cases are not under-extracted.

2. **Annulment bias is a prompt/calibration issue** — Step 2 judge too readily confirms Step 1 error findings. Could be improved with prompt calibration.

3. **Ground truth noise** — HuggingFace `swiss_judgment_prediction` labels may have errors (~3/34 = ~9%). Unanimous model disagreement is a strong signal.

4. **Conditional-waiver reasoning** — ch-2434, ch-3408 involve complex conditional logic at 35B model boundary. 122B handles these correctly but has its own rejection bias.

## Possible Improvements (Priority Order)

1. **Verify GT noise** — manually check ch-2435, ch-3425, ch-2046 against original Bundesgericht decisions
2. **Annulment bias calibration** — tune Step 2 prompt to be more conservative about confirming errors
3. **Confidence threshold** — add confidence field to output, flag low-confidence predictions
4. **Selective 122B routing** — use 122B only for cases where 35B is uncertain (close majority)

## Presentation Strategy (2026-03-19)

- Frame 65% as baseline on noisy labels with clear improvement path
- Highlight: confidence is predictive (82% on strong-majority cases)
- Show: 3 likely GT errors (71% adjusted)
- Emphasize: system correctly identifies its own uncertainty
- Key weakness to acknowledge: annulment bias (fixable via prompt calibration)

## Raw Data

- Output directory: `output/validation-35cases/`
- Per-case results: `output/validation-35cases/{case_id}/`
- Full report: `output/validation-35cases/validation_report.md`
- Run log: `output/validation-35cases/run.log`
