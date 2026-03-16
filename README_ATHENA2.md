# ATHENA2 — Legal Judgment World Model

A hybrid system that combines a data-driven world model trained on 329K Swiss Federal Supreme Court decisions with LLM-based adversarial simulation, formal game theory, and calibrated uncertainty.

## What Makes This Different

Existing legal prediction systems are classifiers: facts in → label out. ATHENA2 is a **world model**: it learns the dynamics of legal reasoning, not just the outcomes.

| Capability | Published SOTA | ATHENA v1 | ATHENA2 |
|------------|---------------|-----------|---------|
| Accuracy | ~71% macro F1 | 65% (LLM) | Target: 75%+ |
| Calibration | None | ECE 0.155 | Conformal prediction |
| Reasoning model | No | LLM-generated | Learned from 329K cases |
| Game theory | No | BATNA, Nash, ZOPA | Enhanced |
| Counterfactual | No | No | Yes |
| Speed (per case) | N/A | ~15 min | <1s |

## Architecture

```
Facts text
    ↓
[Legal Encoder] (Legal-XLM-R Large, 340M)
    ↓
[Feature Heads] (multi-task: law area, error presence, reasoning pattern)
    ↓
[Dynamics Module] (MLP: encoding + features → verdict distribution)
    ↓
[Calibration] (focal loss + temperature scaling + conformal prediction)
    ↓
P(dismissal), P(approval), confidence set, reasoning chain
```

## Setup

```bash
# Core (metrics, features, data pipeline)
uv pip install -e ".[dev]"

# World model training (requires PyTorch)
uv pip install -e ".[worldmodel]"

# Full stack (includes ATHENA v1 features)
uv pip install -e ".[dev,worldmodel,rag,validation,api]"
```

## Phase 1: Data Foundation

```bash
# Download and process all datasets (~30 min)
uv run python scripts/phase1_data_foundation.py

# Just statistics (no download)
uv run python scripts/phase1_data_foundation.py --stats-only

# Individual steps
uv run python scripts/phase1_data_foundation.py --step ingest
uv run python scripts/phase1_data_foundation.py --step features
uv run python scripts/phase1_data_foundation.py --step citation
```

## Phase 2: Baselines (coming next)

```bash
# TF-IDF + Logistic Regression baseline
uv run python scripts/phase2_baselines.py --model tfidf

# Fine-tuned transformer baseline
uv run python scripts/phase2_baselines.py --model bert
```

## Tests

```bash
# ATHENA2 tests only
uv run pytest tests/test_athena2_metrics.py tests/test_athena2_features.py -v

# All tests (ATHENA v1 + ATHENA2)
uv run pytest tests/ -q
```

## Documentation

- `ARCHITECTURE_REVIEW.md` — Deep analysis of ATHENA v1 codebase
- `RESEARCH.md` — Comprehensive literature review (2000+ words)
- `DECISIONS.md` — Architectural decisions with trade-offs
- `configs/data.yaml` — Data pipeline configuration
- `configs/training.yaml` — Training hyperparameters

## Project Structure

```
src/athena2/
├── data/           # Ingestion pipeline (HuggingFace → Parquet)
├── features/       # Feature extraction (regex + LLM + citation graph)
├── models/         # Baselines + world model
├── calibration/    # Temperature scaling, conformal prediction
├── simulation/     # Monte Carlo on world model
└── evaluation/     # Metrics (F1, ECE, Brier, conformal)
```

## Research Gaps Addressed

1. **No world model for legal proceedings** — first system to model legal reasoning as dynamics
2. **No formal calibration** — first conformal prediction for legal judgment
3. **No game-theoretic legal AI** — prediction feeds into BATNA/Nash/ZOPA
4. **No hybrid simulation + data-driven** — world model + LLM + game theory
5. **No counterfactual legal reasoning** — latent perturbation → compare outcomes
