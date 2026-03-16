# ATHENA2 — Architectural Decisions Log

Every non-trivial architectural choice is documented here with context, options, decision, reasoning, and trade-offs.

---

## ADR-001: World Model Architecture

**Date**: 2026-03-15
**Status**: Decided
**Context**: Need to choose the core world model architecture for learning legal reasoning dynamics from 329K Swiss Federal Supreme Court cases.

### Options Considered

**A. Transformer encoder-decoder (seq2seq)**
- Encode facts → decode considerations → predict verdict
- Pro: Natural fit for text generation, can produce reasoning chains
- Con: Slow inference (autoregressive), hard to do fast Monte Carlo

**B. Encoder + classification head (BERT-style)**
- Encode facts → classify verdict
- Pro: Fast inference, well-studied, published baselines exist
- Con: No reasoning modeling, no counterfactuals, just a classifier

**C. Latent dynamics model (DreamerV3-inspired)**
- Encode facts → latent state → learned transition dynamics → predict verdict distribution
- Pro: Fast forward pass (neural net, not autoregressive), counterfactual queries via latent perturbation, principled uncertainty
- Con: More complex to train, novel application to legal domain, no published precedent

**D. Hybrid: encoder + structured intermediate predictions**
- Encode facts → predict intermediate legal features (error types, applicable law, severity) → predict verdict
- Pro: Combines learned representations with structured legal reasoning, explainable intermediate states
- Con: Requires feature extraction pipeline for supervision signal

### Decision

**D (Hybrid) with elements of C.** Multi-stage architecture:

1. **Legal encoder** (Legal-XLM-R Large, 340M) encodes facts into dense representation
2. **Feature extraction heads** predict intermediate legal features from the encoding:
   - Law area classification (4 classes)
   - Applicable law articles (multi-label)
   - Error presence and severity (extracted from considerations)
   - Case complexity / criticality
3. **Dynamics module** takes encoded facts + extracted features → predicts verdict probability distribution
4. **Calibration layer** applies temperature scaling + conformal prediction

### Reasoning

- Option B is what everyone has done — it's the ceiling we need to beat, not our target
- Option C (pure latent dynamics) is elegant but risky for a production system with a 4-day deadline for initial validation
- Option A is too slow for 10K simulations
- Option D gives us explainable intermediate states (critical for lawyers), fast inference, AND a path to counterfactual reasoning by perturbing intermediate features

### Trade-offs

- More complex training pipeline (multi-stage vs end-to-end)
- Feature extraction from considerations requires LLM-assisted labeling (one-time cost)
- Harder to optimize end-to-end (but multi-task learning mitigates this)

---

## ADR-002: Training Infrastructure — MLX vs PyTorch

**Date**: 2026-03-15
**Status**: Decided

### Options

**A. MLX-only**
- Pro: Native Apple Silicon, 24-134% faster than MPS, unified memory
- Con: Smaller ecosystem, fewer examples, custom training loops needed for some architectures

**B. PyTorch with MPS backend**
- Pro: Mature ecosystem, HuggingFace Trainer integration, most examples available
- Con: MPS backend still has gaps, slower than MLX, memory management less efficient

**C. Hybrid: PyTorch for training, MLX for inference**
- Pro: Best of both — mature training, fast inference
- Con: Two codepaths to maintain, model conversion step

### Decision

**C (Hybrid)**. Use PyTorch + HuggingFace Trainer for initial training (widest model compatibility, Trainer handles distributed, logging, checkpointing). Convert to MLX for production inference.

### Reasoning

- HuggingFace Trainer + MPS backend can fine-tune Legal-XLM-R Large on M3 Ultra
- MLX inference is significantly faster (critical for 10K simulation target)
- Model conversion is well-supported (mlx-lm, coremltools)
- Training is a one-time cost; inference performance is ongoing

---

## ADR-003: Embedding Strategy

**Date**: 2026-03-15
**Status**: Decided

### Options

**A. Keep BGE-M3 (current ATHENA v1)**
- Pro: Already deployed, 120 text/s, multilingual, no training needed
- Con: Not legal-domain specific, may miss legal semantic nuances

**B. Fine-tune Legal-XLM-R on Swiss corpus**
- Pro: Domain-specific, can use contrastive learning on case similarity
- Con: Requires training, ~340M params

**C. Use both (cascade)**
- Pro: BGE-M3 for RAG retrieval (fast, good enough), Legal-XLM-R for case representation (high quality)
- Con: Two models in memory

### Decision

**C (Cascade)**. BGE-M3 continues as the RAG embedder (proven, fast). Legal-XLM-R Large serves as the case encoder for the world model (higher quality legal representations).

### Reasoning

- RAG retrieval needs speed and breadth — BGE-M3 excels here
- Case representation needs legal semantic precision — Legal-XLM-R trained on 689GB legal text
- Memory budget: BGE-M3 (~568M) + Legal-XLM-R Large (~340M) = ~2GB total — negligible on 192GB

---

## ADR-004: Calibration Strategy

**Date**: 2026-03-15
**Status**: Decided

### Decision

Three-layer calibration:

1. **Training-time**: Focal loss (reduces overconfidence on easy examples)
2. **Post-hoc**: Temperature scaling (learned on validation set)
3. **Inference-time**: Conformal prediction (distribution-free coverage guarantees)

Plus: Brier score decomposition (reliability + resolution + uncertainty) for evaluation.

### Reasoning

- Focal loss is nearly free (just a loss function change) and acts as implicit calibration
- Temperature scaling is the simplest effective post-hoc method
- Conformal prediction provides formal guarantees — unprecedented in legal AI
- Together they address calibration at every stage of the pipeline

---

## ADR-005: Considerations Field Usage

**Date**: 2026-03-15
**Status**: Decided

### Context

The SJP-XL dataset includes the `considerations` field — full judge reasoning. No published model uses this because it constitutes data leakage for standard classification (the reasoning reveals the verdict).

### Decision

Use considerations as **supervision signal for intermediate features**, not as input at inference time.

**Training**: Extract structured features from considerations (error types, severity, applicable law, reasoning patterns) using LLM. Train the world model to predict these intermediate features from facts alone.

**Inference**: Model receives only facts (like a real lawyer would). It predicts the intermediate reasoning features, then predicts the verdict. The intermediate features serve as explainable reasoning chain.

### Reasoning

- This is the key architectural insight that differentiates ATHENA2
- Considerations encode HOW judges reason, not just what they decide
- By extracting structured features and training the model to predict them, we teach the model to reason like a judge
- No data leakage: at inference, the model only sees facts
- Explainability for free: the predicted intermediate features ARE the explanation

---

## ADR-006: Dataset Strategy

**Date**: 2026-03-15
**Status**: Decided

### Decision

| Dataset | Role |
|---------|------|
| SJP-XL (329K) | Primary training (facts + considerations → features + verdict) |
| SJP (85K) | Benchmark comparison (published baselines use this) |
| Swiss Criticality (139K) | Auxiliary training signal (importance prediction) |
| Swiss Citation Extraction (127K) | Citation graph construction |
| Swiss Law Area (22K) | Auxiliary training signal (law area classification) |
| SCD Zenodo (122K, up to 2024) | Temporal validation (train ≤2020, test 2020-2024) |

### Reasoning

- SJP-XL is the only dataset with considerations — essential for world model training
- SJP provides apples-to-apples comparison with published SOTA
- Auxiliary datasets enable multi-task learning (proven to help in legal NLP)
- SCD enables temporal validation (critical for demonstrating the model doesn't just memorize)

---

## ADR-007: Backward Compatibility with ATHENA v1

**Date**: 2026-03-15
**Status**: Decided

### Decision

ATHENA2 maintains full backward compatibility:
- Reads ATHENA v1 case YAML and simulation YAML formats
- Exposes `run_pipeline()` with same signature
- Outputs same format (probability tables, memos, game theory)
- Adds new capabilities as optional layers on top

### Reasoning

- 36 validated Swiss cases in existing format
- Game theory module is pure computation — reuse entirely
- Scoring framework already handles accuracy/ECE/log loss — extend, don't replace
- Users (law firm) already understand ATHENA v1 outputs

---

## ADR-008: Multi-Language Handling

**Date**: 2026-03-15
**Status**: Decided

### Decision

Train on all three languages jointly (German 58%, French 36%, Italian 5%). Use Legal-XLM-R which is natively multilingual (24 languages). Do NOT translate to a single language.

### Reasoning

- Joint multilingual training improves Italian performance (+4% F1 per Niklaus et al. 2024)
- Legal-XLM-R handles multilingual input natively
- Swiss Federal Supreme Court operates in all three languages — model must too
- Machine translation introduces errors in legal terminology
