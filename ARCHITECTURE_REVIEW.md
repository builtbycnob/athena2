# ATHENA Architecture Review

**Purpose**: Deep analysis of the existing ATHENA v1.4b codebase to inform ATHENA2 design.
**Date**: 2026-03-15
**Codebase**: v1.4b, 532 tests, ~10,500 LOC across `src/athena/`

---

## 1. System Overview

ATHENA is a multi-agent legal simulation system that models adversarial proceedings as a game with incomplete information. It uses LLMs as agent-parties (appellant, respondent, judge), runs Monte Carlo simulations varying strategies and judge profiles, aggregates results statistically, applies game theory (BATNA, Nash bargaining, ZOPA), and generates strategic memos.

The name reflects *intelligent strategy* (Athena the goddess) — not brute force classification, but adversarial simulation.

### Core Design Principle

A legal proceeding is a **multi-agent game with incomplete information**. Each party has private knowledge (case file, evidence visibility), an objective function, and legal constraints. The judge applies a decision function constrained by jurisdiction-specific rules. ATHENA models all of this explicitly.

---

## 2. Directory Structure

```
src/athena/
├── agents/           # LLM integration, prompts, JSON repair (llm.py: 200 LOC)
├── simulation/       # LangGraph orchestration, Monte Carlo, aggregation
│   ├── graph.py      # State machine + phase system (642 LOC) — the heart
│   ├── orchestrator.py  # ThreadPoolExecutor Monte Carlo (251 LOC)
│   ├── aggregator.py    # Wilson CI probability tables (234 LOC)
│   ├── context.py       # Agent context building
│   └── validation.py    # Output validation
├── jurisdiction/     # Multi-jurisdiction: registry, IT, CH
├── schemas/          # Pydantic models + JSON schemas (structured_output.py: 601 LOC)
├── game_theory/      # BATNA, Nash, EV, sensitivity (pure computation)
├── rag/              # BGE-M3 embedder, LanceDB store, retriever
├── knowledge/        # Neo4j KG (optional)
├── api/              # FastAPI + pipeline extraction
├── validation/       # Scorer, dataset fetcher, case extractor
├── output/           # Memo, decision tree, tables
└── cli.py            # Entry point (~150 LOC)
```

---

## 3. Simulation Engine (Core Architecture)

### 3.1 LangGraph State Machine

The simulation is a **compiled LangGraph StateGraph** with typed state:

```python
GraphState = TypedDict(
    case: dict,                    # Immutable case data
    params: dict,                  # Run-specific parameters (temps, models, profiles)
    briefs: dict[pid → brief],     # Accumulates across phases (merge reducer)
    validations: dict[pid → val],
    decision: dict | None,         # Judge verdict
    error: str | None,
)
```

**Phase-based orchestration**: Phases are lists of `AgentConfig` objects. Nodes are dynamically created at graph compilation time. Edges wire sequentially within phases and across phase boundaries.

Standard bilateral flow:
```
Phase "filing"   → appellant node
Phase "response" → respondent node (sees appellant brief)
Phase "decision" → judge node (sees all briefs + all evidence)
```

### 3.2 Agent Types

| Role | Type | Output | Key Fields |
|------|------|--------|------------|
| Appellant | advocate | filed_brief + internal_analysis | arguments, requests, vulnerabilities |
| Respondent | advocate | filed_brief + internal_analysis | objections, responses, counters |
| Judge (IT) | adjudicator | verdict | qualification_correct (bool) |
| Judge (CH) | adjudicator_two_step | Step 1 → Step 2 | errors[] → lower_court_correct |

### 3.3 Two-Step Swiss Judge (Key Innovation)

The most interesting architectural decision in the codebase. Swiss judge uses two LLM calls:

1. **Step 1** (temp=0.7): Error identification. Finds errors in the lower court decision with severity levels: `none`, `minor`, `significant`, `decisive`.
2. **Step 2** (temp=0.4): Outcome decision. Takes Step 1 errors as input, re-evaluates severities, decides `lower_court_correct`.

**Calibration mechanisms**:
- **Severity floor**: Step 2 can't drop `decisive` below `significant`
- **Severity ceiling**: Step 2 can upgrade at most +1 level (prevents false `none→decisive`)
- **Consistency enforcement**: If any error is `decisive` AND `lower_court_correct=True` → force False

This broke the original 93% dismissed-bias (single-step judge always confirmed lower court).

### 3.4 Monte Carlo Orchestrator

```python
combinations = product(judge_profiles, party_profiles, range(runs_per_combination))
# Typical: 1 judge × 1 style × 7 runs = 7 simulations per case
# ThreadPoolExecutor(max_workers=4) for parallel execution
```

Each combination gets its own graph invocation with independent random state (LLM temperature provides stochasticity).

### 3.5 Multi-Model Routing (v1.4b)

Resolution order: `simulation YAML models.{role}` → `JurisdictionConfig.default_models` → `OMLX_MODEL` env var.

Currently 35B default for all roles (122B showed rejection bias in validation).

---

## 4. LLM Integration

### 4.1 Backend

Primary: oMLX HTTP server (OpenAI-compatible API at localhost:8000).
Model: `qwen3.5-35b-a3b-text-hi` (35B MoE, text-only).

```python
payload = {
    "model": model,
    "messages": [system, user],
    "temperature": temperature,
    "max_tokens": max_tokens,
    "repetition_penalty": 1.3,
    "top_p": 0.8,
    "top_k": 20,
    "json_schema": schema,  # XGrammar constrained decoding
}
```

### 4.2 Structured Output

Three-layer defense for JSON reliability:
1. **XGrammar**: Token-level pushdown automaton enforces JSON schema (v1.4)
2. **Dynamic enums**: `schema_builder.py` injects case-specific IDs as enum constraints
3. **JSON repair**: Regex fixes → `json_repair` library → truncation state machine

### 4.3 Throughput

- **35B**: ~18-29 tok/s per call (judge slower due to longer context)
- **122B**: ~20-31 tok/s solo, degrades to 3-7 tok/s with concurrency=4
- Single case (7 runs): ~15 minutes
- 35-case validation: ~9 hours

---

## 5. Statistical Aggregation

### 5.1 Probability Tables

For each (judge_profile, party_style) combination:
- `p_rejection`, `p_annulment` with **Wilson score 95% CI**
- `n_runs` per cell

### 5.2 Argument Effectiveness

Per seed argument:
- `mean_persuasiveness` across runs
- `determinative_rate` (how often judge cited as decisive)
- `by_judge_profile` breakdown

### 5.3 Outcome Detection

Auto-detects jurisdiction from verdict shape:
- `lower_court_correct` → Swiss CH
- `qualification_correct` → Italian IT

---

## 6. Game Theory Module

Pure computational module (no LLM calls):

| Analysis | Function | Output |
|----------|----------|--------|
| BATNA | `compute_batna()` | Expected value of litigation per party |
| Settlement | `compute_settlement_range()` | ZOPA + Nash bargaining solution |
| EV by strategy | Per-strategy ranking | Best strategy identification |
| Sensitivity | `run_sensitivity_analysis()` | Parameter sweeps + tornado ranking |
| Dominated strategies | Statistical from aggregation | Eliminated strategies |

Feeds into Game Theorist Agent (LLM interprets for lawyers).

---

## 7. RAG System

- **Embedder**: BGE-M3 (568M, 120 text/s) or Qwen3-Embedding-4B via MLX
- **Store**: LanceDB (embedded, per-jurisdiction tables)
- **Corpus**: 747,946 chunks from 35,698 Swiss laws
- **Retriever**: Builds queries from seed arguments, hybrid search, dedup, token budget truncation
- **Integration**: Retrieved norms injected into judge context

---

## 8. Validation Framework

### 8.1 Scorer

Computes: accuracy (Wilson CI), log loss, ECE (10-bin), stratification by legal area/year.

### 8.2 Ground Truth

36 Swiss cases with labels from `swiss_judgment_prediction` HuggingFace dataset.
Labels: `rejection` (lower court confirmed) / `annulment` (lower court overturned).

### 8.3 Results (v1.4b, 35-case validation)

| Metric | Value |
|--------|-------|
| Raw accuracy | 64.7% (22/34) |
| Adjusted (excl. GT noise) | ~71% (22/31) |
| Rejection accuracy | 47.1% |
| Annulment accuracy | 76.5% |
| Strong-majority accuracy | 82% |

**Key finding**: Annulment bias. Model over-detects errors in lower court decisions.

---

## 9. Strengths to Preserve in ATHENA2

1. **Multi-agent adversarial simulation** — no published system combines this with game theory
2. **Phase-based graph construction** — extensible to N parties, new agent types
3. **Jurisdiction-aware architecture** — clean abstraction for multi-country support
4. **Two-step judge with calibration** — effective debiasing mechanism
5. **XGrammar constrained decoding** — guarantees valid structured output
6. **Monte Carlo with Wilson CI** — appropriate uncertainty quantification for small samples
7. **Game theory integration** — BATNA/Nash/ZOPA provides actionable strategic output
8. **RAG legal corpus** — 747K Swiss law chunks, hybrid search
9. **Crash-safe validation** — per-case result writing, incremental scoring
10. **Graceful degradation** — KG, RAG, meta-agents all optional

---

## 10. Weaknesses ATHENA2 Must Address

### 10.1 LLM-Dependent Accuracy

ATHENA's predictions are entirely dependent on the quality of the underlying LLM. The 35B model achieves 65% accuracy — comparable to random with confidence weighting. There is no learned model calibrated on actual court decision data.

**ATHENA2 fix**: Train a dedicated world model on 329K Swiss Federal Supreme Court decisions (SJP-XL dataset). The LLM-based simulation becomes one input to a hybrid system, not the sole predictor.

### 10.2 No Learned Representation

ATHENA treats legal text as opaque strings passed to LLMs. There's no embedding space capturing legal semantics, no learned similarity between cases, no representation of how precedents relate.

**ATHENA2 fix**: Train legal embeddings on the Swiss corpus. Build a citation graph. Enable similarity-based retrieval and representation learning.

### 10.3 Binary Outcome Only

Swiss cases have richer outcomes than rejection/annulment: partial approval, remand for reconsideration, inadmissibility, withdrawal. The binary label is a lossy compression.

**ATHENA2 fix**: Extract granular outcomes from the `considerations` field. Model the full outcome distribution.

### 10.4 No Temporal Dynamics

ATHENA treats each case independently. Jurisprudence drifts over time — what was annulled in 2005 might be rejected in 2020. No temporal modeling.

**ATHENA2 fix**: Temporal features (year, era, recent precedent patterns). Evaluate temporal validation (train pre-2020, test 2020+).

### 10.5 Simulation Speed

7 runs × 1 case = ~15 minutes. For strategic use, need 10,000 simulations in under 60 seconds.

**ATHENA2 fix**: World model inference (forward pass through a neural network) instead of 7 sequential LLM calls. 10K forward passes on M3 Ultra GPU: seconds, not hours.

### 10.6 Calibration Gap

ECE of 0.155 is mediocre. No formal calibration pipeline. Confidence is informative but not well-calibrated.

**ATHENA2 fix**: Conformal prediction, temperature scaling, ensemble calibration. Formal calibration pipeline with Brier decomposition.

### 10.7 No Counterfactual Reasoning

ATHENA can vary strategies but can't answer: "what if we had presented argument X instead of Y?" without running a full new simulation.

**ATHENA2 fix**: World model supports counterfactual queries by perturbing the latent state and running the dynamics model forward.

---

## 11. Interfaces to Preserve

ATHENA2 must maintain compatibility with:

1. **Case YAML format** — `cases/validation/*.yaml` structure
2. **Simulation YAML format** — `simulations/*.yaml` structure
3. **Ground truth JSON** — `ground_truth/*.json` format
4. **Pipeline API** — `run_pipeline(case_data, sim_config, options)` signature
5. **Scoring framework** — `score_results(output_dir, gt_dir)` interface
6. **Game theory module** — pure computation, no LLM dependency
7. **Output formats** — probability tables, strategic memos, decision trees

---

## 12. Code Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Type hints | Good | Most functions typed, Pydantic models |
| Error handling | Good | Per-agent error collection, graceful degradation |
| Testing | Good | 532 tests, mocked LLM, integration tests |
| Separation of concerns | Excellent | Pipeline logic vs I/O vs HTTP |
| Documentation | Good | CLAUDE.md is comprehensive, inline docstrings |
| Configuration | Good | YAML-driven, env vars, jurisdiction registry |
| Extensibility | Good | Phase-based, jurisdiction registry, schema builder |
| Performance | Acceptable | Bottleneck is LLM inference, not code |

**Overall**: Well-architected production system. ATHENA2 should preserve the architectural patterns while adding a data-driven world model layer.
