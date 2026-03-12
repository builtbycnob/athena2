# ATHENA — Adversarial Tactical Hearing & Equilibrium Navigation Agent

## What This Is

ATHENA is a multi-agent system for legal case preparation that simulates adversarial proceedings using LLMs as opposing parties, applies game theory to estimate outcomes, and generates strategic recommendations.

The name comes from the Greek goddess of *intelligent strategy* (not brute force).

## Core Concept

A legal proceeding is inherently a multi-agent game with incomplete information. ATHENA models it as such:
- **N configurable agent-parties** (plaintiff, defendant, third parties, insurer, co-defendants...)
- **Structural agents** (judge/arbitrator, expert witness) that model the decision-making environment
- **Meta-agents** (game theorist, red teamer, synthesizer) that analyze and recommend
- Each agent has a **private case file**, **objectives**, **legal constraints**, and a **belief model** about what the other parties know

## Three Operating Modes

1. **Wargame** (simulation) — agents play out the case across N simulations varying strategies and information. Output: vulnerability map, leverage points, scenario distribution
2. **Strategist** (semi-autonomous) — ranks strategies with rationale, estimates settlement ranges via Nash equilibrium, identifies dominated strategies
3. **Sparring** (interactive) — human plays their side, LLM agents respond as counterparties in real-time

## Game Theory Layer

Implemented:
- Dominated strategy identification (v0.4 — statistical, from Monte Carlo aggregation)
- Outcome valuation: verdict→EUR mapping per party perspective (v0.6)
- BATNA analysis: per-party expected value of litigation with CI ranges (v0.6)
- Nash bargaining: bilateral settlement range (ZOPA + Nash solution) (v0.6)
- EV by strategy: per-strategy expected value ranking (v0.6)
- Sensitivity analysis: parameter sweeps + tornado ranking (v0.6)
- Game Theorist Agent: LLM interprets computational GT output for lawyers (v0.8)

Future:
- Extensive form game trees (sequential moves: filing → response → discovery → motions → trial)
- Bayesian updating, mixed strategies, signaling games, multi-stage bargaining

## Architecture Decisions

- **Standalone project**, not inside ARGUS/OpenClaw — different domain, different users
- **Same infrastructure**: MLX on Mac Studio, LangGraph orchestration, Langfuse observability
- **Multi-jurisdiction**: parameterizable per jurisdiction (procedural rules, legal standards)
- **Local-first**: case files contain privileged information → local inference is a feature, not a constraint
- **N-party configurable**: works for simple bilateral disputes and complex multi-party arbitration

## Agent Types

### Party Agents (one per side)
- Private case file (facts, evidence, documents only they know)
- Explicit objective function (maximize recovery, minimize damages, etc.)
- Legal constraints (applicable law, procedural rules)
- Belief model (Bayesian prior on opponent's evidence and strategy)

### Structural Agents
- **Judge/Arbitrator** — evaluates arguments against legal standards, simulates judicial decision-making, calibrated on jurisdiction-specific case law
- **Expert/CTU** — simulates technical expert evaluation of evidence

### Meta-Agents
- **Game Theorist** — formal modeling, equilibrium computation, strategy ranking
- **Red Teamer** — attacks YOUR strategy specifically, finds weaknesses
- **Synthesizer** — aggregates simulations + analysis into actionable recommendations

## Tech Stack

- Python, LangGraph (orchestration), oMLX (local inference via OpenAI-compatible HTTP)
- MLX on Mac Studio M3 Ultra, model: Qwen3.5-35B-A3B-Text (text-only, 35B MoE)
- Langfuse (observability), Neo4j CE + knowledge graph (`src/athena/knowledge/`, optional `--kg` flag)
- CLI entry point (`athena run`), YAML-driven case/simulation definitions
- JSON Schema structured output via oMLX `response_format`
- ThreadPoolExecutor for parallel Monte Carlo runs
- Pure-computation game theory module (`src/athena/game_theory/`)

## Knowledge Graph Layer (v0.7)

Optional Neo4j-backed knowledge graph (`--kg` flag / `ATHENA_KG_ENABLED=1`):
- **Ontology**: Pydantic entity/edge models mapping case schemas to graph nodes (`ontology.py`)
- **Ingestion**: case YAML → graph (`case_loader.py`), per-run results (`result_loader.py`), aggregated stats + game theory (`stats_loader.py`)
- **Context enrichment**: pre-simulation queries — seed arg ranking by judge, best precedent strategy, expected counters (`context_enrichment.py`)
- **Post-analysis**: argument trajectories cross-judge, determinative argument identification (`post_analysis.py`)
- **Semantic search**: embedding-based retrieval on arguments, legal texts, seed arguments (`semantic_search.py`)
- **CLI**: `athena run --kg`, `athena kg-status`
- **Graceful degradation**: KG off by default, all 309 tests pass without Neo4j, import failures → warning + continue
- **Dependencies**: `graphiti-core>=0.5`, `neo4j>=5.0`, `sentence-transformers>=3.0.0` as optional `[kg]` group

## Meta-Agents Layer (v0.8 + v0.9)

Post-processing agents that run AFTER aggregation + game theory, BEFORE memo:
- **Red Teamer** (`run_red_team`): adversarial analysis from opponent's perspective, structured output via `RED_TEAM_SCHEMA`
- **Game Theorist Agent** (`run_game_theorist`): interprets game theory computations for lawyers, structured output via `GAME_THEORIST_SCHEMA`
- **IRAC Extractor** (`run_irac_extraction`): decomposes seed arguments into Issue/Rule/Application/Conclusion, structured output via `IRAC_SCHEMA`
- All use `invoke_llm` (JSON repair + retry + Langfuse), temperature 0.6/0.3/0.3 respectively
- CLI: try/except wrapper (pipeline continues if meta-agent fails), outputs `red_team.json` + `game_theorist_agent.json` + `irac_analysis.json`
- Memo: SYNTHESIZER_SYSTEM_PROMPT includes sections 7 (analisi avversariale) + 8 (posizione negoziale) + 9 (analisi IRAC)

## Temporal Norm Versioning (v0.9)

- `LegalText` and `LegalTextNode` have `valid_from`, `valid_until`, `superseded_by` fields
- `SUPERSEDES` edges between norm versions in KG
- Backward-compatible: all fields optional with None defaults

## Current Phase

v0.9 on main — **fully validated** (unit tests + LLM smoke tests + KG smoke test).
- Monte Carlo run-v07-002: **60/60 OK**, 1833s (30.5 min), 210.7 eff tok/s
- oMLX optimized: continuous batching fixed, hot cache enabled, concurrency=8 (-33% wall clock vs run-001)
- Neo4j smoke test complete (2026-03-12): KG pipeline validated end-to-end
- v0.9 LLM smoke tests: **smoke-v09-1** (186s, 75.9 tok/s) + **smoke-v09-2** (175s, 74.0 tok/s) — all outputs valid, 0 JSON repairs
- 309 tests green (all mocked)

**Immediate next steps**:
1. v1.0 multi-jurisdiction (jurisdiction registry, per-jurisdiction prompts + phase builders)
2. v1.0 thin API layer (extract pipeline from cli.py → FastAPI endpoints)

**Roadmap**: v1.0 multi-jurisdiction + API → v1.1 sparring mode → v1.2 cross-case intelligence

## Key Risks & Open Questions

- Judge agent quality depends on jurisdiction-specific calibration data
- Multi-jurisdiction = different procedural rules per system, must be parameterizable
- Validation: need past cases with known outcomes as test set
- NOT legal advice — strategic analysis tool, decisions remain human
- Confidentiality: another reason for local-only inference
- Generic graph (`build_graph_from_phases`) is the production path — new agents are added as Phase entries
- Knowledge graph requires Neo4j CE — optional, graceful degradation when unavailable

## Related Work

- Harvey AI, CoCounsel, Luminance = single-agent legal assistants (not adversarial simulation)
- RAND wargaming with LLMs = closest pattern (multi-agent adversarial simulation, military domain)
- Academic game theory + litigation (Spier, Bebchuk) = analytical models, not software systems
- Gap: nobody combines multi-agent adversarial simulation + formal game theory + LLM for legal prep
