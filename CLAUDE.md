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
- CLI entry point (`athena run`, `athena serve`, `athena ingest-corpus`), YAML-driven case/simulation definitions
- JSON Schema structured output via oMLX `response_format` + XGrammar token-level enforcement
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

## Multi-Jurisdiction Support (v1.0)

- **Jurisdiction registry**: `src/athena/jurisdiction/` — `JurisdictionConfig` maps country → prompts, schemas, outcome extractors
- **Auto-detection**: `get_jurisdiction_for_case(case_data)` reads `jurisdiction.country`, defaults to IT
- **Italian (IT)**: wraps existing behavior (zero regression), `JUDGE_SCHEMA` with `qualification_correct`
- **Swiss (CH)**: Bundesgericht prompts, two-step judge architecture with `JUDGE_CH_STEP1_SCHEMA` + `JUDGE_CH_STEP2_SCHEMA`
- **Outcome extraction**: aggregator, scorer, valuation all auto-detect jurisdiction from verdict shape
- **Adding a new jurisdiction**: create `src/athena/jurisdiction/{code}.py` with `JurisdictionConfig`, add prompts to `prompts.py`, add judge schema if different from existing

## Two-Step CH Judge (v1.1)

- **Step 1** (temp=0.7): error identification — finds errors in lower court decision, classifies severity (decisive/significant/minor/none)
- **Step 2** (temp=0.4): outcome decision — re-evaluates Step 1 errors, decides `lower_court_correct`
- **Consistency enforcement** (`_ch_enforce_consistency`): deterministic override — decisive errors → force `lower_court_correct=False`, no decisive → force True. Applied at graph merge + outcome extraction.
- **Prompt key**: `judge_ch_step1` / `judge_ch_step2`, schema key: `judge_ch_step1` / `judge_ch_step2`
- Broke monodirezionale dismissed bias (93% → balanced), accuracy 50% → 60%

## Thin API Layer (v1.2)

- **Pipeline extraction**: `src/athena/api/pipeline.py` — `run_pipeline()` pure logic (no file I/O), `prepare_case_data()`, `prepare_sim_config()`, `write_pipeline_outputs()`
- **Pydantic models**: `src/athena/api/models.py` — `PipelineOptions`, `ProgressEvent`, `PipelineResult`, `RunState`, `RunRequest`
- **FastAPI app**: `src/athena/api/app.py` — `GET /health`, `POST /runs` (202 + background), `GET /runs/{id}`, `GET /runs`, `GET /runs/{id}/stream` (SSE)
- **Run registry**: `src/athena/api/registry.py` — in-memory thread-safe state + asyncio.Queue for SSE bridge
- **CLI refactor**: `athena run` now ~30 lines (load YAML → `run_pipeline` → `write_pipeline_outputs`), `athena serve --host --port` for FastAPI
- **Dependencies**: `pip install athena[api]` → fastapi, uvicorn, sse-starlette

## RAG Legal Corpus (v1.3)

- **Embedder**: `src/athena/rag/embedder.py` — BGE-M3 (1024D, multilingual), lazy-load with double-checked locking
- **Vector store**: `src/athena/rag/store.py` — LanceDB (embedded, zero-server), `NormChunk` model, per-jurisdiction tables, hybrid search with RRF
- **Swiss corpus ingestion**: `src/athena/rag/ingestion/swiss.py` — `rcds/swiss_legislation` from HuggingFace, article-level chunking, batch embedding
- **Retriever**: `src/athena/rag/retriever.py` — queries from seed arguments + facts, dedup, filter existing norms, token budget truncation
- **Integration**: judge prompts include `## Testi normativi aggiuntivi (RAG)` section when enabled
- **CLI**: `athena run --rag`, `athena ingest-corpus --jurisdiction CH`, `ATHENA_RAG_ENABLED=1`
- **Graceful degradation**: RAG off by default, all tests pass without lancedb/sentence-transformers
- **Dependencies**: `pip install athena[rag]` → lancedb, sentence-transformers

## XGrammar Constrained Decoding (v1.4)

Token-level JSON schema enforcement via XGrammar pushdown automaton in oMLX:
- **`omlx/xgrammar_processor.py`** (site-packages): GrammarCompilerCache + XGrammarLogitsProcessor
- **`omlx/request.py`**: `json_schema` field on SamplingParams
- **`omlx/server.py`**: extracts JSON schema from response_format
- **`omlx/scheduler.py`**: GrammarCompilerCache init + processor creation in `_build_sampler_and_processors()`
- **Graceful degradation**: xgrammar not installed → prompt-only schema hint (no crash)
- **Performance**: 0.28ms/token overhead (~1.5% of generation time)
- **MLX compat**: bitmask unpacked via numpy → `mx.where(mask, logits, -inf)` (torch bitmask → mx conversion)

### Dynamic Enum Schemas
- **`src/athena/schemas/schema_builder.py`**: deep-copies static schemas, injects enum constraints from case data
- `build_schema_for_agent(schema_key, case_data, prior_briefs, step1_error_count)` → schema with enums
- Fields: `facts_referenced`, `evidence_cited`, `norm_text_cited`, `legal_basis`, `to_argument`, `argument_id`, `error_id`, `precedents_addressed[].id`
- Dynamic `minItems` on `argument_evaluation` = total argument count
- **graph.py**: 4 call sites use `build_schema_for_agent()` instead of `AGENT_SCHEMAS.get()`

### Error-ID Merge Fix
- `graph.py` + `ch.py` severity floor: matches by `error_id` field (not array position)
- Prevents mismatch when Step 2 skips or reorders errors

### Robustness Fixes (v1.4)
- **Embedder pre-load**: `orchestrator.py` loads RAG embedder on main thread before ThreadPoolExecutor; `embedder.py` sets `TOKENIZERS_PARALLELISM=false` to prevent sentence-transformers/loky POSIX semaphore deadlocks
- **Step 1 max_tokens**: 6144 → 8192 (Step 2 stays 6144 via `_step2_max_tokens` template var in graph.py)
- **Severity ceiling**: Step 2 can upgrade severity at most +1 level (graph.py merge + ch.py consistency enforcement). Prevents none→decisive false upgrades while allowing legitimate escalation (e.g. significant→decisive)

## Multi-Model Support (v1.4b)

- **Static role-based routing**: each agent role maps to a model name, threaded through entire call chain
- **Resolution order**: simulation YAML `models.{role}` > `JurisdictionConfig.default_models` > `OMLX_MODEL` env var
- **CH default**: `qwen3.5-122b-a10b-4bit` (122B MoE) for judge, parties use default 35B
- **Implementation**: `model` param on `invoke_llm` → `_call_model` → `_call_model_omlx`; `AgentConfig.model` field; `_model_for_role()` in `graph.py`
- **oMLX**: per-request model selection via `model` field in OpenAI-compatible API; EnginePool handles concurrent loading + LRU eviction
- **Memory**: 122B (~73GB) + 35B (~22GB) = ~95GB — within 223GB budget on Mac Studio
- **Langfuse**: reports `effective_model` (actual model used, not default) in both success and error paths
- **Research basis**: FrugalGPT, RouteLLM, AutoMix, R2-Reasoner, Hybrid LLM — static routing valid for domain-knowledge-driven task decomposition

## Current Phase

v1.4b on main — XGrammar + robustness + multi-model support, **532 tests green**.
- **Multi-model routing**: 122B MoE for judge, 35B for parties (static role-based)
  - Resolution: simulation YAML `models.{role}` > `JurisdictionConfig.default_models` > `OMLX_MODEL`
  - CH default: `qwen3.5-122b-a10b-4bit` for judge, parties use `OMLX_MODEL` (35B)
  - Research: FrugalGPT, RouteLLM, AutoMix, R2-Reasoner validate static routing for known task decomposition
- **Swiss validation 35B: 9/10 accuracy** (ch-3408 = 35B limit on conditional-waiver reasoning)
- **Swiss validation 122B: 10/10 accuracy** (100%)
- ch-741: was 0/6 crash (embedder deadlock) → 5/6 rejection (correct)
- ch-3408: 3/3 tie with 35B (Step 1 false-decisive); expected to resolve with 122B judge
- 3 robustness fixes: embedder pre-load + TOKENIZERS_PARALLELISM, Step 1 max_tokens 8192, severity ceiling
- RAG corpus: 747,946 chunks from 35,698 Swiss laws (BGE-M3 + LanceDB)
- Dual-backend embedder: BGE-M3 default (96 text/s), Qwen3 MLX optional via `ATHENA_RAG_BACKEND=mlx`
- oMLX optimized: continuous batching fixed, hot cache enabled, concurrency=8
- **OMLX_MODEL must be `qwen3.5-35b-a3b-text-hi`** (short name — full HF name gives 404)
- **XGrammar**: `xgrammar==0.1.32`, oMLX patches in site-packages (4 files)
- 10 Swiss Bundesgericht cases in `cases/validation/`, ground truth in `ground_truth/`

**Immediate next steps**:
1. Full validation with 122B judge (expect 10/10)
2. v1.5 sparring mode (interactive adversarial simulation)

**Roadmap**: 122B validation → v1.5 sparring → v1.6 cross-case intelligence

## Key Risks & Open Questions

- Judge agent quality depends on jurisdiction-specific calibration data
- Swiss validation: 100% with 122B, 90% with 35B — model size matters for complex legal reasoning (ch-2434, ch-3408 = conditional-waiver reasoning beyond 35B capacity)
- oMLX XGrammar patches are in site-packages — oMLX update will overwrite them (pin version or upstream PR)
- NOT legal advice — strategic analysis tool, decisions remain human
- Confidentiality: another reason for local-only inference
- Generic graph (`build_graph_from_phases`) is the production path — new agents are added as Phase entries
- Knowledge graph requires Neo4j CE — optional, graceful degradation when unavailable

## Related Work

- Harvey AI, CoCounsel, Luminance = single-agent legal assistants (not adversarial simulation)
- RAND wargaming with LLMs = closest pattern (multi-agent adversarial simulation, military domain)
- Academic game theory + litigation (Spier, Bebchuk) = analytical models, not software systems
- Gap: nobody combines multi-agent adversarial simulation + formal game theory + LLM for legal prep
