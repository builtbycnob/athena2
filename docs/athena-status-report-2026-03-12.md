# ATHENA — Status Report

**Data**: 12 marzo 2026
**Versione**: v0.9 (main) — fully validated
**Repository**: ~/athena (58 commit, 4 giorni di sviluppo)

---

## 1. Cos'è ATHENA

ATHENA (Adversarial Tactical Hearing & Equilibrium Navigation Agent) è un sistema multi-agente per la preparazione di casi legali che:

1. **Simula il contenzioso** — agenti LLM impersonano le parti avverse (ricorrente, resistente, giudice) con case file privati, obiettivi espliciti e belief model bayesiani
2. **Applica teoria dei giochi** — stima valori attesi, BATNA, equilibrio di Nash, ZOPA, sensitivity analysis
3. **Genera raccomandazioni strategiche** — memo strutturato con ranking argomenti, analisi avversariale, posizione negoziale, decomposizione IRAC

Tre modalità operative: **Wargame** (simulazione Monte Carlo), **Strategist** (analisi semi-autonoma), **Sparring** (interattivo, futuro).

### Differenziazione

- Harvey AI, CoCounsel, Luminance = assistenti legali single-agent (non simulazione avversariale)
- RAND wargaming con LLM = pattern più vicino (militare, non legale)
- Letteratura accademica (Spier, Bebchuk) = modelli analitici, non software
- **Gap colmato**: nessuno combina simulazione avversariale multi-agente + teoria dei giochi formale + LLM per preparazione legale

---

## 2. Architettura

### Stack tecnologico

| Componente | Tecnologia | Note |
|---|---|---|
| Linguaggio | Python 3.11+ | ~6500 LoC produzione, ~5650 LoC test |
| Orchestrazione | LangGraph | Grafi computazionali per simulazione multi-agente |
| Inferenza | oMLX (OpenAI-compatible HTTP) | Locale su Mac Studio M3 Ultra 256GB |
| Modello | Qwen3.5-35B-A3B-Text (MoE, 3B attivi) | text-only, structured output JSON |
| Osservabilità | Langfuse v3 | Tracing per-call, usage tracking, graceful no-op |
| Knowledge Graph | Neo4j CE 2026.01.4 (opzionale) | 14 tipi nodo, 20+ tipi edge, semantic search |
| Embedding | nomic-embed-text-v1.5 (768 dim) | sentence-transformers, lazy-load |
| Game Theory | Modulo puro Python | Zero dipendenze aggiuntive |
| Output | JSON + Markdown | Strutturato per integrazione downstream |
| Parallelismo | ThreadPoolExecutor | Monte Carlo parallelo, concurrency=8 |
| Robustezza | json_repair BNF library | 3 livelli: prompt prevention, regex, BNF repair |

### Struttura progetto

```
src/athena/                          # 44 moduli Python, ~6500 LoC
├── agents/                          # LLM interface layer
│   ├── llm.py                       # invoke_llm, JSON repair, retry, Langfuse
│   ├── json_repair.py               # 3-layer defense, BNF repair
│   ├── errors.py                    # JSONTruncatedError, JSONMalformedError
│   ├── meta_agents.py               # Red Team, Game Theorist, IRAC Extractor
│   ├── meta_prompts.py              # Prompt templates per meta-agent
│   ├── prompts.py                   # Prompt templates per party/judge
│   └── prompt_registry.py           # Registry per N-party prompts
├── schemas/                         # Pydantic models + JSON Schema
│   ├── case.py                      # CaseFile, Party, LegalText
│   ├── simulation.py                # SimulationConfig
│   ├── state.py                     # GraphState
│   ├── structured_output.py         # AGENT_SCHEMAS dict (per-agent JSON Schema)
│   └── meta_output.py               # RED_TEAM_SCHEMA, GAME_THEORIST_SCHEMA, IRAC_SCHEMA
├── simulation/                      # Core simulation engine
│   ├── graph.py                     # build_graph_from_phases, AgentConfig, Phase
│   ├── orchestrator.py              # Monte Carlo runner (ThreadPoolExecutor)
│   ├── aggregator.py                # Cross-run aggregation
│   ├── context.py                   # Context builders per agent type
│   └── validation.py                # Output validation per agent
├── game_theory/                     # Pure computation, no LLM
│   ├── valuation.py                 # Outcome → EUR mapping
│   ├── equilibrium.py               # BATNA, Nash bargaining, ZOPA, EV
│   ├── sensitivity.py               # Parameter sweeps, tornado ranking
│   └── schemas.py                   # GameTheoryAnalysis Pydantic model
├── knowledge/                       # Neo4j knowledge graph (optional)
│   ├── config.py                    # Neo4j connection, graceful degradation
│   ├── ontology.py                  # 14 node types, 20+ edge types (Pydantic)
│   ├── embedder.py                  # Sentence-transformers wrapper
│   ├── ingestion/                   # case → graph, results → graph, stats → graph
│   │   ├── case_loader.py
│   │   ├── result_loader.py
│   │   └── stats_loader.py
│   └── queries/                     # Pre-sim enrichment, post-analysis, semantic search
│       ├── context_enrichment.py
│       ├── post_analysis.py
│       └── semantic_search.py
├── output/                          # Report generation
│   ├── memo.py                      # Strategic memo (LLM-generated, 10 sezioni)
│   ├── table.py                     # Probability table
│   ├── decision_tree.py             # Decision tree ASCII
│   └── game_theory_summary.md       # GT summary markdown
└── cli.py                           # Entry point, pipeline orchestration

tests/                               # 27 test modules, ~5650 LoC
├── 309 test passati, 23 skipped, 0 failures
└── Tutti mocked (nessuna dipendenza da LLM/Neo4j per CI)
```

### Tipi di agente

| Tipo | Ruolo | Implementazione |
|---|---|---|
| **Party Agent** (N) | Rappresenta una parte (ricorrente, resistente, terzi...) | Case file privato, obiettivo esplicito, vincoli legali, belief model |
| **Structural Agent** | Giudice/Arbitro, Perito/CTU | Valuta argomenti vs standard giuridici, calibrato per giurisdizione |
| **Red Teamer** | Attacca la strategia dal punto di vista avversario | Post-aggregation, structured output, 4 vulnerability types |
| **Game Theorist** | Interpreta output GT per avvocati | BATNA/ZOPA human-readable, settlement recommendation |
| **IRAC Extractor** | Decompone argomenti in Issue/Rule/Application/Conclusion | Per seed argument, deduplicazione cross-run |
| **Synthesizer** | Genera memo strategico finale | 10 sezioni, integra tutti i dati |

### Pipeline di esecuzione

```
Case YAML + Simulation YAML
    │
    ▼
Monte Carlo Simulation (N run paralleli, concurrency=8)
    │  Per ogni run: Party₁ → Party₂ → ... → Judge
    │  (build_graph_from_phases, LangGraph)
    ▼
Aggregation (cross-run statistics)
    │
    ▼
Game Theory Analysis (pure computation)
    │  Outcome valuation, BATNA, Nash, sensitivity
    ▼
Meta-Agents (3 LLM call paralleli)
    │  Red Team → Game Theorist → IRAC Extractor
    ▼
Report Generation
    │  strategic_memo.md (LLM, 10 sezioni)
    │  game_theory.json, game_theory_summary.md
    │  red_team.json, game_theorist_agent.json
    │  irac_analysis.json, probability_table.md
    │  decision_tree.txt, raw_results.json
    ▼
[Optional] Knowledge Graph Ingestion (--kg flag)
    │  Case → graph, results → graph, GT → graph
    │  Context enrichment, post-analysis, semantic search
    ▼
Output directory con tutti gli artefatti
```

---

## 3. Cronologia versioni

| Versione | Data | Descrizione | Commit |
|---|---|---|---|
| v0.1 | 2026-03-09 | Scaffolding + robustness layer (JSON repair, retry, error classification) | 369ae49–61e521e |
| v0.1.1 | 2026-03-09 | Langfuse v3 observability | f92bfda |
| v0.2 | 2026-03-10 | oMLX HTTP backend (OpenAI-compatible, prefix caching) | 643b5bf |
| v0.3 | 2026-03-10 | Orchestrator parallelization (ThreadPoolExecutor, concurrency flag) | a768bd4 |
| v0.4 | 2026-03-10 | Structured output + inference tuning (JSON Schema per agent, sampling params) | 8fca8cc |
| v0.5 | 2026-03-11 | N-party architecture (generic schemas, prompt registry, graph builder) | 6950905 |
| v0.5.1 | 2026-03-12 | Wire generic graph into production, remove legacy code | 049ccf2 |
| v0.6 | 2026-03-12 | Game theory (BATNA, Nash, ZOPA, sensitivity, tornado ranking) | 9253776 |
| v0.7 | 2026-03-12 | Knowledge graph (Neo4j ontology, ingestion, queries, semantic search) | 94180e2–55fb637 |
| v0.8 | 2026-03-12 | Meta-agents (Red Teamer + Game Theorist Agent) | 9fe4743 |
| v0.9 | 2026-03-12 | Temporal norm versioning + IRAC extraction + embedder + semantic search | (uncommitted) |

**58 commit in 4 giorni** (9–12 marzo 2026). Sviluppo solo, nessun collaboratore.

---

## 4. Validazione

### Unit test

- **309 test passati**, 23 skipped, 0 failures
- 27 moduli di test, ~5650 LoC di test
- Tutti mocked: nessuna dipendenza runtime da LLM o Neo4j
- Copertura: schemas, CLI, LLM pipeline, JSON repair, game theory, KG ontology/ingestion/queries, meta-agents, aggregation, validation, orchestrator, output generation, backward compat, migration

### Monte Carlo (60 run, LLM reale)

| Run | Data | Risultato | Durata | Throughput | Note |
|---|---|---|---|---|---|
| run-001 | 2026-03-11 | 58/60 OK | 2745s (45 min) | 34 avg tok/s | Pre-ottimizzazione, 2 failure (embedded quotes) |
| run-v07-002 | 2026-03-12 | **60/60 OK** | 1833s (30 min) | 210.7 eff tok/s | Post-ottimizzazione, 13 JSON repair, 0 failure |

Miglioramento: **-33% wall clock**, **6.2x throughput** grazie a ottimizzazione oMLX (continuous batching, hot cache, concurrency 4→8).

### Smoke test v0.9 (meta-agents + IRAC, LLM reale)

| Test | LLM Calls | Token | Durata | Throughput |
|---|---|---|---|---|
| smoke-v09-1 | 7 (3 sim + 3 meta + 1 memo) | 14162 | 186s | 75.9 tok/s |
| smoke-v09-2 | 7 (3 sim + 3 meta + 1 memo) | 12909 | 175s | 74.0 tok/s |

Risultati smoke-v09-2:
- `red_team.json`: 4 vulnerabilità identificate, rischio complessivo = **high**
- `game_theorist_agent.json`: should_settle = **false**, raccomanda contenzioso
- `irac_analysis.json`: 2 analisi IRAC (SEED_ARG3, SEED_RARG1)
- `strategic_memo.md`: tutte le 10 sezioni presenti
- **0 JSON repair, 0 crash, 0 meta-agent failure**

### Neo4j / Knowledge Graph (smoke test)

- Neo4j CE 2026.01.4, bolt://localhost:7687
- Case ingestion: 13 nodi, 20 edge
- Result ingestion: 6-10 nodi, 28-40 edge per run
- Game theory ingestion: 8 nodi, 8 edge
- Pipeline end-to-end validato

---

## 5. Dettaglio funzionale per layer

### 5.1 Robustness Layer (v0.1)

Difesa a 3 livelli contro output LLM malformato:
1. **Prompt prevention** — istruzioni esplicite per JSON puro
2. **Targeted regex** — estrazione JSON da output con testo extra
3. **json_repair BNF library** — riparazione strutturale (quote, virgole, truncation)

Retry automatico con 2x token budget al secondo tentativo. 16/16 artefatti reali di failure gestiti. Preservazione apostrofi italiani (`l'articolo`).

### 5.2 Structured Output (v0.4)

JSON Schema per ogni tipo di agente con vincoli `maxLength`, `enum`, `minItems`. Threaded attraverso `invoke_llm` → oMLX `response_format`. Sampling params ottimizzati: `repetition_penalty=1.3`, `top_p=0.8`, `top_k=20`. Max token per agent: appellant 4096, respondent 4096, judge 6144.

### 5.3 N-Party Architecture (v0.5)

Generalizzazione da bilaterale a N parti:
- `Party.visibility` per case file privati
- `build_graph_from_phases(phases)` — grafi computazionali parametrici
- Prompt registry per generazione dinamica prompt
- Combination generator per Monte Carlo multi-profilo
- Backward-compatible: caso bilaterale è caso speciale di N=2

### 5.4 Game Theory Layer (v0.6)

Modulo puro Python (zero dipendenze LLM):
- **Outcome valuation**: verdict → EUR per prospettiva di parte
- **BATNA**: expected value of litigation con intervalli di confidenza
- **Nash bargaining**: ZOPA bilaterale + soluzione di Nash
- **EV by strategy**: ranking per expected value
- **Sensitivity analysis**: parameter sweep + tornado ranking
- **Dominated strategy identification**: esclusione statistica da Monte Carlo

### 5.5 Knowledge Graph (v0.7)

Ontologia Pydantic → Neo4j:
- 14 tipi nodo: Case, Party, LegalText, SeedArgument, Argument, Verdict, JudgeProfile, SimRun, Strategy, GameTheoryResult, IracNode, ...
- 20+ tipi edge: PARTY_IN, CITES_NORM, PRODUCES_ARGUMENT, RESULTS_IN, HAS_IRAC, SUPERSEDES, ...
- Ingestion: case YAML → grafo, risultati per-run → grafo, GT aggregato → grafo
- Query: context enrichment (pre-simulazione), post-analysis (per memo)
- Semantic search: vector index su ArgumentNode, SeedArgumentNode, LegalTextNode
- Opzionale: `--kg` flag, graceful degradation quando Neo4j non disponibile

### 5.6 Meta-Agents (v0.8 + v0.9)

Post-processing con LLM strutturato:
- **Red Teamer** (temp=0.6): analisi avversariale, 4 categorie vulnerabilità, rischio complessivo
- **Game Theorist Agent** (temp=0.3): interpreta output GT per avvocati, raccomandazione settlement
- **IRAC Extractor** (temp=0.3): decompone seed argument in Issue/Rule/Application/Conclusion

Tutti usano `invoke_llm` con JSON repair + retry + Langfuse. Pipeline continua anche se un meta-agent fallisce (try/except).

### 5.7 Temporal Norm Versioning (v0.9)

`LegalText` e `LegalTextNode` con campi `valid_from`, `valid_until`, `superseded_by`. Edge `SUPERSEDES` tra versioni di norme nel KG. Backward-compatible (tutti i campi opzionali).

### 5.8 Strategic Memo (10 sezioni)

1. Sintesi esecutiva
2. Analisi per scenario
3. Argomenti: ranking di efficacia
4. Analisi del precedente
5. Raccomandazione strategica
6. Pattern dal Knowledge Graph
7. **Analisi avversariale** (Red Team)
8. **Posizione negoziale** (Game Theory Agent)
9. **Analisi IRAC** (decomposizione argomentativa)
10. Rischi e caveat

---

## 6. Performance & infrastruttura

### Hardware

Mac Studio M3 Ultra, 256GB RAM unificata, inferenza locale.

### Modello

`nightmedia/Qwen3.5-35B-A3B-Text-qx64-hi-mlx` — MoE 35B totali, 3B attivi per token. Text-only (no vision overhead). `enable_thinking=False` (risparmio token, incompatibile con structured output).

### oMLX (server di inferenza)

| Parametro | Valore | Impatto |
|---|---|---|
| completion_batch_size | 16 | Continuous batching corretto |
| hot_cache_max_size | 16GB | Prefix caching in RAM |
| max_num_seqs | 32 | Headroom per concurrency=8 |
| initial_cache_blocks | 4096 | Evita grow-on-demand overhead |

### Throughput

| Metrica | Valore |
|---|---|
| Per-call throughput | ~27 tok/s (singolo) → ~67-82 tok/s (parallelo) |
| Aggregated throughput | ~211 tok/s (8 call paralleli) |
| Parallelism efficiency | 7.8x su 8 thread |
| Bottleneck | Judge agent (43% LLM time, avg 2422 output tokens) |

---

## 7. Dipendenze

### Core (sempre richieste)
```
langgraph, langchain-core, langchain-community
mlx-lm
langfuse
pydantic>=2.0
pyyaml
httpx>=0.27
json-repair>=0.30
```

### Dev (test)
```
pytest, pytest-asyncio, jsonschema
```

### Knowledge Graph (opzionale, `pip install -e ".[kg]"`)
```
graphiti-core>=0.5
neo4j>=5.0
sentence-transformers>=3.0.0
```

---

## 8. Output di un run tipico

```
output/smoke-v09-2/
├── raw_results.json            # 21KB — risultati grezzi per-run
├── strategic_memo.md           # 13KB — memo strategico (10 sezioni)
├── game_theory.json            #  6KB — analisi GT strutturata
├── game_theory_summary.md      #  1KB — riassunto GT human-readable
├── red_team.json               #  6KB — analisi avversariale
├── game_theorist_agent.json    #  4KB — interpretazione GT per avvocati
├── irac_analysis.json          #  4KB — decomposizione IRAC
├── probability_table.md        #  <1KB — tabella probabilità
└── decision_tree.txt           #  <1KB — albero decisionale ASCII
```

---

## 9. Roadmap

### Completato (v0.1–v0.9)

- Robustness layer + JSON repair (BNF)
- Langfuse observability
- oMLX HTTP backend con prefix caching
- Orchestrator parallelization (ThreadPoolExecutor, concurrency=8)
- Structured output (JSON Schema per agent)
- N-party architecture (N agenti configurabili)
- Game theory (BATNA, Nash, ZOPA, sensitivity, tornado)
- Knowledge graph (Neo4j, ontology, ingestion, queries)
- Meta-agents (Red Team, Game Theorist, IRAC)
- Temporal norm versioning
- Semantic search (embedding-based)
- **Validazione completa**: 309 unit test, 60/60 Monte Carlo, 2x smoke LLM, KG smoke

### v1.0 — Multi-Jurisdiction + Thin API Layer (prossimo)

**Multi-Jurisdiction**:
- Jurisdiction registry: `jurisdiction_id` → prompt templates + phase builder + legal standards
- Almeno una giurisdizione common law oltre all'italiana
- Case YAML con campo `jurisdiction`
- Phase structure variabile per giurisdizione (es. common law: discovery, depositions, motions)

**Thin API Layer**:
- Estrazione `run_pipeline()` da `cli.py` → `src/athena/api/pipeline.py`
- 4 endpoint FastAPI: `POST /runs`, `GET /runs/{id}`, `GET /runs`, `GET /health`
- `PipelineResult` dataclass
- Background task per Monte Carlo (polling-based)

### v1.1 — Sparring Mode

Interattivo: umano gioca la propria parte, agenti LLM rispondono in real-time. WebSocket, session management (save/resume), Graphiti temporal episodes.

### v1.2 — Cross-Case Intelligence

Similarità tra casi via KG + embeddings. Transfer di effectiveness tra casi. Data flywheel: ogni caso processato migliora il successivo.

### v1.3 — Dashboard UI

Frontend completo con visualizzazione game theory, esplorazione KG, mappe argomentative.

### Future Game Theory Enhancements

- Extensive form game trees (mosse sequenziali: filing → response → discovery → motions → trial)
- Bayesian updating (aggiornamento credenze con nuove evidenze)
- Mixed strategies, signaling games, multi-stage bargaining

---

## 10. Rischi e limitazioni note

| Rischio | Severità | Mitigazione |
|---|---|---|
| Qualità agente giudice dipende da dati di calibrazione per giurisdizione | Alta | Test con casi a esito noto (da acquisire) |
| Multi-jurisdiction = regole procedurali diverse | Media | Jurisdiction registry parametrizzabile (v1.0) |
| Modello locale 35B vs. frontier (GPT-4, Claude) | Media | MoE efficiente, structured output compensa; upgrade path a modelli più grandi |
| NON è consulenza legale | Critica | Strumento di analisi strategica, decisioni restano umane |
| Confidenzialità dati caso | Critica | Inferenza locale = feature, non vincolo |
| Knowledge Graph richiede Neo4j | Bassa | Opzionale, graceful degradation |
| JSON output LLM non sempre valido | Bassa | 3-layer repair, 60/60 Monte Carlo superato |

---

## 11. Metriche quantitative riassuntive

| Metrica | Valore |
|---|---|
| Tempo di sviluppo | 4 giorni (9–12 marzo 2026) |
| Commit | 58 |
| LoC produzione | ~6500 |
| LoC test | ~5650 |
| Moduli Python | 44 (produzione) + 27 (test) |
| Test passati | 309 (+ 23 skipped) |
| Monte Carlo success rate | 60/60 (100%) |
| LLM smoke test | 2/2 passati (0 crash, 0 repair) |
| Throughput | 211 tok/s aggregato (8 thread) |
| Wall clock (60 run) | 30 min |
| Versioni rilasciate | 10 (v0.1 → v0.9) |
| Tipi agente | 6 (party, structural, red team, game theorist, IRAC, synthesizer) |
| Output per run | 9 file (JSON + Markdown) |
| Dipendenze opzionali | KG (Neo4j + sentence-transformers) |
