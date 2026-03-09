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

Implemented (high value):
- Extensive form game trees (sequential moves: filing → response → discovery → motions → trial)
- Settlement range estimation (BATNA analysis, Nash bargaining)
- Bayesian updating as new information emerges
- Sensitivity analysis ("if judge weighs evidence X as weak, case flips")
- Dominated strategy identification

Future:
- Mixed strategies, signaling games, multi-stage bargaining

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

## Tech Stack (Target)

- Python, LangGraph (orchestration), MLX (local inference on Mac Studio M3 Ultra)
- Langfuse (observability), Graphiti (knowledge graph for precedents/case law — future)
- Telegram or CLI for interaction
- YAML-driven case definitions

## Current Phase

Brainstorming / early design. First milestone: model a real case as proof-of-concept.

## Key Risks & Open Questions

- Judge agent quality depends on jurisdiction-specific calibration data
- Multi-jurisdiction = different procedural rules per system, must be parameterizable
- Validation: need past cases with known outcomes as test set
- NOT legal advice — strategic analysis tool, decisions remain human
- Confidentiality: another reason for local-only inference

## Related Work

- Harvey AI, CoCounsel, Luminance = single-agent legal assistants (not adversarial simulation)
- RAND wargaming with LLMs = closest pattern (multi-agent adversarial simulation, military domain)
- Academic game theory + litigation (Spier, Bebchuk) = analytical models, not software systems
- Gap: nobody combines multi-agent adversarial simulation + formal game theory + LLM for legal prep
