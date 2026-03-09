# ATHENA PoC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working Monte Carlo adversarial simulation for an Italian traffic violation case, producing strategic recommendations.

**Architecture:** YAML-driven case definition → LangGraph sequential graph (appellant → respondent → judge) → Monte Carlo orchestrator → statistical aggregator → 3 output formats (table, decision tree, strategic memo). Information asymmetry enforced by context builders.

**Tech Stack:** Python 3.11, LangGraph (StateGraph), MLX (Qwen3.5-122B-A10B-4bit as primary candidate), Langfuse (observability), Pydantic (schema validation), PyYAML, pytest.

**Design doc:** `docs/plans/2026-03-09-athena-poc-design.md` — reference for all schemas, prompts, and architectural decisions.

**Environment:** Mac Studio M3 Ultra 256GB. All dependencies (mlx, langgraph, langfuse, pydantic, pyyaml) already installed.

---

## Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/athena/__init__.py`
- Create: `src/athena/schemas/__init__.py`
- Create: `src/athena/agents/__init__.py`
- Create: `src/athena/simulation/__init__.py`
- Create: `src/athena/output/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `cases/gdp-milano-17928-2025.yaml` (stub)
- Create: `simulations/run-001.yaml` (stub)

**Step 1: Create pyproject.toml**

```toml
[project]
name = "athena"
version = "0.1.0"
description = "Adversarial Tactical Hearing & Equilibrium Navigation Agent"
requires-python = ">=3.11"
dependencies = [
    "langgraph",
    "langchain-core",
    "langchain-community",
    "mlx-lm",
    "langfuse",
    "pydantic>=2.0",
    "pyyaml",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Create directory structure**

Run: `mkdir -p src/athena/{schemas,agents,simulation,output} tests && touch src/athena/__init__.py src/athena/schemas/__init__.py src/athena/agents/__init__.py src/athena/simulation/__init__.py src/athena/output/__init__.py tests/__init__.py`

**Step 3: Create tests/conftest.py with shared fixtures**

```python
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def sample_case_data():
    """Minimal case data for testing."""
    return {
        "case_id": "gdp-milano-17928-2025",
        "jurisdiction": {
            "country": "IT",
            "court": "giudice_di_pace",
            "venue": "Milano",
            "applicable_law": ["D.Lgs. 285/1992", "L. 689/1981"],
            "key_precedents": [
                {
                    "id": "cass_16515_2005",
                    "citation": "Cass. civ. n. 16515/2005",
                    "holding": "Equiparazione contromano/controsenso",
                    "weight": "contested",
                }
            ],
            "procedural_rules": {
                "rite": "opposizione_sanzione_amministrativa",
                "phases": ["ricorso", "costituzione_resistente", "udienza", "decisione"],
                "allowed_moves": {
                    "appellant": ["memoria", "produzione_documenti"],
                    "respondent": ["memoria_costituzione", "produzione_documenti"],
                },
            },
        },
        "parties": [
            {
                "id": "opponente",
                "role": "appellant",
                "type": "persona_fisica",
                "objectives": {
                    "primary": "annullamento_verbale",
                    "subordinate": "riqualificazione_artt_6_7",
                },
            },
            {
                "id": "comune_milano",
                "role": "respondent",
                "type": "pubblica_amministrazione",
                "entity": "Comune di Milano — Polizia Locale",
                "objectives": {
                    "primary": "conferma_verbale",
                    "subordinate": "conferma_anche_con_riduzione",
                },
            },
        ],
        "stakes": {
            "current_sanction": {
                "norm": "art. 143 CdS",
                "fine_range": [170, 680],
                "points_deducted": 4,
            },
            "alternative_sanction": {
                "norm": "artt. 6-7 CdS",
                "fine_range": [42, 173],
                "points_deducted": 0,
            },
            "litigation_cost_estimate": 1500,
        },
        "evidence": [
            {
                "id": "DOC1",
                "type": "atto_pubblico",
                "description": "Verbale Polizia Locale",
                "produced_by": "comune_milano",
                "admissibility": "uncontested",
                "supports_facts": ["F1", "F2", "F3"],
            },
            {
                "id": "DOC2",
                "type": "prova_documentale",
                "description": "Documentazione segnaletica",
                "produced_by": "opponente",
                "admissibility": "uncontested",
                "supports_facts": ["F3"],
            },
        ],
        "facts": {
            "undisputed": [
                {"id": "F1", "description": "Transito in senso vietato", "evidence": ["DOC1"]},
                {"id": "F2", "description": "Verbale ex art. 143 CdS", "evidence": ["DOC1"]},
                {"id": "F3", "description": "Strada a senso unico", "evidence": ["DOC1", "DOC2"]},
            ],
            "disputed": [
                {
                    "id": "D1",
                    "description": "Correttezza qualificazione giuridica",
                    "appellant_position": "Art. 143 inapplicabile",
                    "respondent_position": "Art. 143 applicabile per Cass. 16515/2005",
                    "depends_on_facts": ["F1", "F3"],
                }
            ],
        },
        "legal_texts": [
            {
                "id": "art_143_cds",
                "norm": "Art. 143 D.Lgs. 285/1992",
                "text": "I veicoli devono circolare sulla parte destra della carreggiata e in prossimità del margine destro della medesima, anche quando la strada è libera. [testo di esempio per test]",
            },
            {
                "id": "art_6_cds",
                "norm": "Art. 6 D.Lgs. 285/1992",
                "text": "Il prefetto può, per motivi di sicurezza pubblica o inerenti alla sicurezza della circolazione... [testo di esempio per test]",
            },
            {
                "id": "art_1_l689",
                "norm": "Art. 1 L. 689/1981",
                "text": "Nessuno può essere assoggettato a sanzioni amministrative se non in forza di una legge che sia entrata in vigore prima della commissione della violazione. Le leggi che prevedono sanzioni amministrative si applicano soltanto nei casi e per i tempi in esse considerati.",
            },
        ],
        "seed_arguments": {
            "appellant": [
                {
                    "id": "SEED_ARG1",
                    "claim": "Errata qualificazione giuridica",
                    "direction": "Art. 143 non copre la fattispecie",
                    "references_facts": ["F1", "F3", "D1"],
                },
                {
                    "id": "SEED_ARG2",
                    "claim": "Contraddizione interna del verbale",
                    "direction": "Verbale descrive senso unico, applica norma da doppio senso",
                    "references_facts": ["F3"],
                },
            ],
            "respondent": [
                {
                    "id": "SEED_RARG1",
                    "claim": "Legittimità ex Cass. 16515/2005",
                    "direction": "Cassazione equipara le due condotte",
                    "references_facts": ["F1", "D1"],
                },
            ],
        },
        "key_precedents": [
            {
                "id": "cass_16515_2005",
                "citation": "Cass. civ. n. 16515/2005",
                "holding": "Equiparazione contromano/controsenso",
                "weight": "contested",
            }
        ],
        "timeline": [],
    }


@pytest.fixture
def sample_run_params():
    """Minimal run params for testing."""
    return {
        "run_id": "test__aggressivo__000",
        "judge_profile": {
            "id": "formalista_pro_cass",
            "jurisprudential_orientation": "follows_cassazione",
            "formalism": "high",
        },
        "appellant_profile": {
            "id": "aggressivo",
            "style": "Attacca frontalmente la giurisprudenza sfavorevole.",
        },
        "temperature": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
        "language": "it",
    }


@pytest.fixture
def sample_appellant_brief():
    """Valid appellant brief output for testing downstream agents."""
    return {
        "filed_brief": {
            "arguments": [
                {
                    "id": "ARG1",
                    "type": "derived",
                    "derived_from": "SEED_ARG1",
                    "claim": "Errata qualificazione giuridica del fatto",
                    "legal_reasoning": "L'art. 143 disciplina la marcia contromano su strada a doppio senso. Il fatto è avvenuto su senso unico.",
                    "norm_text_cited": ["art_143_cds"],
                    "facts_referenced": ["F1", "F3"],
                    "evidence_cited": ["DOC1"],
                    "precedents_addressed": [
                        {
                            "id": "cass_16515_2005",
                            "strategy": "distinguish",
                            "reasoning": "Il precedente non è in punto.",
                        }
                    ],
                    "supports": None,
                },
            ],
            "requests": {
                "primary": "Annullamento del verbale",
                "subordinate": "Riqualificazione sotto artt. 6-7 CdS",
            },
        },
        "internal_analysis": {
            "strength_self_assessments": {"ARG1": 0.7},
            "key_vulnerabilities": ["Cassazione 16515/2005 contraria"],
            "strongest_point": "Testo letterale art. 143 non copre senso unico",
            "gaps": [],
        },
    }


@pytest.fixture
def sample_respondent_brief():
    """Valid respondent brief output for testing judge."""
    return {
        "filed_brief": {
            "preliminary_objections": [],
            "responses_to_opponent": [
                {
                    "to_argument": "ARG1",
                    "counter_strategy": "rebut",
                    "counter_reasoning": "La Cassazione ha equiparato le due fattispecie.",
                    "norm_text_cited": ["art_143_cds"],
                    "precedents_cited": [
                        {"id": "cass_16515_2005", "relevance": "Direttamente in punto."}
                    ],
                }
            ],
            "affirmative_defenses": [
                {
                    "id": "RARG1",
                    "type": "derived",
                    "derived_from": "SEED_RARG1",
                    "claim": "Legittimità del verbale ex Cass. 16515/2005",
                    "legal_reasoning": "La Cassazione equipara contromano e controsenso.",
                    "norm_text_cited": ["art_143_cds"],
                    "facts_referenced": ["F1"],
                    "evidence_cited": ["DOC1"],
                }
            ],
            "requests": {
                "primary": "Rigetto dell'opposizione",
                "fallback": "Conferma sanzione anche in caso di riqualificazione",
            },
        },
        "internal_analysis": {
            "strength_self_assessments": {"response_to_ARG1": 0.6, "RARG1": 0.6},
            "key_vulnerabilities": ["Testo letterale art. 143 non chiarissimo"],
            "opponent_strongest_point": "Argomento testuale sull'art. 143",
            "gaps": [],
        },
    }
```

**Step 4: Create stub YAML files**

Copy `case.yaml` and `simulation.yaml` from design doc into `cases/gdp-milano-17928-2025.yaml` and `simulations/run-001.yaml`.

**Step 5: Verify setup**

Run: `cd /Users/cnob/athena && pip install -e ".[dev]" && pytest --collect-only`
Expected: 0 tests collected, no errors.

**Step 6: Initialize git**

Run: `cd /Users/cnob/athena && git init && git add -A && git commit -m "chore: project scaffolding"`

---

## Task 1: Pydantic Schema Models

**Files:**
- Create: `src/athena/schemas/case.py`
- Create: `src/athena/schemas/simulation.py`
- Create: `src/athena/schemas/agents.py`
- Create: `src/athena/schemas/state.py`
- Test: `tests/test_schemas.py`

**Step 1: Write failing tests**

```python
# tests/test_schemas.py
import pytest
from athena.schemas.case import CaseFile
from athena.schemas.simulation import SimulationConfig
from athena.schemas.agents import (
    AppellantBrief,
    RespondentBrief,
    JudgeDecision,
)
from athena.schemas.state import SimulationState, RunParams, ValidationResult


class TestCaseFile:
    def test_loads_valid_case(self, sample_case_data):
        case = CaseFile(**sample_case_data)
        assert case.case_id == "gdp-milano-17928-2025"
        assert len(case.parties) == 2
        assert len(case.evidence) == 2

    def test_rejects_missing_case_id(self, sample_case_data):
        del sample_case_data["case_id"]
        with pytest.raises(Exception):
            CaseFile(**sample_case_data)

    def test_extract_all_valid_ids(self, sample_case_data):
        case = CaseFile(**sample_case_data)
        ids = case.extract_all_ids()
        assert "F1" in ids
        assert "DOC1" in ids
        assert "D1" in ids
        assert "art_143_cds" in ids
        assert "cass_16515_2005" in ids
        assert "NONEXISTENT" not in ids


class TestSimulationConfig:
    def test_loads_valid_config(self):
        config = SimulationConfig(
            case_ref="gdp-milano-17928-2025",
            language="it",
            judge_profiles=[
                {
                    "id": "formalista_pro_cass",
                    "jurisprudential_orientation": "follows_cassazione",
                    "formalism": "high",
                }
            ],
            appellant_profiles=[
                {"id": "aggressivo", "style": "Attacca frontalmente."}
            ],
            temperature={"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            runs_per_combination=5,
        )
        assert config.total_runs == 5  # 1 judge × 1 style × 5

    def test_total_runs_calculation(self):
        config = SimulationConfig(
            case_ref="test",
            language="it",
            judge_profiles=[{"id": "a", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
                           {"id": "b", "jurisprudential_orientation": "distinguishes_cassazione", "formalism": "low"}],
            appellant_profiles=[{"id": "x", "style": "s1"}, {"id": "y", "style": "s2"}, {"id": "z", "style": "s3"}],
            temperature={"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            runs_per_combination=5,
        )
        assert config.total_runs == 30  # 2 × 3 × 5


class TestAppellantBrief:
    def test_valid_brief(self, sample_appellant_brief):
        brief = AppellantBrief(**sample_appellant_brief)
        assert len(brief.filed_brief.arguments) == 1
        assert brief.filed_brief.arguments[0].id == "ARG1"

    def test_rejects_empty_arguments(self):
        with pytest.raises(Exception):
            AppellantBrief(
                filed_brief={"arguments": [], "requests": {"primary": "x", "subordinate": "y"}},
                internal_analysis={
                    "strength_self_assessments": {},
                    "key_vulnerabilities": [],
                    "strongest_point": "",
                    "gaps": [],
                },
            )


class TestRespondentBrief:
    def test_valid_brief(self, sample_respondent_brief):
        brief = RespondentBrief(**sample_respondent_brief)
        assert len(brief.filed_brief.responses_to_opponent) == 1

    def test_response_references_argument(self, sample_respondent_brief):
        brief = RespondentBrief(**sample_respondent_brief)
        assert brief.filed_brief.responses_to_opponent[0].to_argument == "ARG1"


class TestJudgeDecision:
    def test_valid_decision(self):
        decision = JudgeDecision(
            preliminary_objections_ruling=[],
            case_reaches_merits=True,
            argument_evaluation=[
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.7,
                    "strengths": "Argomento testuale forte",
                    "weaknesses": "Cassazione contraria",
                    "determinative": True,
                }
            ],
            precedent_analysis={
                "cass_16515_2005": {
                    "followed": False,
                    "distinguished": True,
                    "reasoning": "Il caso è distinguibile.",
                }
            },
            verdict={
                "qualification_correct": False,
                "qualification_reasoning": "La qualificazione è errata.",
                "if_incorrect": {
                    "consequence": "reclassification",
                    "consequence_reasoning": "Va riqualificata.",
                    "applied_norm": "artt. 6-7 CdS",
                    "sanction_determined": 87,
                    "points_deducted": 0,
                },
                "costs_ruling": "a carico del Comune",
            },
            reasoning="Motivazione completa della sentenza...",
            gaps=[],
        )
        assert decision.verdict.qualification_correct is False
        assert decision.verdict.if_incorrect.consequence == "reclassification"


class TestValidationResult:
    def test_valid_result(self):
        v = ValidationResult(valid=True, errors=[], warnings=["test warning"])
        assert v.valid is True

    def test_invalid_with_errors(self):
        v = ValidationResult(valid=False, errors=["missing ID"], warnings=[])
        assert v.valid is False
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/cnob/athena && pytest tests/test_schemas.py -v`
Expected: ImportError — modules don't exist yet.

**Step 3: Implement schemas**

`src/athena/schemas/case.py`:
```python
from pydantic import BaseModel, field_validator


class Precedent(BaseModel):
    id: str
    citation: str
    holding: str
    weight: str


class ProceduralRules(BaseModel):
    rite: str
    phases: list[str]
    allowed_moves: dict[str, list[str]]


class Jurisdiction(BaseModel):
    country: str
    court: str
    venue: str
    applicable_law: list[str]
    key_precedents: list[Precedent]
    procedural_rules: ProceduralRules


class PartyObjectives(BaseModel):
    primary: str
    subordinate: str


class Party(BaseModel):
    id: str
    role: str
    type: str
    objectives: PartyObjectives
    entity: str | None = None


class Sanction(BaseModel):
    norm: str
    fine_range: list[int]
    points_deducted: int


class Stakes(BaseModel):
    current_sanction: Sanction
    alternative_sanction: Sanction
    litigation_cost_estimate: int
    non_monetary: str | None = None


class Evidence(BaseModel):
    id: str
    type: str
    description: str
    produced_by: str
    admissibility: str
    supports_facts: list[str]


class Fact(BaseModel):
    id: str
    description: str
    evidence: list[str]


class DisputedFact(BaseModel):
    id: str
    description: str
    appellant_position: str
    respondent_position: str
    depends_on_facts: list[str]


class Facts(BaseModel):
    undisputed: list[Fact]
    disputed: list[DisputedFact]


class LegalText(BaseModel):
    id: str
    norm: str
    text: str


class SeedArgument(BaseModel):
    id: str
    claim: str
    direction: str
    references_facts: list[str]


class SeedArguments(BaseModel):
    appellant: list[SeedArgument]
    respondent: list[SeedArgument]


class TimelineEvent(BaseModel):
    date: str
    event: str


class CaseFile(BaseModel):
    case_id: str
    jurisdiction: Jurisdiction
    parties: list[Party]
    stakes: Stakes
    evidence: list[Evidence]
    facts: Facts
    legal_texts: list[LegalText]
    seed_arguments: SeedArguments
    key_precedents: list[Precedent]
    timeline: list[TimelineEvent]

    def extract_all_ids(self) -> set[str]:
        """Extract all valid IDs from the case file for referential integrity checks."""
        ids: set[str] = set()
        for e in self.evidence:
            ids.add(e.id)
        for f in self.facts.undisputed:
            ids.add(f.id)
        for f in self.facts.disputed:
            ids.add(f.id)
        for lt in self.legal_texts:
            ids.add(lt.id)
        for p in self.key_precedents:
            ids.add(p.id)
        for sa in self.seed_arguments.appellant:
            ids.add(sa.id)
        for sa in self.seed_arguments.respondent:
            ids.add(sa.id)
        return ids
```

`src/athena/schemas/simulation.py`:
```python
from pydantic import BaseModel


class JudgeProfile(BaseModel):
    id: str
    jurisprudential_orientation: str  # follows_cassazione | distinguishes_cassazione
    formalism: str  # high | low


class AppellantProfile(BaseModel):
    id: str
    style: str


class TemperatureConfig(BaseModel):
    appellant: float
    respondent: float
    judge: float


class SimulationConfig(BaseModel):
    case_ref: str
    language: str
    judge_profiles: list[JudgeProfile]
    appellant_profiles: list[AppellantProfile]
    temperature: TemperatureConfig
    runs_per_combination: int

    @property
    def total_runs(self) -> int:
        return len(self.judge_profiles) * len(self.appellant_profiles) * self.runs_per_combination
```

`src/athena/schemas/agents.py`:
```python
from pydantic import BaseModel, field_validator


# --- Appellant ---

class PrecedentAddress(BaseModel):
    id: str
    strategy: str  # distinguish | criticize | limit_scope
    reasoning: str


class Argument(BaseModel):
    id: str
    type: str  # derived | new
    derived_from: str | None = None
    claim: str
    legal_reasoning: str
    norm_text_cited: list[str]
    facts_referenced: list[str]
    evidence_cited: list[str]
    precedents_addressed: list[PrecedentAddress] = []
    supports: str | None = None


class AppellantRequests(BaseModel):
    primary: str
    subordinate: str


class AppellantFiledBrief(BaseModel):
    arguments: list[Argument]
    requests: AppellantRequests

    @field_validator("arguments")
    @classmethod
    def at_least_one_argument(cls, v):
        if len(v) == 0:
            raise ValueError("Brief must contain at least one argument")
        return v


class AppellantInternalAnalysis(BaseModel):
    strength_self_assessments: dict[str, float]
    key_vulnerabilities: list[str]
    strongest_point: str
    gaps: list[str]


class AppellantBrief(BaseModel):
    filed_brief: AppellantFiledBrief
    internal_analysis: AppellantInternalAnalysis


# --- Respondent ---

class PrecedentCitation(BaseModel):
    id: str
    relevance: str


class PreliminaryObjection(BaseModel):
    id: str
    type: str  # tardività | inammissibilità | incompetenza | difetto_legittimazione
    claim: str
    legal_basis: list[str]
    reasoning: str


class ResponseToOpponent(BaseModel):
    to_argument: str
    counter_strategy: str  # rebut | distinguish | concede_partially
    counter_reasoning: str
    norm_text_cited: list[str]
    precedents_cited: list[PrecedentCitation] = []


class AffirmativeDefense(BaseModel):
    id: str
    type: str  # derived | new
    derived_from: str | None = None
    claim: str
    legal_reasoning: str
    norm_text_cited: list[str]
    facts_referenced: list[str]
    evidence_cited: list[str]


class RespondentRequests(BaseModel):
    primary: str
    fallback: str


class RespondentFiledBrief(BaseModel):
    preliminary_objections: list[PreliminaryObjection]
    responses_to_opponent: list[ResponseToOpponent]
    affirmative_defenses: list[AffirmativeDefense]
    requests: RespondentRequests


class RespondentInternalAnalysis(BaseModel):
    strength_self_assessments: dict[str, float]
    key_vulnerabilities: list[str]
    opponent_strongest_point: str
    gaps: list[str]


class RespondentBrief(BaseModel):
    filed_brief: RespondentFiledBrief
    internal_analysis: RespondentInternalAnalysis


# --- Judge ---

class PreliminaryObjectionRuling(BaseModel):
    objection_id: str
    sustained: bool
    reasoning: str


class ArgumentEvaluation(BaseModel):
    argument_id: str
    party: str  # appellant | respondent
    persuasiveness: float
    strengths: str
    weaknesses: str
    determinative: bool


class PrecedentAnalysisItem(BaseModel):
    followed: bool
    distinguished: bool
    reasoning: str


class IncorrectQualification(BaseModel):
    consequence: str  # annulment | reclassification
    consequence_reasoning: str
    applied_norm: str
    sanction_determined: int
    points_deducted: int


class Verdict(BaseModel):
    qualification_correct: bool
    qualification_reasoning: str
    if_incorrect: IncorrectQualification | None = None
    costs_ruling: str


class JudgeDecision(BaseModel):
    preliminary_objections_ruling: list[PreliminaryObjectionRuling]
    case_reaches_merits: bool
    argument_evaluation: list[ArgumentEvaluation]
    precedent_analysis: dict[str, PrecedentAnalysisItem]
    verdict: Verdict
    reasoning: str
    gaps: list[str]
```

`src/athena/schemas/state.py`:
```python
from pydantic import BaseModel
from typing import Any


class RunParams(BaseModel):
    run_id: str
    judge_profile: dict[str, Any]
    appellant_profile: dict[str, Any]
    temperature: dict[str, float]
    language: str


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/cnob/athena && pytest tests/test_schemas.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/athena/schemas/ tests/test_schemas.py
git commit -m "feat: pydantic schema models for case, simulation, agent outputs"
```

---

## Task 2: Context Builders (Information Asymmetry Layer)

**Files:**
- Create: `src/athena/simulation/context.py`
- Test: `tests/test_context_builders.py`

**Step 1: Write failing tests**

```python
# tests/test_context_builders.py
import pytest
from athena.simulation.context import (
    build_context_appellant,
    build_context_respondent,
    build_context_judge,
)


class TestBuildContextAppellant:
    def test_includes_own_seed_arguments(self, sample_case_data, sample_run_params):
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        assert "seed_arguments" in ctx
        assert ctx["seed_arguments"][0]["id"] == "SEED_ARG1"

    def test_excludes_respondent_seeds(self, sample_case_data, sample_run_params):
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        seed_ids = {s["id"] for s in ctx["seed_arguments"]}
        assert "SEED_RARG1" not in seed_ids

    def test_includes_advocacy_style(self, sample_case_data, sample_run_params):
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        assert "advocacy_style" in ctx

    def test_excludes_judge_profile(self, sample_case_data, sample_run_params):
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        assert "judge_profile" not in ctx


class TestBuildContextRespondent:
    def test_includes_only_filed_brief(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_context_respondent(
            sample_case_data, sample_run_params, sample_appellant_brief
        )
        # Should have filed_brief fields but NOT internal_analysis
        assert "arguments" in ctx["appellant_brief"]
        assert "requests" in ctx["appellant_brief"]
        assert "internal_analysis" not in ctx.get("appellant_brief", {})
        assert "key_vulnerabilities" not in ctx.get("appellant_brief", {})

    def test_excludes_judge_profile(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_context_respondent(
            sample_case_data, sample_run_params, sample_appellant_brief
        )
        assert "judge_profile" not in ctx

    def test_excludes_advocacy_style(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_context_respondent(
            sample_case_data, sample_run_params, sample_appellant_brief
        )
        assert "advocacy_style" not in ctx


class TestBuildContextJudge:
    def test_includes_both_filed_briefs(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        assert "appellant_brief" in ctx
        assert "respondent_brief" in ctx
        # Only filed_brief content
        assert "arguments" in ctx["appellant_brief"]
        assert "internal_analysis" not in ctx["appellant_brief"]

    def test_includes_judge_profile(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        assert "judge_profile" in ctx

    def test_excludes_advocacy_style(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        assert "advocacy_style" not in ctx

    def test_excludes_seed_arguments(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        assert "seed_arguments" not in ctx
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_context_builders.py -v`
Expected: ImportError.

**Step 3: Implement context builders**

```python
# src/athena/simulation/context.py
from typing import Any


def _sanitize_brief_for_opponent(brief: dict) -> dict:
    """Strip internal_analysis, return only filed_brief contents."""
    return brief["filed_brief"]


def _sanitize_brief_for_judge(brief: dict) -> dict:
    """Strip internal_analysis, return only filed_brief contents."""
    return brief["filed_brief"]


def build_context_appellant(case_data: dict, run_params: dict) -> dict:
    return {
        "facts": case_data["facts"],
        "evidence": [
            e for e in case_data["evidence"]
            if e["produced_by"] == "opponente"
            or e["admissibility"] == "uncontested"
        ],
        "legal_texts": case_data["legal_texts"],
        "precedents": case_data["key_precedents"],
        "seed_arguments": case_data["seed_arguments"]["appellant"],
        "own_party": next(
            p for p in case_data["parties"] if p["role"] == "appellant"
        ),
        "stakes": case_data["stakes"],
        "procedural_rules": case_data["jurisdiction"]["procedural_rules"],
        "advocacy_style": run_params["appellant_profile"]["style"],
    }


def build_context_respondent(
    case_data: dict, run_params: dict, appellant_brief: dict
) -> dict:
    return {
        "facts": case_data["facts"],
        "evidence": case_data["evidence"],
        "legal_texts": case_data["legal_texts"],
        "precedents": case_data["key_precedents"],
        "seed_arguments": case_data["seed_arguments"]["respondent"],
        "own_party": next(
            p for p in case_data["parties"] if p["role"] == "respondent"
        ),
        "stakes": case_data["stakes"],
        "procedural_rules": case_data["jurisdiction"]["procedural_rules"],
        "appellant_brief": _sanitize_brief_for_opponent(appellant_brief),
    }


def build_context_judge(
    case_data: dict,
    run_params: dict,
    appellant_brief: dict,
    respondent_brief: dict,
) -> dict:
    return {
        "facts": case_data["facts"],
        "evidence": case_data["evidence"],
        "legal_texts": case_data["legal_texts"],
        "precedents": case_data["key_precedents"],
        "stakes": case_data["stakes"],
        "procedural_rules": case_data["jurisdiction"]["procedural_rules"],
        "appellant_brief": _sanitize_brief_for_judge(appellant_brief),
        "respondent_brief": _sanitize_brief_for_judge(respondent_brief),
        "judge_profile": run_params["judge_profile"],
    }
```

**Step 4: Run tests**

Run: `pytest tests/test_context_builders.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/athena/simulation/context.py tests/test_context_builders.py
git commit -m "feat: context builders with information asymmetry enforcement"
```

---

## Task 3: Validation Layer

**Files:**
- Create: `src/athena/simulation/validation.py`
- Test: `tests/test_validation.py`

**Step 1: Write failing tests**

```python
# tests/test_validation.py
import pytest
from athena.simulation.validation import validate_agent_output
from athena.schemas.case import CaseFile


class TestValidateAppellant:
    def test_valid_output_passes(self, sample_case_data, sample_appellant_brief):
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=sample_appellant_brief,
            agent_role="appellant",
            case=case,
        )
        assert result.valid is True
        assert len(result.errors) == 0

    def test_phantom_id_fails(self, sample_case_data, sample_appellant_brief):
        sample_appellant_brief["filed_brief"]["arguments"][0]["facts_referenced"] = ["F999"]
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=sample_appellant_brief,
            agent_role="appellant",
            case=case,
        )
        assert result.valid is False
        assert any("F999" in e for e in result.errors)

    def test_missing_filed_brief_fails(self, sample_case_data):
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output={"internal_analysis": {}},
            agent_role="appellant",
            case=case,
        )
        assert result.valid is False

    def test_all_high_self_assessment_warns(self, sample_case_data, sample_appellant_brief):
        sample_appellant_brief["internal_analysis"]["strength_self_assessments"] = {"ARG1": 0.95}
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=sample_appellant_brief,
            agent_role="appellant",
            case=case,
        )
        assert result.valid is True
        assert any("self_assessment" in w.lower() or "0.8" in w for w in result.warnings)


class TestValidateRespondent:
    def test_missed_argument_fails(
        self, sample_case_data, sample_appellant_brief, sample_respondent_brief
    ):
        # Add a second argument to appellant that respondent doesn't address
        sample_appellant_brief["filed_brief"]["arguments"].append({
            "id": "ARG2",
            "type": "new",
            "derived_from": None,
            "claim": "Test",
            "legal_reasoning": "Test",
            "norm_text_cited": ["art_143_cds"],
            "facts_referenced": ["F1"],
            "evidence_cited": ["DOC1"],
            "precedents_addressed": [],
            "supports": None,
        })
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=sample_respondent_brief,
            agent_role="respondent",
            case=case,
            appellant_brief=sample_appellant_brief,
        )
        assert result.valid is False
        assert any("ARG2" in e for e in result.errors)


class TestValidateJudge:
    def test_missed_evaluation_fails(
        self, sample_case_data, sample_appellant_brief, sample_respondent_brief
    ):
        judge_output = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [],  # Empty — should fail
            "precedent_analysis": {},
            "verdict": {
                "qualification_correct": True,
                "qualification_reasoning": "Test",
                "costs_ruling": "test",
            },
            "reasoning": "Test",
            "gaps": [],
        }
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=judge_output,
            agent_role="judge",
            case=case,
            appellant_brief=sample_appellant_brief,
            respondent_brief=sample_respondent_brief,
        )
        assert result.valid is False
        assert any("non valutati" in e.lower() or "not evaluated" in e.lower() for e in result.errors)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation.py -v`
Expected: ImportError.

**Step 3: Implement validation**

```python
# src/athena/simulation/validation.py
from athena.schemas.case import CaseFile
from athena.schemas.state import ValidationResult


def _extract_cited_ids(data: dict, collected: set | None = None) -> set[str]:
    """Recursively extract all ID-like references from agent output."""
    if collected is None:
        collected = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if key in (
                "facts_referenced", "evidence_cited", "norm_text_cited",
                "supports_facts", "legal_basis",
            ):
                if isinstance(value, list):
                    collected.update(v for v in value if isinstance(v, str))
            elif key == "supports" and isinstance(value, str):
                collected.add(value)
            elif key in ("to_argument", "derived_from", "objection_id") and isinstance(value, str):
                collected.add(value)
            elif key == "id" and isinstance(value, str):
                # Agent-generated IDs (ARG1, RARG1) — don't validate these
                pass
            elif key == "precedents_addressed" or key == "precedents_cited":
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "id" in item:
                            collected.add(item["id"])
            else:
                _extract_cited_ids(value, collected)
    elif isinstance(data, list):
        for item in data:
            _extract_cited_ids(item, collected)
    return collected


def validate_agent_output(
    output: dict,
    agent_role: str,
    case: CaseFile,
    appellant_brief: dict | None = None,
    respondent_brief: dict | None = None,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    # --- Level 1: structure ---
    if agent_role in ("appellant", "respondent"):
        if "filed_brief" not in output:
            errors.append("Output deve contenere 'filed_brief'")
            return ValidationResult(valid=False, errors=errors, warnings=[])
        if "internal_analysis" not in output:
            errors.append("Output deve contenere 'internal_analysis'")
            return ValidationResult(valid=False, errors=errors, warnings=[])

    # --- Level 2: referential integrity ---
    valid_ids = case.extract_all_ids()
    # Also add IDs generated by appellant (for respondent/judge validation)
    if appellant_brief and "filed_brief" in appellant_brief:
        for arg in appellant_brief["filed_brief"].get("arguments", []):
            valid_ids.add(arg["id"])
    if respondent_brief and "filed_brief" in respondent_brief:
        for defense in respondent_brief["filed_brief"].get("affirmative_defenses", []):
            valid_ids.add(defense["id"])

    cited_ids = _extract_cited_ids(output)
    # Filter out None values
    cited_ids.discard(None)
    phantom_ids = cited_ids - valid_ids
    if phantom_ids:
        errors.append(f"ID inesistenti nel fascicolo: {phantom_ids}")

    # --- Level 3: completeness ---
    if agent_role == "respondent" and appellant_brief:
        appellant_arg_ids = {
            a["id"] for a in appellant_brief["filed_brief"]["arguments"]
        }
        responded_to = {
            r["to_argument"]
            for r in output["filed_brief"].get("responses_to_opponent", [])
        }
        missed = appellant_arg_ids - responded_to
        if missed:
            errors.append(f"Argomenti opponente non affrontati: {missed}")

    if agent_role == "judge" and appellant_brief:
        all_arg_ids = set()
        all_arg_ids.update(
            a["id"] for a in appellant_brief["filed_brief"]["arguments"]
        )
        if respondent_brief and "filed_brief" in respondent_brief:
            all_arg_ids.update(
                d["id"]
                for d in respondent_brief["filed_brief"].get("affirmative_defenses", [])
            )
        evaluated = {
            e["argument_id"] for e in output.get("argument_evaluation", [])
        }
        missed = all_arg_ids - evaluated
        if missed:
            errors.append(f"Argomenti non valutati dal giudice: {missed}")

    # --- Warnings ---
    if agent_role in ("appellant", "respondent"):
        assessments = output.get("internal_analysis", {}).get(
            "strength_self_assessments", {}
        )
        if assessments and all(v > 0.8 for v in assessments.values()):
            warnings.append(
                "Tutti i self_assessment > 0.8 — possibile mancanza di autocritica"
            )
        if not output.get("internal_analysis", {}).get("gaps"):
            warnings.append(
                "Campo 'gaps' vuoto — verificare completezza fascicolo"
            )

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings
    )
```

**Step 4: Run tests**

Run: `pytest tests/test_validation.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/athena/simulation/validation.py tests/test_validation.py
git commit -m "feat: validation layer with referential integrity and completeness checks"
```

---

## Task 4: Prompt Templates

**Files:**
- Create: `src/athena/agents/prompts.py`
- Test: `tests/test_prompts.py`

**Step 1: Write failing tests**

```python
# tests/test_prompts.py
import json
from athena.agents.prompts import (
    build_appellant_prompt,
    build_respondent_prompt,
    build_judge_prompt,
)


class TestAppellantPrompt:
    def test_includes_advocacy_style(self, sample_case_data, sample_run_params):
        from athena.simulation.context import build_context_appellant
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        system, user = build_appellant_prompt(ctx)
        assert "Attacca frontalmente" in system

    def test_includes_legal_texts(self, sample_case_data, sample_run_params):
        from athena.simulation.context import build_context_appellant
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        system, user = build_appellant_prompt(ctx)
        assert "art_143_cds" in user or "Art. 143" in user

    def test_returns_system_and_user(self, sample_case_data, sample_run_params):
        from athena.simulation.context import build_context_appellant
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        system, user = build_appellant_prompt(ctx)
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 100
        assert len(user) > 100


class TestRespondentPrompt:
    def test_includes_appellant_arguments(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        from athena.simulation.context import build_context_respondent
        ctx = build_context_respondent(
            sample_case_data, sample_run_params, sample_appellant_brief
        )
        system, user = build_respondent_prompt(ctx)
        assert "ARG1" in user

    def test_no_internal_analysis_in_prompt(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        from athena.simulation.context import build_context_respondent
        ctx = build_context_respondent(
            sample_case_data, sample_run_params, sample_appellant_brief
        )
        system, user = build_respondent_prompt(ctx)
        assert "key_vulnerabilities" not in user
        assert "strongest_point" not in user


class TestJudgePrompt:
    def test_includes_judge_profile(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        from athena.simulation.context import build_context_judge
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        system, user = build_judge_prompt(ctx)
        assert "follows_cassazione" in system

    def test_no_advocacy_style_in_prompt(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        from athena.simulation.context import build_context_judge
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        system, user = build_judge_prompt(ctx)
        assert "advocacy_style" not in user
        assert "Attacca frontalmente" not in user
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompts.py -v`
Expected: ImportError.

**Step 3: Implement prompt templates**

```python
# src/athena/agents/prompts.py
import json


def _format_context_block(label: str, data) -> str:
    """Format a context block for injection into prompts."""
    return f"\n## {label}\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```\n"


def build_appellant_prompt(context: dict) -> tuple[str, str]:
    """Build system + user prompt for the appellant agent.
    Returns (system_prompt, user_prompt).
    """
    system = APPELLANT_SYSTEM_PROMPT.replace(
        "{advocacy_style}", context["advocacy_style"]
    )
    user_parts = [
        "Di seguito il fascicolo del caso su cui devi lavorare.",
        _format_context_block("Fatti", context["facts"]),
        _format_context_block("Prove", context["evidence"]),
        _format_context_block("Testi normativi", context["legal_texts"]),
        _format_context_block("Precedenti", context["precedents"]),
        _format_context_block("Seed arguments", context["seed_arguments"]),
        _format_context_block("Obiettivi della tua parte", context["own_party"]),
        _format_context_block("Stakes", context["stakes"]),
        _format_context_block("Regole procedurali", context["procedural_rules"]),
        "\nProduci la tua memoria difensiva in formato JSON come specificato nelle istruzioni.",
    ]
    return system, "\n".join(user_parts)


def build_respondent_prompt(context: dict) -> tuple[str, str]:
    """Build system + user prompt for the respondent agent."""
    system = RESPONDENT_SYSTEM_PROMPT
    user_parts = [
        "Di seguito il fascicolo del caso e la memoria dell'opponente.",
        _format_context_block("Fatti", context["facts"]),
        _format_context_block("Prove", context["evidence"]),
        _format_context_block("Testi normativi", context["legal_texts"]),
        _format_context_block("Precedenti", context["precedents"]),
        _format_context_block("Seed arguments difensivi", context["seed_arguments"]),
        _format_context_block("Obiettivi della tua parte", context["own_party"]),
        _format_context_block("Stakes", context["stakes"]),
        _format_context_block("Regole procedurali", context["procedural_rules"]),
        _format_context_block("Memoria dell'opponente (depositata)", context["appellant_brief"]),
        "\nProduci la tua memoria di costituzione in formato JSON come specificato nelle istruzioni.",
    ]
    return system, "\n".join(user_parts)


def build_judge_prompt(context: dict) -> tuple[str, str]:
    """Build system + user prompt for the judge agent."""
    profile = context["judge_profile"]
    system = JUDGE_SYSTEM_PROMPT.replace(
        "{jurisprudential_orientation}", profile["jurisprudential_orientation"]
    ).replace(
        "{formalism}", profile["formalism"]
    )
    user_parts = [
        "Di seguito il fascicolo completo e le memorie delle parti.",
        _format_context_block("Fatti", context["facts"]),
        _format_context_block("Prove", context["evidence"]),
        _format_context_block("Testi normativi", context["legal_texts"]),
        _format_context_block("Precedenti", context["precedents"]),
        _format_context_block("Stakes", context["stakes"]),
        _format_context_block("Regole procedurali", context["procedural_rules"]),
        _format_context_block("Memoria dell'opponente (depositata)", context["appellant_brief"]),
        _format_context_block("Memoria del Comune (depositata)", context["respondent_brief"]),
        "\nProduci la tua sentenza in formato JSON come specificato nelle istruzioni.",
    ]
    return system, "\n".join(user_parts)


# System prompts — full text from design doc
# These are stored as module-level constants

APPELLANT_SYSTEM_PROMPT = """Sei l'avvocato dell'opponente in un procedimento di opposizione a sanzione amministrativa davanti al Giudice di Pace.

## Ruolo
Rappresenti la parte che ha ricevuto la sanzione e ne contesta la legittimità. Produci una memoria difensiva.

## Obiettivo
- Principale: annullamento del verbale
- Subordinato: riqualificazione della sanzione sotto la norma corretta

## Stile di advocacy (parametrico)
{advocacy_style}

Questo parametro orienta il tuo approccio argomentativo. Non cambia i fatti né le norme — cambia come li presenti e quale strategia priorizzi.

## Gerarchia delle fonti (diritto italiano)
Costituzione > Legge ordinaria > Regolamento > Giurisprudenza di Cassazione > Prassi.
La Cassazione è autorevole ma NON vincolante. Puoi argomentare contro un orientamento di Cassazione motivando adeguatamente. In caso di contrasto tra testo di legge e interpretazione giurisprudenziale, prevale il testo.

## Output — JSON strutturato

L'output è diviso in due blocchi:
- "filed_brief": ciò che viene depositato e che l'avversario e il giudice vedranno
- "internal_analysis": work product interno, visibile solo all'analisi strategica

{
  "filed_brief": {
    "arguments": [
      {
        "id": "ARG1",
        "type": "derived | new",
        "derived_from": "SEED_ARG1 | null",
        "claim": "[1 frase]",
        "legal_reasoning": "[3-8 frasi strutturate]",
        "norm_text_cited": ["art_143_cds"],
        "facts_referenced": ["F1", "F3"],
        "evidence_cited": ["DOC1"],
        "precedents_addressed": [
          {
            "id": "cass_16515_2005",
            "strategy": "distinguish | criticize | limit_scope",
            "reasoning": "[2-4 frasi]"
          }
        ],
        "supports": "ARG1 | null"
      }
    ],
    "requests": {
      "primary": "[1-2 frasi]",
      "subordinate": "[1-2 frasi]"
    }
  },
  "internal_analysis": {
    "strength_self_assessments": {
      "ARG1": 0.0
    },
    "key_vulnerabilities": ["[1 frase ciascuna]"],
    "strongest_point": "[1-2 frasi]",
    "gaps": ["Elementi mancanti nel fascicolo"]
  }
}

## Esempio di buon reasoning

EVITA: "La norma non si applica perché la situazione è diversa."

PREFERISCI: "Il testo dell'art. [X] comma [Y] recita '[citazione dal testo fornito]'. Questa formulazione presuppone [condizione specifica]. Nel caso di specie, il fatto [ID fatto] dimostra che tale condizione non ricorre, in quanto [spiegazione]. La fattispecie concreta è invece tipizzata dall'art. [Z] che disciplina [ambito], come risulta dal testo fornito: '[citazione]'."

## Vincoli
- Ragiona ESCLUSIVAMENTE sui testi normativi forniti in input. Se hai bisogno di una norma non fornita, segnalala in "gaps".
- Puoi referenziare SOLO ID (fatti, prove, norme, precedenti) presenti nel fascicolo. Non inventare ID.
- Devi affrontare la giurisprudenza sfavorevole — non puoi ignorarla.
- I self_assessment devono essere onesti. 0.3 = argomento debole, 0.7 = solido, 0.9 = molto forte.
- Rispondi ESCLUSIVAMENTE con il JSON richiesto, senza testo aggiuntivo."""


RESPONDENT_SYSTEM_PROMPT = """Sei l'avvocato del Comune di Milano in un procedimento di opposizione a sanzione amministrativa davanti al Giudice di Pace.

## Ruolo
Rappresenti l'ente che ha emesso la sanzione tramite la Polizia Locale. Produci una memoria di costituzione.

## Obiettivo
- Principale: conferma integrale del verbale, rigetto dell'opposizione
- Subordinato: anche in caso di riqualificazione, la sanzione resta dovuta

## Gerarchia delle fonti (diritto italiano)
Costituzione > Legge ordinaria > Regolamento > Giurisprudenza di Cassazione > Prassi.
La Cassazione è autorevole ma NON vincolante. Se ti è favorevole, usala esplicitamente ma riconoscine eventuali limiti.

## Strategia — ordine obbligatorio
1. ECCEZIONI PRELIMINARI: valuta se esistono eccezioni di rito fondate. Se non ne trovi di fondate, lascia la lista vuota.
2. RISPOSTE NEL MERITO: rispondi a ogni argomento dell'opponente. Per ciascuno scegli: rebut, distinguish, concede_partially.
3. DIFESE AFFERMATIVE: sviluppa argomenti autonomi.

## Output — JSON strutturato

{
  "filed_brief": {
    "preliminary_objections": [],
    "responses_to_opponent": [
      {
        "to_argument": "ARG1",
        "counter_strategy": "rebut | distinguish | concede_partially",
        "counter_reasoning": "[3-8 frasi]",
        "norm_text_cited": ["art_143_cds"],
        "precedents_cited": [{"id": "cass_16515_2005", "relevance": "[1-2 frasi]"}]
      }
    ],
    "affirmative_defenses": [
      {
        "id": "RARG1",
        "type": "derived | new",
        "derived_from": "SEED_RARG1 | null",
        "claim": "[1 frase]",
        "legal_reasoning": "[3-8 frasi]",
        "norm_text_cited": ["..."],
        "facts_referenced": ["F1"],
        "evidence_cited": ["DOC1"]
      }
    ],
    "requests": {
      "primary": "[1-2 frasi]",
      "fallback": "[1-2 frasi]"
    }
  },
  "internal_analysis": {
    "strength_self_assessments": {},
    "key_vulnerabilities": ["..."],
    "opponent_strongest_point": "[1-2 frasi]",
    "gaps": []
  }
}

## Esempio di buon reasoning

EVITA: "La Cassazione ha stabilito che sono equivalenti, quindi il verbale è legittimo."

PREFERISCI: "L'opponente argomenta (ARG1) che l'art. [X] non copre la fattispecie. Questa tesi va disattesa. La Cass. n. [Y] ha affrontato specificamente la questione, stabilendo che '[citazione dalla massima fornita]'."

## Vincoli
- Ragiona ESCLUSIVAMENTE sui testi normativi forniti. Segnala lacune in "gaps".
- Referenzia SOLO ID presenti nel fascicolo.
- Rispondi a OGNI argomento dell'opponente.
- "opponent_strongest_point" è obbligatorio.
- Rispondi ESCLUSIVAMENTE con il JSON richiesto, senza testo aggiuntivo."""


JUDGE_SYSTEM_PROMPT = """Sei il Giudice di Pace di Milano. Decidi un procedimento di opposizione a sanzione amministrativa ex art. 204-bis CdS.

## Ruolo
Valuti le memorie depositate da entrambe le parti e pronunci sentenza.

## Profilo

Orientamento giurisprudenziale: {jurisprudential_orientation}
- "follows_cassazione": tendi a seguire la Cassazione, valorizzando uniformità e certezza del diritto
- "distinguishes_cassazione": valuti criticamente i precedenti, dai più peso al testo letterale

Formalismo: {formalism}
- "high": dai peso significativo ai vizi formali, la precisione dell'azione amministrativa è un valore in sé
- "low": guardi alla sostanza del fatto e alla ratio della norma

Questi parametri orientano il ragionamento. NON predeterminano l'esito.

## Gerarchia delle fonti
Costituzione > Legge ordinaria > Regolamento > Giurisprudenza di Cassazione > Prassi.
La Cassazione è autorevole ma NON vincolante. Puoi discostarti motivando adeguatamente.

## Struttura della decisione — ordine obbligatorio
1. SVOLGIMENTO DEL PROCESSO
2. ECCEZIONI PRELIMINARI — se assorbenti, case_reaches_merits = false
3. QUALIFICAZIONE GIURIDICA
4. CONSEGUENZE — annullamento o riqualificazione (questioni distinte)
5. P.Q.M.

## Output — JSON strutturato

{
  "preliminary_objections_ruling": [],
  "case_reaches_merits": true,
  "argument_evaluation": [
    {
      "argument_id": "ARG1",
      "party": "appellant | respondent",
      "persuasiveness": 0.0,
      "strengths": "[1-3 frasi]",
      "weaknesses": "[1-3 frasi]",
      "determinative": true | false
    }
  ],
  "precedent_analysis": {
    "cass_16515_2005": {
      "followed": true | false,
      "distinguished": true | false,
      "reasoning": "[3-5 frasi]"
    }
  },
  "verdict": {
    "qualification_correct": true | false,
    "qualification_reasoning": "[5-10 frasi]",
    "if_incorrect": {
      "consequence": "annulment | reclassification",
      "consequence_reasoning": "[3-5 frasi]",
      "applied_norm": "artt. 6-7 CdS",
      "sanction_determined": 0,
      "points_deducted": 0
    },
    "costs_ruling": "a carico di [parte]"
  },
  "reasoning": "[500-1500 parole] Motivazione completa.",
  "gaps": []
}

## Vincoli
- Valuta OGNI argomento di entrambe le parti.
- Se riqualifichi: determina la sanzione specifica usando gli importi nelle stakes.
- Ragiona ESCLUSIVAMENTE sui testi normativi forniti.
- Referenzia SOLO ID presenti nel fascicolo.
- NON produrre probabilità — tu decidi.
- qualification_correct e if_incorrect sono DUE questioni distinte.
- Rispondi ESCLUSIVAMENTE con il JSON richiesto, senza testo aggiuntivo."""
```

**Step 4: Run tests**

Run: `pytest tests/test_prompts.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/athena/agents/prompts.py tests/test_prompts.py
git commit -m "feat: prompt templates for all 3 agents with Italian legal reasoning"
```

---

## Task 5: LLM Integration Layer

**Files:**
- Create: `src/athena/agents/llm.py`
- Test: `tests/test_llm.py`

This task creates the abstraction that calls the LLM and parses JSON output. Integration tests require the model, unit tests use mocks.

**Step 1: Write failing tests**

```python
# tests/test_llm.py
import pytest
import json
from unittest.mock import patch, MagicMock
from athena.agents.llm import invoke_llm, parse_json_response


class TestParseJsonResponse:
    def test_parses_clean_json(self):
        raw = '{"key": "value"}'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parses_json_in_markdown_block(self):
        raw = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parses_json_with_trailing_text(self):
        raw = '{"key": "value"}\n\nSome explanation after.'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_raises_on_invalid_json(self):
        raw = "This is not JSON at all"
        with pytest.raises(ValueError):
            parse_json_response(raw)


class TestInvokeLLM:
    @patch("athena.agents.llm._call_model")
    def test_returns_parsed_dict(self, mock_call):
        mock_call.return_value = '{"test": true}'
        result = invoke_llm("system", "user", temperature=0.5)
        assert result == {"test": True}

    @patch("athena.agents.llm._call_model")
    def test_raises_on_invalid_response(self, mock_call):
        mock_call.return_value = "not json"
        with pytest.raises(ValueError):
            invoke_llm("system", "user", temperature=0.5)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v`

**Step 3: Implement LLM layer**

```python
# src/athena/agents/llm.py
import json
import re
from mlx_lm import load, generate


_MODEL = None
_TOKENIZER = None
_MODEL_PATH = "mlx-community/Qwen3.5-122B-A10B-4bit"


def _ensure_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        _MODEL, _TOKENIZER = load(_MODEL_PATH)


def _call_model(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call the MLX model and return raw text response."""
    _ensure_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(
        _MODEL,
        _TOKENIZER,
        prompt=prompt,
        temp=temperature,
        max_tokens=8192,
    )
    return response


def parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown blocks."""
    # Try direct parse first
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def invoke_llm(
    system_prompt: str, user_prompt: str, temperature: float
) -> dict:
    """Invoke LLM and return parsed JSON dict."""
    raw = _call_model(system_prompt, user_prompt, temperature)
    return parse_json_response(raw)
```

**Step 4: Run tests**

Run: `pytest tests/test_llm.py -v`
Expected: All PASS (uses mocks, no model needed).

**Step 5: Commit**

```bash
git add src/athena/agents/llm.py tests/test_llm.py
git commit -m "feat: LLM integration layer with MLX and JSON parsing"
```

---

## Task 6: LangGraph Single Run

**Files:**
- Create: `src/athena/simulation/graph.py`
- Test: `tests/test_graph.py`

**Step 1: Write failing tests**

```python
# tests/test_graph.py
import pytest
from unittest.mock import patch
from athena.simulation.graph import build_graph, run_single


class TestBuildGraph:
    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        # LangGraph compiled graph should have the node names
        assert graph is not None


class TestRunSingle:
    @patch("athena.simulation.graph.invoke_llm")
    def test_end_to_end_with_mock(
        self,
        mock_llm,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        """Full run with mocked LLM returning valid outputs."""
        mock_llm.side_effect = [
            sample_appellant_brief,   # appellant
            sample_respondent_brief,  # respondent
            {                         # judge
                "preliminary_objections_ruling": [],
                "case_reaches_merits": True,
                "argument_evaluation": [
                    {
                        "argument_id": "ARG1",
                        "party": "appellant",
                        "persuasiveness": 0.7,
                        "strengths": "Forte",
                        "weaknesses": "Cassazione contraria",
                        "determinative": True,
                    },
                    {
                        "argument_id": "RARG1",
                        "party": "respondent",
                        "persuasiveness": 0.5,
                        "strengths": "Precedente",
                        "weaknesses": "Testo contrario",
                        "determinative": False,
                    },
                ],
                "precedent_analysis": {
                    "cass_16515_2005": {
                        "followed": False,
                        "distinguished": True,
                        "reasoning": "Caso distinguibile.",
                    }
                },
                "verdict": {
                    "qualification_correct": False,
                    "qualification_reasoning": "La qualificazione è errata.",
                    "if_incorrect": {
                        "consequence": "reclassification",
                        "consequence_reasoning": "Va riqualificata.",
                        "applied_norm": "artt. 6-7 CdS",
                        "sanction_determined": 87,
                        "points_deducted": 0,
                    },
                    "costs_ruling": "a carico del Comune",
                },
                "reasoning": "Motivazione della sentenza.",
                "gaps": [],
            },
        ]

        result = run_single(sample_case_data, sample_run_params)

        assert result["error"] is None
        assert result["appellant_brief"] is not None
        assert result["respondent_brief"] is not None
        assert result["judge_decision"] is not None
        assert result["judge_decision"]["verdict"]["qualification_correct"] is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph.py -v`

**Step 3: Implement graph**

```python
# src/athena/simulation/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any

from athena.simulation.context import (
    build_context_appellant,
    build_context_respondent,
    build_context_judge,
)
from athena.simulation.validation import validate_agent_output
from athena.agents.prompts import (
    build_appellant_prompt,
    build_respondent_prompt,
    build_judge_prompt,
)
from athena.agents.llm import invoke_llm
from athena.schemas.case import CaseFile


class GraphState(TypedDict):
    case: dict
    params: dict
    appellant_brief: dict | None
    appellant_validation: dict | None
    respondent_brief: dict | None
    respondent_validation: dict | None
    judge_decision: dict | None
    judge_validation: dict | None
    retry_count: int
    error: str | None


MAX_RETRIES = 2


def _node_appellant(state: GraphState) -> dict:
    ctx = build_context_appellant(state["case"], state["params"])
    system, user = build_appellant_prompt(ctx)
    temp = state["params"]["temperature"]["appellant"]
    try:
        output = invoke_llm(system, user, temp)
        case = CaseFile(**state["case"])
        validation = validate_agent_output(output, "appellant", case)
        if not validation.valid and state["retry_count"] < MAX_RETRIES:
            error_feedback = "\n".join(validation.errors)
            retry_user = f"{user}\n\n## ERRORI DA CORREGGERE\n{error_feedback}\n\nRiproduci l'output completo corretto."
            output = invoke_llm(system, retry_user, temp)
            validation = validate_agent_output(output, "appellant", case)
            return {
                "appellant_brief": output,
                "appellant_validation": validation.model_dump(),
                "retry_count": state["retry_count"] + 1,
            }
        return {
            "appellant_brief": output,
            "appellant_validation": validation.model_dump(),
        }
    except Exception as e:
        return {"error": f"Appellant failed: {e}"}


def _node_respondent(state: GraphState) -> dict:
    if state.get("error"):
        return {}
    ctx = build_context_respondent(
        state["case"], state["params"], state["appellant_brief"]
    )
    system, user = build_respondent_prompt(ctx)
    temp = state["params"]["temperature"]["respondent"]
    try:
        output = invoke_llm(system, user, temp)
        case = CaseFile(**state["case"])
        validation = validate_agent_output(
            output, "respondent", case,
            appellant_brief=state["appellant_brief"],
        )
        if not validation.valid and state["retry_count"] < MAX_RETRIES:
            error_feedback = "\n".join(validation.errors)
            retry_user = f"{user}\n\n## ERRORI DA CORREGGERE\n{error_feedback}\n\nRiproduci l'output completo corretto."
            output = invoke_llm(system, retry_user, temp)
            validation = validate_agent_output(
                output, "respondent", case,
                appellant_brief=state["appellant_brief"],
            )
            return {
                "respondent_brief": output,
                "respondent_validation": validation.model_dump(),
                "retry_count": state["retry_count"] + 1,
            }
        return {
            "respondent_brief": output,
            "respondent_validation": validation.model_dump(),
        }
    except Exception as e:
        return {"error": f"Respondent failed: {e}"}


def _node_judge(state: GraphState) -> dict:
    if state.get("error"):
        return {}
    ctx = build_context_judge(
        state["case"],
        state["params"],
        state["appellant_brief"],
        state["respondent_brief"],
    )
    system, user = build_judge_prompt(ctx)
    temp = state["params"]["temperature"]["judge"]
    try:
        output = invoke_llm(system, user, temp)
        case = CaseFile(**state["case"])
        validation = validate_agent_output(
            output, "judge", case,
            appellant_brief=state["appellant_brief"],
            respondent_brief=state["respondent_brief"],
        )
        if not validation.valid and state["retry_count"] < MAX_RETRIES:
            error_feedback = "\n".join(validation.errors)
            retry_user = f"{user}\n\n## ERRORI DA CORREGGERE\n{error_feedback}\n\nRiproduci l'output completo corretto."
            output = invoke_llm(system, retry_user, temp)
            validation = validate_agent_output(
                output, "judge", case,
                appellant_brief=state["appellant_brief"],
                respondent_brief=state["respondent_brief"],
            )
            return {
                "judge_decision": output,
                "judge_validation": validation.model_dump(),
                "retry_count": state["retry_count"] + 1,
            }
        return {
            "judge_decision": output,
            "judge_validation": validation.model_dump(),
        }
    except Exception as e:
        return {"error": f"Judge failed: {e}"}


def _should_continue(state: GraphState) -> str:
    if state.get("error"):
        return END
    return "next"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("appellant", _node_appellant)
    graph.add_node("respondent", _node_respondent)
    graph.add_node("judge", _node_judge)

    graph.set_entry_point("appellant")
    graph.add_edge("appellant", "respondent")
    graph.add_edge("respondent", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


def run_single(case_data: dict, run_params: dict) -> dict:
    """Run a single simulation and return results."""
    graph = build_graph()
    initial_state: GraphState = {
        "case": case_data,
        "params": run_params,
        "appellant_brief": None,
        "appellant_validation": None,
        "respondent_brief": None,
        "respondent_validation": None,
        "judge_decision": None,
        "judge_validation": None,
        "retry_count": 0,
        "error": None,
    }
    return graph.invoke(initial_state)
```

**Step 4: Run tests**

Run: `pytest tests/test_graph.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/athena/simulation/graph.py tests/test_graph.py
git commit -m "feat: LangGraph single run with sequential appellant → respondent → judge"
```

---

## Task 7: Monte Carlo Orchestrator + Aggregator

**Files:**
- Create: `src/athena/simulation/orchestrator.py`
- Create: `src/athena/simulation/aggregator.py`
- Test: `tests/test_aggregator.py`

**Step 1: Write failing tests for aggregator**

```python
# tests/test_aggregator.py
import pytest
from athena.simulation.aggregator import aggregate_results, wilson_ci


class TestWilsonCI:
    def test_all_successes(self):
        low, high = wilson_ci(5, 5)
        assert low > 0.5
        assert high <= 1.0

    def test_no_successes(self):
        low, high = wilson_ci(0, 5)
        assert low >= 0.0
        assert high < 0.5

    def test_zero_trials(self):
        low, high = wilson_ci(0, 0)
        assert low == 0.0 and high == 0.0


class TestAggregateResults:
    def test_basic_aggregation(self):
        results = [
            {
                "run_id": "jp1__style1__000",
                "judge_profile": "formalista_pro_cass",
                "appellant_profile": "aggressivo",
                "judge_decision": {
                    "verdict": {"qualification_correct": False, "if_incorrect": {"consequence": "reclassification"}},
                    "argument_evaluation": [
                        {"argument_id": "ARG1", "persuasiveness": 0.7, "determinative": True, "party": "appellant"},
                    ],
                    "precedent_analysis": {"cass_16515_2005": {"followed": False, "distinguished": True}},
                },
            },
            {
                "run_id": "jp1__style1__001",
                "judge_profile": "formalista_pro_cass",
                "appellant_profile": "aggressivo",
                "judge_decision": {
                    "verdict": {"qualification_correct": True},
                    "argument_evaluation": [
                        {"argument_id": "ARG1", "persuasiveness": 0.4, "determinative": False, "party": "appellant"},
                    ],
                    "precedent_analysis": {"cass_16515_2005": {"followed": True, "distinguished": False}},
                },
            },
        ]

        agg = aggregate_results(results, total_expected=2)
        key = ("formalista_pro_cass", "aggressivo")
        assert key in agg["probability_table"]
        assert agg["probability_table"][key]["n_runs"] == 2
        assert agg["probability_table"][key]["p_rejection"] == 0.5
        assert agg["total_runs"] == 2
        assert agg["failed_runs"] == 0

    def test_argument_effectiveness(self):
        results = [
            {
                "run_id": "test__test__000",
                "judge_profile": "jp1",
                "appellant_profile": "s1",
                "judge_decision": {
                    "verdict": {"qualification_correct": False, "if_incorrect": {"consequence": "annulment"}},
                    "argument_evaluation": [
                        {"argument_id": "ARG1", "persuasiveness": 0.9, "determinative": True, "party": "appellant"},
                        {"argument_id": "ARG2", "persuasiveness": 0.3, "determinative": False, "party": "appellant"},
                    ],
                    "precedent_analysis": {},
                },
            },
        ]
        agg = aggregate_results(results, total_expected=1)
        assert agg["argument_effectiveness"]["ARG1"]["mean_persuasiveness"] == 0.9
        assert agg["argument_effectiveness"]["ARG2"]["mean_persuasiveness"] == 0.3
        assert agg["argument_effectiveness"]["ARG1"]["determinative_rate"] == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_aggregator.py -v`

**Step 3: Implement aggregator and orchestrator**

Implement `src/athena/simulation/aggregator.py` with the `wilson_ci` and `aggregate_results` functions from the design doc (section 3.6). Implement `src/athena/simulation/orchestrator.py` with the `run_monte_carlo` function from section 3.5.

**Step 4: Run tests**

Run: `pytest tests/test_aggregator.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/athena/simulation/orchestrator.py src/athena/simulation/aggregator.py tests/test_aggregator.py
git commit -m "feat: Monte Carlo orchestrator and statistical aggregator with Wilson CI"
```

---

## Task 8: Output Generators

**Files:**
- Create: `src/athena/output/table.py`
- Create: `src/athena/output/decision_tree.py`
- Create: `src/athena/output/memo.py`
- Test: `tests/test_output.py`

Implement the 3 output generators from design doc section 3.7. The table and decision tree are pure Python. The memo generator calls the Synthesizer LLM.

Tests should verify table formatting, decision tree logic, and dominated strategy detection. Memo test uses a mock LLM.

**Step 1-5:** Same TDD pattern. Tests verify: table has correct dimensions, CI are shown, decision tree finds best style per profile, dominated strategies detected, memo prompt includes all data sections.

**Commit:** `feat: output generators — probability table, decision tree, strategic memo`

---

## Task 9: Capability Test (Step 0 from design)

**Files:**
- Create: `scripts/capability_test.py`

This is a manual validation script, not part of the automated test suite. It loads the MLX model, sends one prompt per agent (using hardcoded context from the test fixtures), and prints the results for human inspection.

```python
# scripts/capability_test.py
"""
Run: python scripts/capability_test.py

Tests the MLX model's ability to:
1. Produce valid JSON
2. Follow the agent prompt structure
3. Differentiate between judge profiles
4. Maintain referential integrity
"""
```

Run manually, inspect results, document findings in `docs/plans/capability-test-results.md`.

**Commit:** `chore: capability test script for MLX model validation`

---

## Task 10: CLI Entry Point + End-to-End Test

**Files:**
- Create: `src/athena/cli.py`
- Modify: `pyproject.toml` (add script entry)

Simple CLI: `athena run --case cases/gdp-milano-17928-2025.yaml --simulation simulations/run-001.yaml --output results/`

End-to-end test: run with N=1 on one combination, verify all 3 outputs are generated.

**Commit:** `feat: CLI entry point for running simulations`

---

## Build Sequence

```
Task 0: Scaffolding          ← no dependencies
Task 1: Schemas              ← depends on 0
Task 2: Context Builders     ← depends on 1
Task 3: Validation           ← depends on 1
Task 4: Prompts              ← depends on 2
Task 5: LLM Integration      ← no dependencies (can parallel with 2-4)
Task 6: Graph                ← depends on 2, 3, 4, 5
Task 7: Orchestrator + Agg   ← depends on 6
Task 8: Output Generators    ← depends on 7
Task 9: Capability Test      ← depends on 5 (run manually)
Task 10: CLI + E2E           ← depends on 7, 8
```

Tasks 2, 3, 4 can run in parallel after Task 1.
Task 5 can run in parallel with Tasks 2-4.
Task 9 should run before Task 6 to validate model choice.
