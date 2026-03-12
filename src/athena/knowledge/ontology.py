# src/athena/knowledge/ontology.py
"""Pydantic entity and edge models for the ATHENA legal knowledge graph.

Maps 1:1 from schemas/case.py and schemas/structured_output.py to graph
node/edge types. Used for serialization to/from Neo4j properties.
"""

from pydantic import BaseModel


# ---- Node Models ----

class CaseNode(BaseModel):
    case_id: str
    title: str = ""
    jurisdiction_country: str = ""
    jurisdiction_court: str = ""
    jurisdiction_venue: str = ""

    @classmethod
    def from_case_data(cls, case_data: dict) -> "CaseNode":
        jur = case_data.get("jurisdiction", {})
        if isinstance(jur, dict):
            country = jur.get("country", "")
            court = jur.get("court", "")
            venue = jur.get("venue", "")
        else:
            country = court = venue = ""
        return cls(
            case_id=case_data["case_id"],
            title=case_data.get("title", case_data["case_id"]),
            jurisdiction_country=country,
            jurisdiction_court=court,
            jurisdiction_venue=venue,
        )


class PartyNode(BaseModel):
    party_id: str
    role: str
    type: str
    primary_objective: str = ""
    subordinate_objective: str = ""

    @classmethod
    def from_party(cls, party: dict) -> "PartyNode":
        obj = party.get("objectives", {})
        return cls(
            party_id=party["id"],
            role=party["role"],
            type=party["type"],
            primary_objective=obj.get("primary", ""),
            subordinate_objective=obj.get("subordinate", ""),
        )


class FactNode(BaseModel):
    fact_id: str
    description: str
    is_disputed: bool = False
    positions: dict[str, str] | None = None  # party_id → position (disputed only)


class EvidenceNode(BaseModel):
    evidence_id: str
    type: str
    description: str
    produced_by: str
    admissibility: str = ""


class LegalTextNode(BaseModel):
    legal_text_id: str
    norm: str
    text: str
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None
    text_embedding: list[float] | None = None


class PrecedentNode(BaseModel):
    precedent_id: str
    citation: str
    holding: str
    weight: str = ""
    # Aggregated stats (updated by stats_loader)
    followed_rate: float | None = None
    distinguished_rate: float | None = None


class SeedArgumentNode(BaseModel):
    seed_arg_id: str
    claim: str
    direction: str
    party_id: str
    references_facts: list[str] = []
    claim_embedding: list[float] | None = None
    # Aggregated stats (updated by stats_loader)
    mean_persuasiveness: float | None = None
    std_persuasiveness: float | None = None
    determinative_rate: float | None = None


class ArgumentNode(BaseModel):
    argument_id: str
    type: str  # "derived" | "new"
    derived_from: str | None = None
    claim: str
    legal_reasoning: str = ""
    run_id: str = ""
    party_id: str = ""
    norm_text_cited: list[str] = []
    facts_referenced: list[str] = []
    evidence_cited: list[str] = []
    claim_embedding: list[float] | None = None


class ResponseNode(BaseModel):
    response_id: str  # composite: run_id + to_argument
    to_argument: str
    counter_strategy: str
    counter_reasoning: str = ""
    run_id: str = ""
    party_id: str = ""


class JudgeDecisionNode(BaseModel):
    run_id: str
    qualification_correct: bool
    consequence: str | None = None  # "annulment" | "reclassification" | None
    reasoning: str = ""


class SimRunNode(BaseModel):
    run_id: str
    judge_profile_id: str = ""
    party_profile_ids: dict[str, str] = {}  # party_id → profile_id


class BATNANode(BaseModel):
    key: str  # case_id + party_id
    case_id: str
    party_id: str
    expected_value: float
    expected_value_range_low: float = 0.0
    expected_value_range_high: float = 0.0
    best_strategy: str | None = None


class SettlementNode(BaseModel):
    key: str  # case_id
    case_id: str
    settlement_exists: bool
    zopa_low: float | None = None
    zopa_high: float | None = None
    nash_solution: float | None = None
    surplus: float = 0.0


class SensitivityNode(BaseModel):
    key: str  # case_id + parameter
    case_id: str
    parameter: str
    impact: float
    threshold: float | None = None
    base_value: float = 0.0


class IracNode(BaseModel):
    irac_id: str          # case_id + "__" + seed_arg_id
    seed_arg_id: str
    case_id: str
    issue: str
    rule: str
    application: str
    conclusion: str


# ---- Edge type constants ----

EDGE_HAS_PARTY = "HAS_PARTY"
EDGE_HAS_FACT = "HAS_FACT"
EDGE_HAS_EVIDENCE = "HAS_EVIDENCE"
EDGE_HAS_LEGAL_TEXT = "HAS_LEGAL_TEXT"
EDGE_HAS_PRECEDENT = "HAS_PRECEDENT"
EDGE_HAS_SEED_ARGUMENT = "HAS_SEED_ARGUMENT"
EDGE_SUPPORTS_FACT = "SUPPORTS_FACT"
EDGE_REFERENCES_FACT = "REFERENCES_FACT"
EDGE_DEPENDS_ON = "DEPENDS_ON"
EDGE_POSITION = "POSITION"
EDGE_PRODUCED_BY = "PRODUCED_BY"
EDGE_DERIVES_FROM = "DERIVES_FROM"
EDGE_CITES_NORM = "CITES_NORM"
EDGE_CITES_EVIDENCE = "CITES_EVIDENCE"
EDGE_ADDRESSES_PRECEDENT = "ADDRESSES_PRECEDENT"
EDGE_RESPONDS_TO = "RESPONDS_TO"
EDGE_EVALUATES = "EVALUATES"
EDGE_FOLLOWS_PRECEDENT = "FOLLOWS_PRECEDENT"
EDGE_PRODUCED_IN = "PRODUCED_IN"
EDGE_HAS_BATNA = "HAS_BATNA"
EDGE_HAS_SETTLEMENT = "HAS_SETTLEMENT"
EDGE_HAS_SENSITIVITY = "HAS_SENSITIVITY"
EDGE_SUPERSEDES = "SUPERSEDES"
EDGE_HAS_IRAC = "HAS_IRAC"
