# tests/test_knowledge_ingestion.py
"""Tests for knowledge graph ingestion — case, results, stats, game theory.

Requires a running Neo4j instance. Skipped when KG not enabled.
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ATHENA_KG_ENABLED") != "1",
    reason="Knowledge graph not enabled (set ATHENA_KG_ENABLED=1 and NEO4J_PASSWORD)",
)


@pytest.fixture
def sample_case_data():
    """Minimal case data with all entity types."""
    return {
        "case_id": "test-kg-001",
        "title": "KG Test Case",
        "jurisdiction": {
            "country": "Italy",
            "court": "Giudice di Pace",
            "venue": "Milano",
            "applicable_law": ["CdS"],
            "key_precedents": [],
            "procedural_rules": {"rite": "opposizione", "phases": [], "allowed_moves": {}},
        },
        "parties": [
            {"id": "appellant_001", "role": "appellant", "type": "natural_person",
             "objectives": {"primary": "annul", "subordinate": "reduce"}},
            {"id": "respondent_001", "role": "respondent", "type": "public_entity",
             "objectives": {"primary": "confirm", "subordinate": "costs"}},
        ],
        "facts": {
            "undisputed": [
                {"id": "F01", "description": "Vehicle was stopped", "evidence": ["E01"]},
            ],
            "disputed": [
                {"id": "DF01", "description": "Speed limit sign present",
                 "positions": {"appellant_001": "Not visible", "respondent_001": "Clearly visible"},
                 "depends_on_facts": ["F01"]},
            ],
        },
        "evidence": [
            {"id": "E01", "type": "document", "description": "Police report",
             "produced_by": "respondent_001", "admissibility": "uncontested",
             "supports_facts": ["F01"]},
            {"id": "E02", "type": "photo", "description": "Dashboard cam",
             "produced_by": "appellant_001", "admissibility": "contested",
             "supports_facts": ["DF01"]},
        ],
        "legal_texts": [
            {"id": "LT01", "norm": "art. 143 CdS", "text": "Norm text here"},
        ],
        "key_precedents": [
            {"id": "P01", "citation": "Cass. 16515/2005", "holding": "Holding text", "weight": "high"},
        ],
        "seed_arguments": {
            "by_party": {
                "appellant_001": [
                    {"id": "SA01", "claim": "Signage inadequate", "direction": "attack",
                     "references_facts": ["DF01"]},
                    {"id": "SA02", "claim": "Procedural defect", "direction": "attack",
                     "references_facts": ["F01"]},
                ],
                "respondent_001": [
                    {"id": "SA03", "claim": "Sign was visible", "direction": "defense",
                     "references_facts": ["DF01"]},
                ],
            },
        },
        "stakes": {
            "current_sanction": {"norm": "art. 143 CdS", "fine_range": [170, 680], "points_deducted": 4},
            "alternative_sanction": {"norm": "artt. 6-7 CdS", "fine_range": [42, 173], "points_deducted": 0},
            "litigation_cost_estimate": 1500,
        },
        "timeline": [],
    }


@pytest.fixture
def sample_run_result():
    """Single run result with arguments, responses, and judge decision."""
    return {
        "run_id": "formalista__aggressivo__000",
        "judge_profile": "formalista",
        "appellant_profile": "aggressivo",
        "party_profiles": {"appellant_001": "aggressivo", "respondent_001": "default"},
        "appellant_brief": {
            "filed_brief": {
                "arguments": [
                    {
                        "id": "ARG01",
                        "type": "derived",
                        "derived_from": "SA01",
                        "claim": "Signage was inadequate",
                        "legal_reasoning": "Per art. 143...",
                        "norm_text_cited": ["LT01"],
                        "facts_referenced": ["DF01"],
                        "evidence_cited": ["E02"],
                        "precedents_addressed": [
                            {"id": "P01", "strategy": "distinguish", "reasoning": "Different context"},
                        ],
                        "supports": None,
                    },
                    {
                        "id": "ARG02",
                        "type": "new",
                        "derived_from": None,
                        "claim": "Due process violated",
                        "legal_reasoning": "Notification was late",
                        "norm_text_cited": [],
                        "facts_referenced": ["F01"],
                        "evidence_cited": ["E01"],
                        "precedents_addressed": [],
                        "supports": None,
                    },
                ],
                "requests": {"primary": "Annul", "subordinate": "Reduce"},
            },
            "internal_analysis": {
                "strength_self_assessments": {"ARG01": 0.7, "ARG02": 0.4},
                "key_vulnerabilities": ["Weak evidence"],
                "strongest_point": "ARG01",
                "gaps": [],
            },
        },
        "respondent_brief": {
            "filed_brief": {
                "preliminary_objections": [],
                "responses_to_opponent": [
                    {
                        "to_argument": "ARG01",
                        "counter_strategy": "rebut",
                        "counter_reasoning": "Sign was visible",
                        "norm_text_cited": ["LT01"],
                        "precedents_cited": [],
                    },
                    {
                        "to_argument": "ARG02",
                        "counter_strategy": "distinguish",
                        "counter_reasoning": "Notification was timely",
                        "norm_text_cited": [],
                        "precedents_cited": [],
                    },
                ],
                "affirmative_defenses": [],
                "requests": {"primary": "Reject", "fallback": "Costs"},
            },
            "internal_analysis": {
                "strength_self_assessments": {},
                "key_vulnerabilities": [],
                "opponent_strongest_point": "ARG01",
                "gaps": [],
            },
        },
        "judge_decision": {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {"argument_id": "ARG01", "party": "appellant", "persuasiveness": 0.7,
                 "strengths": "Well-reasoned", "weaknesses": "Weak evidence", "determinative": True},
                {"argument_id": "ARG02", "party": "appellant", "persuasiveness": 0.3,
                 "strengths": "Valid point", "weaknesses": "Late raised", "determinative": False},
            ],
            "precedent_analysis": {
                "P01": {"followed": False, "distinguished": True, "reasoning": "Different facts"},
            },
            "verdict": {
                "qualification_correct": False,
                "qualification_reasoning": "Sign placement was deficient",
                "if_incorrect": {
                    "consequence": "reclassification",
                    "consequence_reasoning": "Apply minor violation norm",
                    "applied_norm": "artt. 6-7 CdS",
                    "sanction_determined": 100,
                    "points_deducted": 0,
                },
                "costs_ruling": "Compensated",
            },
            "reasoning": "Full reasoning here",
            "gaps": [],
        },
        "validation_warnings": {"appellant": [], "respondent": [], "judge": []},
    }


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data before and after each test."""
    from athena.knowledge.config import get_session
    yield
    with get_session() as session:
        session.run("MATCH (n) WHERE n.case_id = 'test-kg-001' DETACH DELETE n")
        session.run("MATCH (n:SeedArgumentNode) WHERE n.seed_arg_id STARTS WITH 'SA0' DETACH DELETE n")
        session.run("MATCH (n:FactNode) WHERE n.fact_id IN ['F01', 'DF01'] DETACH DELETE n")
        session.run("MATCH (n:EvidenceNode) WHERE n.evidence_id IN ['E01', 'E02'] DETACH DELETE n")
        session.run("MATCH (n:LegalTextNode) WHERE n.legal_text_id = 'LT01' DETACH DELETE n")
        session.run("MATCH (n:PrecedentNode) WHERE n.precedent_id = 'P01' DETACH DELETE n")
        session.run("MATCH (n:PartyNode) WHERE n.party_id IN ['appellant_001', 'respondent_001'] DETACH DELETE n")
        session.run("MATCH (n:ArgumentNode) WHERE n.run_id STARTS WITH 'formalista__' DETACH DELETE n")
        session.run("MATCH (n:ResponseNode) WHERE n.run_id STARTS WITH 'formalista__' DETACH DELETE n")
        session.run("MATCH (n:JudgeDecisionNode) WHERE n.run_id STARTS WITH 'formalista__' DETACH DELETE n")
        session.run("MATCH (n:SimRunNode) WHERE n.run_id STARTS WITH 'formalista__' DETACH DELETE n")


class TestCaseIngestion:
    def test_ingest_creates_nodes(self, sample_case_data):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.config import get_session

        counts = ingest_case(sample_case_data)
        assert counts["nodes"] > 0
        assert counts["edges"] > 0

        # Verify case node exists
        with get_session() as session:
            result = session.run(
                "MATCH (c:CaseNode {case_id: $cid}) RETURN c",
                cid="test-kg-001",
            ).single()
            assert result is not None

    def test_ingest_idempotent(self, sample_case_data):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)
        ingest_case(sample_case_data)  # Second call should not duplicate

        with get_session() as session:
            result = session.run(
                "MATCH (c:CaseNode {case_id: $cid}) RETURN count(c) AS n",
                cid="test-kg-001",
            ).single()
            assert result["n"] == 1

    def test_party_nodes(self, sample_case_data):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)

        with get_session() as session:
            result = session.run(
                "MATCH (c:CaseNode {case_id: $cid})-[:HAS_PARTY]->(p:PartyNode) "
                "RETURN count(p) AS n",
                cid="test-kg-001",
            ).single()
            assert result["n"] == 2

    def test_fact_nodes(self, sample_case_data):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)

        with get_session() as session:
            result = session.run(
                "MATCH (c:CaseNode {case_id: $cid})-[:HAS_FACT]->(f:FactNode) "
                "RETURN count(f) AS n",
                cid="test-kg-001",
            ).single()
            assert result["n"] == 2  # 1 undisputed + 1 disputed

    def test_seed_argument_nodes(self, sample_case_data):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)

        with get_session() as session:
            result = session.run(
                "MATCH (c:CaseNode {case_id: $cid})-[:HAS_SEED_ARGUMENT]->(sa:SeedArgumentNode) "
                "RETURN count(sa) AS n",
                cid="test-kg-001",
            ).single()
            assert result["n"] == 3  # SA01, SA02, SA03

    def test_evidence_supports_fact_edge(self, sample_case_data):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)

        with get_session() as session:
            result = session.run(
                "MATCH (e:EvidenceNode {evidence_id: 'E01'})-[:SUPPORTS_FACT]->(f:FactNode {fact_id: 'F01'}) "
                "RETURN count(*) AS n",
            ).single()
            assert result["n"] == 1


class TestResultIngestion:
    def test_store_run_result(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)
        counts = store_run_result("test-kg-001", sample_run_result)
        assert counts["nodes"] > 0

        # Verify SimRunNode
        with get_session() as session:
            result = session.run(
                "MATCH (r:SimRunNode {run_id: $rid}) RETURN r",
                rid="formalista__aggressivo__000",
            ).single()
            assert result is not None

    def test_argument_edges(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)
        store_run_result("test-kg-001", sample_run_result)

        with get_session() as session:
            # Check DERIVES_FROM edge
            result = session.run(
                "MATCH (a:ArgumentNode)-[:DERIVES_FROM]->(sa:SeedArgumentNode {seed_arg_id: 'SA01'}) "
                "RETURN count(*) AS n",
            ).single()
            assert result["n"] >= 1

    def test_evaluates_edge(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)
        store_run_result("test-kg-001", sample_run_result)

        with get_session() as session:
            result = session.run(
                "MATCH (jd:JudgeDecisionNode)-[e:EVALUATES]->(a:ArgumentNode) "
                "WHERE jd.run_id = 'formalista__aggressivo__000' "
                "RETURN count(e) AS n",
            ).single()
            assert result["n"] == 2  # ARG01, ARG02

    def test_response_nodes(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)
        store_run_result("test-kg-001", sample_run_result)

        with get_session() as session:
            result = session.run(
                "MATCH (resp:ResponseNode) "
                "WHERE resp.run_id = 'formalista__aggressivo__000' "
                "RETURN count(resp) AS n",
            ).single()
            assert result["n"] == 2  # Two responses


class TestStatsIngestion:
    def test_store_aggregation(self, sample_case_data):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.stats_loader import store_aggregation
        from athena.knowledge.config import get_session

        ingest_case(sample_case_data)

        aggregated = {
            "argument_effectiveness": {
                "SA01": {"mean_persuasiveness": 0.7, "std_persuasiveness": 0.1, "determinative_rate": 0.5},
            },
            "precedent_analysis": {
                "P01": {"followed_rate": 0.3, "distinguished_rate": 0.7},
            },
            "dominated_strategies": ["passivo"],
            "total_runs": 10,
            "failed_runs": 0,
        }
        counts = store_aggregation("test-kg-001", aggregated)

        with get_session() as session:
            result = session.run(
                "MATCH (sa:SeedArgumentNode {seed_arg_id: 'SA01'}) "
                "RETURN sa.mean_persuasiveness AS mp",
            ).single()
            assert result["mp"] == pytest.approx(0.7)

    def test_store_game_theory(self, sample_case_data):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.stats_loader import store_game_theory
        from athena.knowledge.config import get_session
        from athena.game_theory.schemas import (
            GameTheoryAnalysis, BATNA, SettlementRange, SensitivityResult,
            PartyValuations, OutcomeValuation,
        )

        ingest_case(sample_case_data)

        game_analysis = GameTheoryAnalysis(
            party_valuations={},
            batna={
                "appellant": BATNA(
                    party_id="appellant", expected_value=-500.0,
                    expected_value_range=(-800.0, -200.0), best_strategy="aggressivo",
                    outcome_probabilities={"rejection": 0.5, "annulment": 0.3},
                ),
            },
            settlement=SettlementRange(
                zopa=(-400.0, -200.0), nash_solution=-300.0, surplus=200.0,
                settlement_exists=True,
            ),
            sensitivity=[
                SensitivityResult(
                    parameter="litigation_cost", base_value=1500.0,
                    sweep_values=[0, 500, 1000, 1500, 2000],
                    ev_at_each=[-200, -350, -425, -500, -575],
                    threshold=800.0, impact=375.0,
                ),
            ],
            expected_value_by_strategy={"aggressivo": -500.0},
            recommended_strategy="aggressivo",
            analysis_metadata={},
        )

        counts = store_game_theory("test-kg-001", game_analysis)
        assert counts["nodes"] >= 2  # BATNA + Settlement + Sensitivity

        with get_session() as session:
            result = session.run(
                "MATCH (c:CaseNode {case_id: 'test-kg-001'})-[:HAS_BATNA]->(b:BATNANode) "
                "RETURN b.expected_value AS ev",
            ).single()
            assert result["ev"] == pytest.approx(-500.0)
