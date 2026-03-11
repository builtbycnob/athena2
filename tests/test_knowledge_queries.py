# tests/test_knowledge_queries.py
"""Tests for knowledge graph queries — context enrichment and post-analysis.

Requires a running Neo4j instance with ingested test data.
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ATHENA_KG_ENABLED") != "1",
    reason="Knowledge graph not enabled (set ATHENA_KG_ENABLED=1 and NEO4J_PASSWORD)",
)


@pytest.fixture
def sample_case_data():
    """Minimal case data."""
    return {
        "case_id": "test-kg-query-001",
        "title": "KG Query Test",
        "jurisdiction": {
            "country": "Italy", "court": "GdP", "venue": "Milano",
            "applicable_law": [], "key_precedents": [],
            "procedural_rules": {"rite": "opposizione", "phases": [], "allowed_moves": {}},
        },
        "parties": [
            {"id": "app1", "role": "appellant", "type": "natural_person",
             "objectives": {"primary": "annul", "subordinate": "reduce"}},
            {"id": "resp1", "role": "respondent", "type": "public_entity",
             "objectives": {"primary": "confirm", "subordinate": "costs"}},
        ],
        "facts": {
            "undisputed": [{"id": "QF01", "description": "Fact 1", "evidence": []}],
            "disputed": [],
        },
        "evidence": [],
        "legal_texts": [{"id": "QLT01", "norm": "art. 1", "text": "..."}],
        "key_precedents": [{"id": "QP01", "citation": "Cass. 1/2020", "holding": "H", "weight": "high"}],
        "seed_arguments": {
            "by_party": {
                "app1": [
                    {"id": "QSA01", "claim": "Claim 1", "direction": "attack", "references_facts": ["QF01"]},
                ],
            },
        },
        "stakes": {
            "current_sanction": {"norm": "art. 1", "fine_range": [100, 200], "points_deducted": 2},
            "alternative_sanction": {"norm": "art. 2", "fine_range": [50, 100], "points_deducted": 0},
            "litigation_cost_estimate": 500,
        },
        "timeline": [],
    }


@pytest.fixture
def sample_run_result():
    """Run result for query tests."""
    return {
        "run_id": "qformal__qaggr__000",
        "judge_profile": "qformal",
        "appellant_profile": "qaggr",
        "party_profiles": {"app1": "qaggr"},
        "appellant_brief": {
            "filed_brief": {
                "arguments": [
                    {
                        "id": "QARG01", "type": "derived", "derived_from": "QSA01",
                        "claim": "Query test arg", "legal_reasoning": "...",
                        "norm_text_cited": ["QLT01"], "facts_referenced": ["QF01"],
                        "evidence_cited": [],
                        "precedents_addressed": [
                            {"id": "QP01", "strategy": "distinguish", "reasoning": "..."},
                        ],
                        "supports": None,
                    },
                ],
                "requests": {"primary": "Annul", "subordinate": "Reduce"},
            },
            "internal_analysis": {
                "strength_self_assessments": {"QARG01": 0.8},
                "key_vulnerabilities": [], "strongest_point": "QARG01", "gaps": [],
            },
        },
        "respondent_brief": {
            "filed_brief": {
                "preliminary_objections": [],
                "responses_to_opponent": [
                    {
                        "to_argument": "QARG01", "counter_strategy": "rebut",
                        "counter_reasoning": "...", "norm_text_cited": [], "precedents_cited": [],
                    },
                ],
                "affirmative_defenses": [],
                "requests": {"primary": "Reject", "fallback": "Costs"},
            },
            "internal_analysis": {
                "strength_self_assessments": {}, "key_vulnerabilities": [],
                "opponent_strongest_point": "QARG01", "gaps": [],
            },
        },
        "judge_decision": {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {"argument_id": "QARG01", "party": "appellant", "persuasiveness": 0.8,
                 "strengths": "Strong", "weaknesses": "None", "determinative": True},
            ],
            "precedent_analysis": {
                "QP01": {"followed": True, "distinguished": False, "reasoning": "Applicable"},
            },
            "verdict": {
                "qualification_correct": False,
                "qualification_reasoning": "...",
                "if_incorrect": {
                    "consequence": "annulment", "consequence_reasoning": "...",
                    "applied_norm": "art. 2", "sanction_determined": 75, "points_deducted": 0,
                },
                "costs_ruling": "Compensated",
            },
            "reasoning": "...",
            "gaps": [],
        },
        "validation_warnings": {"appellant": [], "respondent": [], "judge": []},
    }


@pytest.fixture(autouse=True)
def cleanup_query_data():
    """Clean up test data."""
    from athena.knowledge.config import get_session
    yield
    with get_session() as session:
        session.run("MATCH (n) WHERE n.case_id = 'test-kg-query-001' DETACH DELETE n")
        session.run("MATCH (n:SeedArgumentNode) WHERE n.seed_arg_id = 'QSA01' DETACH DELETE n")
        session.run("MATCH (n:FactNode) WHERE n.fact_id = 'QF01' DETACH DELETE n")
        session.run("MATCH (n:LegalTextNode) WHERE n.legal_text_id = 'QLT01' DETACH DELETE n")
        session.run("MATCH (n:PrecedentNode) WHERE n.precedent_id = 'QP01' DETACH DELETE n")
        session.run("MATCH (n:PartyNode) WHERE n.party_id IN ['app1', 'resp1'] DETACH DELETE n")
        session.run("MATCH (n:ArgumentNode) WHERE n.run_id STARTS WITH 'qformal__' DETACH DELETE n")
        session.run("MATCH (n:ResponseNode) WHERE n.run_id STARTS WITH 'qformal__' DETACH DELETE n")
        session.run("MATCH (n:JudgeDecisionNode) WHERE n.run_id STARTS WITH 'qformal__' DETACH DELETE n")
        session.run("MATCH (n:SimRunNode) WHERE n.run_id STARTS WITH 'qformal__' DETACH DELETE n")


class TestContextEnrichment:
    def test_seed_argument_ranking(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.queries.context_enrichment import query_seed_argument_ranking

        ingest_case(sample_case_data)
        store_run_result("test-kg-query-001", sample_run_result)

        ranking = query_seed_argument_ranking("test-kg-query-001", "qformal")
        assert len(ranking) >= 1
        assert ranking[0]["seed_arg_id"] == "QSA01"
        assert ranking[0]["avg_persuasiveness"] == pytest.approx(0.8)

    def test_best_precedent_strategy(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.queries.context_enrichment import query_best_precedent_strategy

        ingest_case(sample_case_data)
        store_run_result("test-kg-query-001", sample_run_result)

        strategies = query_best_precedent_strategy("test-kg-query-001", "qformal")
        assert len(strategies) >= 1
        assert strategies[0]["precedent_id"] == "QP01"

    def test_expected_counters(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.queries.context_enrichment import query_expected_counters

        ingest_case(sample_case_data)
        store_run_result("test-kg-query-001", sample_run_result)

        counters = query_expected_counters("test-kg-query-001", "QSA01")
        assert len(counters) >= 1
        assert counters[0]["counter_strategy"] == "rebut"

    def test_get_enrichment_for_run(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.queries.context_enrichment import get_enrichment_for_run

        ingest_case(sample_case_data)
        store_run_result("test-kg-query-001", sample_run_result)

        enrichment = get_enrichment_for_run("test-kg-query-001", "qformal")
        assert "seed_arg_ranking" in enrichment
        assert "precedent_strategies" in enrichment
        assert "counter_strategies" in enrichment


class TestPostAnalysis:
    def test_argument_trajectory(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.queries.post_analysis import query_argument_trajectory

        ingest_case(sample_case_data)
        store_run_result("test-kg-query-001", sample_run_result)

        trajectories = query_argument_trajectory("test-kg-query-001")
        assert len(trajectories) >= 1

    def test_determinative_arguments(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.queries.post_analysis import query_determinative_arguments

        ingest_case(sample_case_data)
        store_run_result("test-kg-query-001", sample_run_result)

        det_args = query_determinative_arguments("test-kg-query-001")
        assert len(det_args) >= 1
        assert det_args[0]["times_determinative"] >= 1

    def test_get_post_analysis(self, sample_case_data, sample_run_result):
        from athena.knowledge.ingestion.case_loader import ingest_case
        from athena.knowledge.ingestion.result_loader import store_run_result
        from athena.knowledge.queries.post_analysis import get_post_analysis

        ingest_case(sample_case_data)
        store_run_result("test-kg-query-001", sample_run_result)

        analysis = get_post_analysis("test-kg-query-001")
        assert "argument_trajectories" in analysis
        assert "determinative_arguments" in analysis
        assert "precedent_follow_rates" in analysis
