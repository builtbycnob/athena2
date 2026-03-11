# tests/test_knowledge_ontology.py
"""Tests for knowledge graph ontology models — serialization and validation."""

import pytest
from athena.knowledge.ontology import (
    CaseNode,
    PartyNode,
    FactNode,
    EvidenceNode,
    LegalTextNode,
    PrecedentNode,
    SeedArgumentNode,
    ArgumentNode,
    ResponseNode,
    JudgeDecisionNode,
    SimRunNode,
    BATNANode,
    SettlementNode,
    SensitivityNode,
)


class TestCaseNode:
    def test_from_case_data(self):
        case_data = {
            "case_id": "gdp-milano-17928-2025",
            "title": "Traffic Violation Appeal",
            "jurisdiction": {
                "country": "Italy",
                "court": "Giudice di Pace",
                "venue": "Milano",
            },
        }
        node = CaseNode.from_case_data(case_data)
        assert node.case_id == "gdp-milano-17928-2025"
        assert node.jurisdiction_country == "Italy"
        assert node.jurisdiction_court == "Giudice di Pace"

    def test_from_case_data_minimal(self):
        node = CaseNode.from_case_data({"case_id": "test"})
        assert node.case_id == "test"
        assert node.title == "test"
        assert node.jurisdiction_country == ""

    def test_roundtrip(self):
        node = CaseNode(case_id="x", title="T", jurisdiction_country="IT")
        d = node.model_dump()
        assert CaseNode(**d) == node


class TestPartyNode:
    def test_from_party(self):
        party = {
            "id": "ricorrente_rossi",
            "role": "appellant",
            "type": "natural_person",
            "objectives": {"primary": "annul", "subordinate": "reduce"},
        }
        node = PartyNode.from_party(party)
        assert node.party_id == "ricorrente_rossi"
        assert node.role == "appellant"
        assert node.primary_objective == "annul"

    def test_roundtrip(self):
        node = PartyNode(party_id="p1", role="r", type="t")
        assert PartyNode(**node.model_dump()) == node


class TestFactNode:
    def test_undisputed(self):
        node = FactNode(fact_id="F01", description="Car stopped", is_disputed=False)
        assert not node.is_disputed
        assert node.positions is None

    def test_disputed(self):
        node = FactNode(
            fact_id="DF01",
            description="Speed limit applies",
            is_disputed=True,
            positions={"appellant": "No", "respondent": "Yes"},
        )
        assert node.is_disputed
        assert "appellant" in node.positions


class TestEvidenceNode:
    def test_basic(self):
        node = EvidenceNode(
            evidence_id="E01", type="document", description="PV",
            produced_by="respondent", admissibility="uncontested",
        )
        d = node.model_dump()
        assert d["evidence_id"] == "E01"


class TestLegalTextNode:
    def test_basic(self):
        node = LegalTextNode(legal_text_id="LT01", norm="art. 143 CdS", text="...")
        assert node.norm == "art. 143 CdS"


class TestPrecedentNode:
    def test_with_stats(self):
        node = PrecedentNode(
            precedent_id="P01", citation="Cass.", holding="...",
            followed_rate=0.8, distinguished_rate=0.2,
        )
        assert node.followed_rate == 0.8

    def test_without_stats(self):
        node = PrecedentNode(precedent_id="P01", citation="C", holding="H")
        assert node.followed_rate is None


class TestSeedArgumentNode:
    def test_with_stats(self):
        node = SeedArgumentNode(
            seed_arg_id="SA01", claim="claim", direction="attack",
            party_id="appellant", mean_persuasiveness=0.7,
        )
        assert node.mean_persuasiveness == 0.7

    def test_references(self):
        node = SeedArgumentNode(
            seed_arg_id="SA01", claim="c", direction="d",
            party_id="p", references_facts=["F01", "F02"],
        )
        assert len(node.references_facts) == 2


class TestArgumentNode:
    def test_derived(self):
        node = ArgumentNode(
            argument_id="run1__ARG01", type="derived",
            derived_from="SA01", claim="claim", run_id="run1",
        )
        assert node.derived_from == "SA01"

    def test_new(self):
        node = ArgumentNode(
            argument_id="run1__ARG02", type="new",
            claim="new claim",
        )
        assert node.derived_from is None


class TestResponseNode:
    def test_basic(self):
        node = ResponseNode(
            response_id="run1__resp__ARG01",
            to_argument="ARG01",
            counter_strategy="rebut",
            run_id="run1",
        )
        assert node.counter_strategy == "rebut"


class TestJudgeDecisionNode:
    def test_rejection(self):
        node = JudgeDecisionNode(
            run_id="run1", qualification_correct=True,
        )
        assert node.consequence is None

    def test_annulment(self):
        node = JudgeDecisionNode(
            run_id="run1", qualification_correct=False,
            consequence="annulment",
        )
        assert node.consequence == "annulment"


class TestSimRunNode:
    def test_basic(self):
        node = SimRunNode(
            run_id="jp1__ap1__000",
            judge_profile_id="formalista",
            party_profile_ids={"appellant": "aggressivo"},
        )
        assert node.judge_profile_id == "formalista"


class TestBATNANode:
    def test_basic(self):
        node = BATNANode(
            key="case1__appellant", case_id="case1",
            party_id="appellant", expected_value=-500.0,
            expected_value_range_low=-800.0, expected_value_range_high=-200.0,
        )
        assert node.expected_value == -500.0


class TestSettlementNode:
    def test_with_zopa(self):
        node = SettlementNode(
            key="case1", case_id="case1", settlement_exists=True,
            zopa_low=-400.0, zopa_high=-200.0, nash_solution=-300.0,
        )
        assert node.nash_solution == -300.0

    def test_no_zopa(self):
        node = SettlementNode(
            key="case1", case_id="case1", settlement_exists=False,
        )
        assert node.zopa_low is None


class TestSensitivityNode:
    def test_basic(self):
        node = SensitivityNode(
            key="case1__litigation_cost", case_id="case1",
            parameter="litigation_cost", impact=500.0,
        )
        assert node.impact == 500.0
