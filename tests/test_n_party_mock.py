# tests/test_n_party_mock.py
"""Tests for N-party architecture with mock LLM."""

import pytest
from unittest.mock import patch

from athena.schemas.case import CaseFile
from athena.schemas.simulation import SimulationConfig
from athena.simulation.context import build_party_context, build_adjudicator_context
from athena.simulation.validation import validate_agent_output
from athena.simulation.graph import (
    AgentConfig, Phase, build_graph_from_phases, build_bilateral_phases,
)
from athena.simulation.orchestrator import _generate_combinations


def _make_3party_case():
    """Case with 3 parties: plaintiff, defendant, co-defendant."""
    return {
        "case_id": "test-3party",
        "jurisdiction": {
            "country": "IT",
            "court": "tribunale",
            "venue": "Milano",
            "applicable_law": ["C.C."],
            "key_precedents": [],
            "procedural_rules": {
                "rite": "ordinario",
                "phases": ["citazione", "comparsa", "decisione"],
                "allowed_moves": {"plaintiff": ["atto_citazione"], "defendant": ["comparsa"]},
            },
        },
        "parties": [
            {"id": "plaintiff", "role": "appellant", "type": "persona_fisica",
             "objectives": {"primary": "risarcimento", "subordinate": "risarcimento_parziale"}},
            {"id": "defendant", "role": "respondent", "type": "società",
             "objectives": {"primary": "rigetto", "subordinate": "riduzione"}},
            {"id": "co_defendant", "role": "co_respondent", "type": "assicurazione",
             "objectives": {"primary": "manleva", "subordinate": "limitazione"},
             "visibility": {"evidence_visibility": "all", "brief_visibility": []}},
        ],
        "stakes": {
            "current_sanction": {"norm": "art. 2043 CC", "fine_range": [5000, 50000], "points_deducted": 0},
            "alternative_sanction": {"norm": "art. 1218 CC", "fine_range": [3000, 30000], "points_deducted": 0},
            "litigation_cost_estimate": 10000,
        },
        "evidence": [
            {"id": "DOC1", "type": "referto", "description": "Referto medico",
             "produced_by": "plaintiff", "admissibility": "uncontested", "supports_facts": ["F1"]},
            {"id": "DOC2", "type": "polizza", "description": "Polizza assicurativa",
             "produced_by": "co_defendant", "admissibility": "uncontested", "supports_facts": ["F2"]},
            {"id": "DOC3", "type": "perizia", "description": "Perizia tecnica",
             "produced_by": "defendant", "admissibility": "contested", "supports_facts": ["F1"]},
        ],
        "facts": {
            "undisputed": [
                {"id": "F1", "description": "Incidente avvenuto", "evidence": ["DOC1"]},
            ],
            "disputed": [
                {
                    "id": "D1",
                    "description": "Responsabilità",
                    "positions": {
                        "plaintiff": "Responsabilità del convenuto",
                        "defendant": "Concorso di colpa",
                        "co_defendant": "Nessuna copertura assicurativa",
                    },
                    "depends_on_facts": ["F1"],
                }
            ],
        },
        "legal_texts": [{"id": "art_2043_cc", "norm": "Art. 2043 C.C.", "text": "Risarcimento..."}],
        "seed_arguments": {
            "by_party": {
                "plaintiff": [{"id": "SA_P1", "claim": "Danno", "direction": "Risarcimento pieno", "references_facts": ["F1"]}],
                "defendant": [{"id": "SA_D1", "claim": "Concorso", "direction": "Riduzione", "references_facts": ["F1"]}],
                "co_defendant": [{"id": "SA_C1", "claim": "Manleva", "direction": "Non coperto", "references_facts": ["F2"]}],
            },
        },
        "key_precedents": [],
        "timeline": [],
    }


class TestNPartyCaseFile:
    def test_3party_case_loads(self):
        case = CaseFile(**_make_3party_case())
        assert len(case.parties) == 3
        assert len(case.seed_arguments.by_party) == 3

    def test_3party_extract_all_ids(self):
        case = CaseFile(**_make_3party_case())
        ids = case.extract_all_ids()
        assert "SA_P1" in ids
        assert "SA_D1" in ids
        assert "SA_C1" in ids
        assert "DOC1" in ids
        assert "DOC2" in ids
        assert "DOC3" in ids

    def test_3party_disputed_positions(self):
        case = CaseFile(**_make_3party_case())
        d1 = case.facts.disputed[0]
        assert len(d1.positions) == 3
        assert "plaintiff" in d1.positions
        assert "defendant" in d1.positions
        assert "co_defendant" in d1.positions


class TestNPartyContext:
    def test_party_context_filters_evidence(self):
        case_data = _make_3party_case()
        run_params = {"party_profiles": {}, "judge_profile": {}}
        # Plaintiff sees own + uncontested evidence
        ctx = build_party_context(case_data, run_params, "plaintiff")
        ev_ids = {e["id"] for e in ctx["evidence"]}
        assert "DOC1" in ev_ids  # own
        assert "DOC2" in ev_ids  # uncontested
        assert "DOC3" not in ev_ids  # defendant's contested

    def test_party_context_with_visibility_all(self):
        case_data = _make_3party_case()
        run_params = {"party_profiles": {}, "judge_profile": {}}
        # co_defendant has visibility: all
        ctx = build_party_context(
            case_data, run_params, "co_defendant",
            visibility={"evidence_visibility": "all"},
        )
        ev_ids = {e["id"] for e in ctx["evidence"]}
        assert "DOC1" in ev_ids
        assert "DOC2" in ev_ids
        assert "DOC3" in ev_ids

    def test_party_context_includes_own_seeds(self):
        case_data = _make_3party_case()
        run_params = {"party_profiles": {}, "judge_profile": {}}
        ctx = build_party_context(case_data, run_params, "defendant")
        assert len(ctx["seed_arguments"]) == 1
        assert ctx["seed_arguments"][0]["id"] == "SA_D1"

    def test_adjudicator_context_sees_all(self):
        case_data = _make_3party_case()
        run_params = {"judge_profile": {"id": "j1", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"}}
        # Mock briefs from all parties
        briefs = {
            "plaintiff": {"filed_brief": {"arguments": [{"id": "ARG1"}]}, "internal_analysis": {}},
            "defendant": {"filed_brief": {"arguments": [{"id": "DARG1"}]}, "internal_analysis": {}},
            "co_defendant": {"filed_brief": {"arguments": [{"id": "CARG1"}]}, "internal_analysis": {}},
        }
        ctx = build_adjudicator_context(case_data, run_params, briefs)
        # Should see all 3 sanitized briefs (no internal_analysis)
        assert len(ctx["all_briefs"]) == 3
        for pid, brief in ctx["all_briefs"].items():
            assert "arguments" in brief
            assert "internal_analysis" not in brief


class TestNPartyValidation:
    def test_validate_with_prior_briefs(self):
        """Respondent must address arguments from all prior parties."""
        case = CaseFile(**_make_3party_case())
        plaintiff_brief = {
            "filed_brief": {
                "arguments": [
                    {"id": "ARG1", "type": "new", "derived_from": None, "claim": "c",
                     "legal_reasoning": "r", "norm_text_cited": ["art_2043_cc"],
                     "facts_referenced": ["F1"], "evidence_cited": ["DOC1"],
                     "precedents_addressed": [], "supports": None}
                ],
            },
            "internal_analysis": {"strength_self_assessments": {}, "key_vulnerabilities": [],
                                  "strongest_point": "x", "gaps": []},
        }
        # Respondent doesn't address ARG1
        respondent_output = {
            "filed_brief": {
                "preliminary_objections": [],
                "responses_to_opponent": [],  # Missing!
                "affirmative_defenses": [],
                "requests": {"primary": "p", "fallback": "f"},
            },
            "internal_analysis": {"strength_self_assessments": {}, "key_vulnerabilities": ["x"],
                                  "opponent_strongest_point": "x", "gaps": []},
        }
        result = validate_agent_output(
            respondent_output, "respondent", case,
            prior_briefs={"plaintiff": plaintiff_brief},
        )
        assert result.valid is False
        assert any("ARG1" in e for e in result.errors)


class TestNPartyCombinations:
    def test_3party_combination_count(self):
        sim_config = {
            "judge_profiles": [
                {"id": "j1", "party_id": "judge", "role_type": "adjudicator", "parameters": {}},
                {"id": "j2", "party_id": "judge", "role_type": "adjudicator", "parameters": {}},
            ],
            "party_profiles": {
                "plaintiff": [
                    {"id": "p1", "party_id": "plaintiff", "role_type": "advocate", "parameters": {}},
                    {"id": "p2", "party_id": "plaintiff", "role_type": "advocate", "parameters": {}},
                ],
                "defendant": [
                    {"id": "d1", "party_id": "defendant", "role_type": "advocate", "parameters": {}},
                    {"id": "d2", "party_id": "defendant", "role_type": "advocate", "parameters": {}},
                    {"id": "d3", "party_id": "defendant", "role_type": "advocate", "parameters": {}},
                ],
            },
            "temperatures": {"plaintiff": 0.5, "defendant": 0.4, "judge": 0.3},
            "runs_per_combination": 2,
            "language": "it",
        }
        combos = _generate_combinations(sim_config)
        # 2 judges × 2 plaintiffs × 3 defendants × 2 runs = 24
        assert len(combos) == 24

    def test_combination_run_ids_unique(self):
        sim_config = {
            "judge_profiles": [
                {"id": "j1", "party_id": "judge", "role_type": "adjudicator", "parameters": {}},
            ],
            "party_profiles": {
                "a": [
                    {"id": "a1", "party_id": "a", "role_type": "advocate", "parameters": {}},
                    {"id": "a2", "party_id": "a", "role_type": "advocate", "parameters": {}},
                ],
            },
            "temperatures": {},
            "runs_per_combination": 3,
            "language": "it",
        }
        combos = _generate_combinations(sim_config)
        run_ids = [c["run_id"] for c in combos]
        assert len(run_ids) == len(set(run_ids))  # all unique


class TestBilateralPhases:
    def test_build_bilateral_phases_structure(self, sample_case_data, sample_run_params):
        phases = build_bilateral_phases(sample_case_data, sample_run_params)
        assert len(phases) == 3
        assert phases[0].name == "filing"
        assert phases[1].name == "response"
        assert phases[2].name == "decision"
        assert phases[0].agents[0].role_type == "advocate"
        assert phases[2].agents[0].role_type == "adjudicator"

    def test_build_bilateral_phases_party_ids(self, sample_case_data, sample_run_params):
        phases = build_bilateral_phases(sample_case_data, sample_run_params)
        assert phases[0].agents[0].party_id == "opponente"
        assert phases[1].agents[0].party_id == "comune_milano"
        assert phases[2].agents[0].party_id == "judge"
