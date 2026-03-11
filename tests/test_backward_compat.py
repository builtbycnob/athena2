# tests/test_backward_compat.py
"""Tests for backward compatibility: old-format YAML → migrate → validate → run."""

import pytest
from unittest.mock import patch

from athena.cli import migrate_case_v1
from athena.schemas.case import CaseFile
from athena.schemas.simulation import SimulationConfig, migrate_simulation_v1


def _make_old_case():
    """Old-format case data with appellant/respondent keys."""
    return {
        "case_id": "gdp-milano-17928-2025",
        "jurisdiction": {
            "country": "IT",
            "court": "giudice_di_pace",
            "venue": "Milano",
            "applicable_law": ["D.Lgs. 285/1992"],
            "key_precedents": [
                {"id": "cass_16515_2005", "citation": "Cass. civ. n. 16515/2005",
                 "holding": "Test", "weight": "contested"},
            ],
            "procedural_rules": {
                "rite": "opposizione",
                "phases": ["ricorso", "decisione"],
                "allowed_moves": {"appellant": ["memoria"], "respondent": ["memoria"]},
            },
        },
        "parties": [
            {"id": "opponente", "role": "appellant", "type": "persona_fisica",
             "objectives": {"primary": "annullamento", "subordinate": "riqualificazione"}},
            {"id": "comune_milano", "role": "respondent", "type": "pa",
             "objectives": {"primary": "conferma", "subordinate": "conferma_ridotta"}},
        ],
        "stakes": {
            "current_sanction": {"norm": "art. 143 CdS", "fine_range": [170, 680], "points_deducted": 4},
            "alternative_sanction": {"norm": "artt. 6-7 CdS", "fine_range": [42, 173], "points_deducted": 0},
            "litigation_cost_estimate": 1500,
        },
        "evidence": [
            {"id": "DOC1", "type": "atto", "description": "Verbale",
             "produced_by": "comune_milano", "admissibility": "uncontested", "supports_facts": ["F1"]},
        ],
        "facts": {
            "undisputed": [{"id": "F1", "description": "Transito", "evidence": ["DOC1"]}],
            "disputed": [
                {
                    "id": "D1",
                    "description": "Qualificazione",
                    "appellant_position": "Art. 143 inapplicabile",
                    "respondent_position": "Art. 143 applicabile",
                    "depends_on_facts": ["F1"],
                }
            ],
        },
        "legal_texts": [{"id": "art_143_cds", "norm": "Art. 143", "text": "Testo"}],
        "seed_arguments": {
            "appellant": [{"id": "SA1", "claim": "Errata qualif.", "direction": "143 no", "references_facts": ["F1"]}],
            "respondent": [{"id": "SR1", "claim": "Legittimità", "direction": "Cass.", "references_facts": ["F1"]}],
        },
        "key_precedents": [
            {"id": "cass_16515_2005", "citation": "Cass. civ. n. 16515/2005",
             "holding": "Test", "weight": "contested"},
        ],
        "timeline": [],
    }


def _make_old_sim():
    """Old-format simulation config."""
    return {
        "case_ref": "gdp-milano-17928-2025",
        "language": "it",
        "judge_profiles": [
            {"id": "formalista", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
        ],
        "appellant_profiles": [
            {"id": "aggressivo", "style": "Attacca frontalmente."},
        ],
        "temperature": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
        "runs_per_combination": 3,
    }


class TestBackwardCompatCase:
    def test_old_case_migrates_and_loads(self):
        old = _make_old_case()
        migrated = migrate_case_v1(old)
        case = CaseFile(**migrated)
        assert case.case_id == "gdp-milano-17928-2025"
        assert "opponente" in case.seed_arguments.by_party
        assert "comune_milano" in case.seed_arguments.by_party
        assert "opponente" in case.facts.disputed[0].positions

    def test_old_case_extract_all_ids(self):
        old = _make_old_case()
        migrated = migrate_case_v1(old)
        case = CaseFile(**migrated)
        ids = case.extract_all_ids()
        assert "F1" in ids
        assert "DOC1" in ids
        assert "SA1" in ids
        assert "SR1" in ids


class TestBackwardCompatSim:
    def test_old_sim_migrates_and_loads(self):
        old = _make_old_sim()
        migrated = migrate_simulation_v1(old)
        config = SimulationConfig(**migrated)
        assert config.total_runs == 3
        assert "opponente" in config.party_profiles

    def test_old_sim_total_runs_matches(self):
        old = _make_old_sim()
        old["judge_profiles"].append(
            {"id": "sost", "jurisprudential_orientation": "distinguishes_cassazione", "formalism": "low"},
        )
        old["appellant_profiles"].append({"id": "prudente", "style": "Prudente."})
        migrated = migrate_simulation_v1(old)
        config = SimulationConfig(**migrated)
        # 2 judges × 2 styles × 3 runs = 12
        assert config.total_runs == 12


class TestBackwardCompatPipeline:
    @patch("athena.simulation.graph.invoke_llm")
    def test_full_pipeline_with_old_format(self, mock_llm):
        """Old-format data goes through migration and produces a valid run."""
        from athena.simulation.graph import run_single

        # Briefs that match the old case data's IDs (SA1, SR1, F1, DOC1, art_143_cds, cass_16515_2005)
        appellant_brief = {
            "filed_brief": {
                "arguments": [
                    {
                        "id": "ARG1", "type": "derived", "derived_from": "SA1",
                        "claim": "Errata qualif.", "legal_reasoning": "Art. 143 non copre.",
                        "norm_text_cited": ["art_143_cds"], "facts_referenced": ["F1"],
                        "evidence_cited": ["DOC1"],
                        "precedents_addressed": [{"id": "cass_16515_2005", "strategy": "distinguish", "reasoning": "Non in punto."}],
                        "supports": None,
                    }
                ],
                "requests": {"primary": "Annullamento", "subordinate": "Riqualificazione"},
            },
            "internal_analysis": {
                "strength_self_assessments": {"ARG1": 0.7},
                "key_vulnerabilities": ["Cass. contraria"],
                "strongest_point": "Testo letterale",
                "gaps": [],
            },
        }
        respondent_brief = {
            "filed_brief": {
                "preliminary_objections": [],
                "responses_to_opponent": [
                    {"to_argument": "ARG1", "counter_strategy": "rebut",
                     "counter_reasoning": "Cass. equipara.", "norm_text_cited": ["art_143_cds"],
                     "precedents_cited": [{"id": "cass_16515_2005", "relevance": "In punto."}]},
                ],
                "affirmative_defenses": [
                    {"id": "RARG1", "type": "derived", "derived_from": "SR1",
                     "claim": "Legittimità", "legal_reasoning": "Cass. equipara.",
                     "norm_text_cited": ["art_143_cds"], "facts_referenced": ["F1"],
                     "evidence_cited": ["DOC1"]},
                ],
                "requests": {"primary": "Rigetto", "fallback": "Conferma"},
            },
            "internal_analysis": {
                "strength_self_assessments": {"RARG1": 0.6},
                "key_vulnerabilities": ["Testo letterale"],
                "opponent_strongest_point": "Art. 143",
                "gaps": [],
            },
        }
        judge_decision = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {"argument_id": "ARG1", "party": "appellant", "persuasiveness": 0.7,
                 "strengths": "Forte", "weaknesses": "Debole", "determinative": True},
                {"argument_id": "RARG1", "party": "respondent", "persuasiveness": 0.5,
                 "strengths": "Ok", "weaknesses": "Debole", "determinative": False},
            ],
            "precedent_analysis": {"cass_16515_2005": {"followed": False, "distinguished": True, "reasoning": "Test."}},
            "verdict": {
                "qualification_correct": False,
                "qualification_reasoning": "Errata qualificazione.",
                "if_incorrect": {"consequence": "reclassification", "consequence_reasoning": "Riqualifica.",
                                 "applied_norm": "artt. 6-7 CdS", "sanction_determined": 87, "points_deducted": 0},
                "costs_ruling": "a carico del Comune",
            },
            "reasoning": "Motivazione.",
            "gaps": [],
        }
        mock_llm.side_effect = [appellant_brief, respondent_brief, judge_decision]

        old_case = _make_old_case()
        migrated_case = migrate_case_v1(old_case)

        run_params = {
            "run_id": "test__agg__000",
            "judge_profile": {"id": "formalista", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
            "appellant_profile": {"id": "aggressivo", "style": "Attacca frontalmente."},
            "party_profiles": {},
            "temperature": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            "temperatures": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            "language": "it",
        }

        result = run_single(migrated_case, run_params)
        assert result["error"] is None
        assert result["judge_decision"]["verdict"]["qualification_correct"] is False
