# tests/test_generic_graph.py
"""Tests for the generic graph path (build_graph_from_phases).

Replaces test_graph.py legacy tests. Exercises the full path:
build_bilateral_phases → build_graph_from_phases → invoke with GraphState.
"""

import pytest
from unittest.mock import patch

from athena.simulation.graph import (
    build_graph_from_phases,
    build_bilateral_phases,
    GraphState,
    _MAX_TOKENS,
)
from athena.schemas.structured_output import AGENT_SCHEMAS


class TestGenericGraphEndToEnd:
    """Integration test: build_graph_from_phases invoked with mock LLM."""

    @patch("athena.simulation.graph.invoke_llm")
    def test_bilateral_produces_briefs_and_decision(
        self,
        mock_llm,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        """Full bilateral run through generic graph produces correct state."""
        judge_decision = {
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
        }
        mock_llm.side_effect = [
            sample_appellant_brief,
            sample_respondent_brief,
            judge_decision,
        ]

        phases = build_bilateral_phases(sample_case_data, sample_run_params)
        graph = build_graph_from_phases(phases)

        initial_state = {
            "case": sample_case_data,
            "params": sample_run_params,
            "briefs": {},
            "validations": {},
            "decision": None,
            "decision_validation": None,
            "error": None,
        }

        result = graph.invoke(initial_state)

        # Briefs stored by party_id
        assert "opponente" in result["briefs"]
        assert "comune_milano" in result["briefs"]
        assert result["briefs"]["opponente"] is not None
        assert result["briefs"]["comune_milano"] is not None

        # Decision stored
        assert result["decision"] is not None
        assert result["decision"]["verdict"]["qualification_correct"] is False

        # Validations stored
        assert "opponente" in result["validations"]
        assert "comune_milano" in result["validations"]

        # Decision validation stored
        assert result["decision_validation"] is not None

        # No errors
        assert result.get("error") is None

    @patch("athena.simulation.graph.invoke_llm")
    def test_passes_correct_schemas_and_max_tokens(
        self,
        mock_llm,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        """Each agent gets correct json_schema and max_tokens."""
        judge_decision = {
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
                "qualification_reasoning": "Errata.",
                "if_incorrect": {
                    "consequence": "reclassification",
                    "consequence_reasoning": "Riqualificata.",
                    "applied_norm": "artt. 6-7 CdS",
                    "sanction_determined": 87,
                    "points_deducted": 0,
                },
                "costs_ruling": "a carico del Comune",
            },
            "reasoning": "Motivazione.",
            "gaps": [],
        }
        mock_llm.side_effect = [
            sample_appellant_brief,
            sample_respondent_brief,
            judge_decision,
        ]

        phases = build_bilateral_phases(sample_case_data, sample_run_params)
        graph = build_graph_from_phases(phases)
        initial_state = {
            "case": sample_case_data,
            "params": sample_run_params,
            "briefs": {},
            "validations": {},
            "decision": None,
            "decision_validation": None,
            "error": None,
        }
        graph.invoke(initial_state)

        assert mock_llm.call_count == 3
        # Appellant call — dynamic schema (deep copy with enum constraints)
        _, kwargs0 = mock_llm.call_args_list[0]
        assert kwargs0["json_schema"]["type"] == AGENT_SCHEMAS["appellant"]["type"]
        assert kwargs0["json_schema"]["required"] == AGENT_SCHEMAS["appellant"]["required"]
        assert kwargs0["max_tokens"] == _MAX_TOKENS["appellant"]
        # Respondent call
        _, kwargs1 = mock_llm.call_args_list[1]
        assert kwargs1["json_schema"]["type"] == AGENT_SCHEMAS["respondent"]["type"]
        assert kwargs1["json_schema"]["required"] == AGENT_SCHEMAS["respondent"]["required"]
        assert kwargs1["max_tokens"] == _MAX_TOKENS["respondent"]
        # Judge call
        _, kwargs2 = mock_llm.call_args_list[2]
        assert kwargs2["json_schema"]["type"] == AGENT_SCHEMAS["judge"]["type"]
        assert kwargs2["json_schema"]["required"] == AGENT_SCHEMAS["judge"]["required"]
        assert kwargs2["max_tokens"] == _MAX_TOKENS["judge"]

    @patch("athena.simulation.graph.invoke_llm")
    def test_error_propagation(
        self,
        mock_llm,
        sample_case_data,
        sample_run_params,
    ):
        """If appellant fails, error is set and downstream agents are skipped."""
        mock_llm.side_effect = RuntimeError("LLM connection failed")

        phases = build_bilateral_phases(sample_case_data, sample_run_params)
        graph = build_graph_from_phases(phases)
        initial_state = {
            "case": sample_case_data,
            "params": sample_run_params,
            "briefs": {},
            "validations": {},
            "decision": None,
            "decision_validation": None,
            "error": None,
        }
        result = graph.invoke(initial_state)

        assert result["error"] is not None
        assert "failed" in result["error"]


class TestOrchestratorWithGenericGraph:
    """Test orchestrator uses generic graph and extracts results correctly."""

    @patch("athena.simulation.orchestrator.build_graph_from_phases")
    @patch("athena.simulation.orchestrator.build_bilateral_phases")
    def test_run_monte_carlo_uses_generic_graph(
        self, mock_build_phases, mock_build_graph,
    ):
        """run_monte_carlo calls build_bilateral_phases + build_graph_from_phases."""
        import os
        from athena.simulation.orchestrator import run_monte_carlo

        mock_phases = [object()]  # dummy
        mock_build_phases.return_value = mock_phases

        mock_graph = type("MockGraph", (), {})()
        mock_graph.invoke = lambda self_state: {
            "briefs": {"opponente": {"filed_brief": {"arguments": []}}, "comune_milano": {"filed_brief": {}}},
            "validations": {"opponente": {"valid": True, "warnings": []}, "comune_milano": {"valid": True, "warnings": []}},
            "decision": {"verdict": {"qualification_correct": True}},
            "decision_validation": {"valid": True, "warnings": []},
            "error": None,
        }
        # Make invoke a proper method
        mock_graph.invoke = lambda state: {
            "briefs": {"opponente": {"filed_brief": {"arguments": []}}, "comune_milano": {"filed_brief": {}}},
            "validations": {"opponente": {"valid": True, "warnings": []}, "comune_milano": {"valid": True, "warnings": []}},
            "decision": {"verdict": {"qualification_correct": True}},
            "decision_validation": {"valid": True, "warnings": []},
            "error": None,
        }
        mock_build_graph.return_value = mock_graph

        case_data = {
            "parties": [
                {"id": "opponente", "role": "appellant"},
                {"id": "comune_milano", "role": "respondent"},
            ],
        }
        sim_config = {
            "judge_profiles": [
                {"id": "j1", "party_id": "judge", "role_type": "adjudicator", "parameters": {}},
            ],
            "party_profiles": {
                "opponente": [
                    {"id": "a1", "party_id": "opponente", "role_type": "advocate", "parameters": {}},
                ],
            },
            "runs_per_combination": 1,
            "temperatures": {"appellant": 0.5, "respondent": 0.3, "judge": 0.2},
            "language": "it",
        }

        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": "1"}):
            results = run_monte_carlo(case_data, sim_config)

        mock_build_phases.assert_called_once()
        mock_build_graph.assert_called_once_with(mock_phases)
        assert len(results) == 1

    def test_run_one_extracts_from_graphstate(self):
        """_run_one maps GraphState keys to result dict correctly."""
        from unittest.mock import MagicMock
        from athena.simulation.orchestrator import _run_one

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "briefs": {
                "opponente": {"filed_brief": {"arguments": [{"id": "ARG1"}]}},
                "comune_milano": {"filed_brief": {"responses_to_opponent": []}},
            },
            "validations": {
                "opponente": {"valid": True, "warnings": ["warn1"]},
                "comune_milano": {"valid": True, "warnings": []},
            },
            "decision": {"verdict": {"qualification_correct": False}},
            "decision_validation": {"valid": True, "warnings": ["jwarn"]},
            "error": None,
        }

        case_data = {
            "parties": [
                {"id": "opponente", "role": "appellant"},
                {"id": "comune_milano", "role": "respondent"},
            ],
        }
        run_params = {
            "run_id": "j1__a1__000",
            "judge_profile": {"id": "j1"},
            "party_profiles": {
                "opponente": {"id": "a1", "role_type": "advocate"},
            },
            "temperatures": {"appellant": 0.5},
            "language": "it",
        }

        outcome = _run_one(mock_graph, case_data, run_params, 1, 1)

        assert outcome["status"] == "ok"
        r = outcome["result"]

        # Result dict preserves downstream-compatible keys
        assert r["appellant_brief"] == {"filed_brief": {"arguments": [{"id": "ARG1"}]}}
        assert r["respondent_brief"] == {"filed_brief": {"responses_to_opponent": []}}
        assert r["judge_decision"] == {"verdict": {"qualification_correct": False}}

        # Validation warnings mapped correctly
        assert r["validation_warnings"]["appellant"] == ["warn1"]
        assert r["validation_warnings"]["respondent"] == []
        assert r["validation_warnings"]["judge"] == ["jwarn"]

    def test_run_one_error_from_graphstate(self):
        """_run_one detects error from GraphState."""
        from unittest.mock import MagicMock
        from athena.simulation.orchestrator import _run_one

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "error": "opponente failed: LLM timeout",
            "briefs": {},
            "validations": {},
            "decision": None,
            "decision_validation": None,
        }

        run_params = {
            "run_id": "j1__a1__000",
            "judge_profile": {"id": "j1"},
            "party_profiles": {},
            "temperatures": {},
            "language": "it",
        }

        outcome = _run_one(mock_graph, {}, run_params, 1, 1)
        assert outcome["status"] == "fail"
        assert "LLM timeout" in outcome["error"]
