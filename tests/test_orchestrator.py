# tests/test_orchestrator.py
"""Tests for the Monte Carlo orchestrator with parallel execution."""

import os
from unittest.mock import patch, MagicMock

import pytest

from athena.simulation.orchestrator import _get_concurrency, _run_one, run_monte_carlo


class TestGetConcurrency:
    """Test concurrency env var parsing."""

    def test_default_is_4(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ATHENA_CONCURRENCY", None)
            assert _get_concurrency() == 4

    def test_valid_value(self):
        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": "8"}):
            assert _get_concurrency() == 8

    def test_zero_returns_default(self):
        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": "0"}):
            assert _get_concurrency() == 4

    def test_negative_returns_default(self):
        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": "-1"}):
            assert _get_concurrency() == 4

    def test_non_numeric_returns_default(self):
        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": "abc"}):
            assert _get_concurrency() == 4

    def test_empty_returns_default(self):
        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": ""}):
            assert _get_concurrency() == 4

    def test_one_is_valid(self):
        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": "1"}):
            assert _get_concurrency() == 1


class TestRunOne:
    """Test _run_one with GraphState-shaped mock returns."""

    def test_returns_ok_on_success(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "briefs": {
                "opponente": {"filed_brief": {"arguments": []}},
                "comune_milano": {"filed_brief": {}},
            },
            "validations": {
                "opponente": {"valid": True, "warnings": []},
                "comune_milano": {"valid": True, "warnings": []},
            },
            "decision": {"verdict": {"qualification_correct": True}},
            "decision_validation": {"valid": True, "warnings": []},
            "error": None,
        }
        case_data = {
            "parties": [
                {"id": "opponente", "role": "appellant"},
                {"id": "comune_milano", "role": "respondent"},
            ],
        }
        run_params = {
            "run_id": "judge1__app1__000",
            "judge_profile": {"id": "judge1"},
            "party_profiles": {
                "opponente": {"id": "app1", "role_type": "advocate"},
            },
            "appellant_profile": {"id": "app1"},
            "temperatures": {"appellant": 0.5},
            "language": "it",
        }
        result = _run_one(mock_graph, case_data, run_params, 1, 1)
        assert result["status"] == "ok"
        assert result["run_id"] == "judge1__app1__000"
        assert "result" in result

    def test_returns_fail_on_error_state(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"error": "JSON parse failed"}
        run_params = {
            "run_id": "j__a__000",
            "judge_profile": {"id": "j"},
            "party_profiles": {},
            "temperatures": {},
            "language": "it",
        }
        result = _run_one(mock_graph, {}, run_params, 1, 1)
        assert result["status"] == "fail"
        assert "JSON parse" in result["error"]

    def test_returns_exception_on_raise(self):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("boom")
        run_params = {
            "run_id": "j__a__000",
            "judge_profile": {"id": "j"},
            "party_profiles": {},
            "temperatures": {},
            "language": "it",
        }
        result = _run_one(mock_graph, {}, run_params, 1, 1)
        assert result["status"] == "exception"
        assert "boom" in result["error"]


class TestParallelRuns:
    """Test parallel orchestrator execution."""

    @patch("athena.simulation.orchestrator.build_graph_from_phases")
    @patch("athena.simulation.orchestrator.build_bilateral_phases")
    def test_parallel_runs_all_complete(self, mock_build_phases, mock_build_graph):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "briefs": {
                "opponente": {"filed_brief": {"arguments": []}},
                "comune_milano": {"filed_brief": {}},
            },
            "validations": {
                "opponente": {"valid": True, "warnings": []},
                "comune_milano": {"valid": True, "warnings": []},
            },
            "decision": {"verdict": {"qualification_correct": True}},
            "decision_validation": {"valid": True, "warnings": []},
            "error": None,
        }
        mock_build_phases.return_value = []
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
                {"id": "j2", "party_id": "judge", "role_type": "adjudicator", "parameters": {}},
            ],
            "party_profiles": {
                "opponente": [
                    {"id": "a1", "party_id": "opponente", "role_type": "advocate", "parameters": {}},
                ],
            },
            "runs_per_combination": 2,
            "temperatures": {"appellant": 0.5, "respondent": 0.3, "judge": 0.2},
            "language": "it",
        }

        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": "4"}):
            results = run_monte_carlo(case_data, sim_config)

        assert len(results) == 4  # 2 judges × 1 appellant × 2 runs
        assert mock_graph.invoke.call_count == 4
        run_ids = {r["run_id"] for r in results}
        assert "j1__a1__000" in run_ids
        assert "j2__a1__001" in run_ids

    @patch("athena.simulation.orchestrator.build_graph_from_phases")
    @patch("athena.simulation.orchestrator.build_bilateral_phases")
    def test_mixed_success_and_failure(self, mock_build_phases, mock_build_graph):
        """Some runs succeed, some fail — all are collected."""
        call_count = 0

        def invoke_side_effect(state):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return {**state, "error": "simulated failure"}
            return {
                **state,
                "briefs": {
                    "opponente": {"filed_brief": {"arguments": []}},
                    "comune_milano": {"filed_brief": {}},
                },
                "validations": {},
                "decision": {"verdict": {"qualification_correct": True}},
                "decision_validation": None,
                "error": None,
            }

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = invoke_side_effect
        mock_build_phases.return_value = []
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
            "runs_per_combination": 4,
            "temperatures": {},
            "language": "it",
        }

        with patch.dict(os.environ, {"ATHENA_CONCURRENCY": "2"}):
            results = run_monte_carlo(case_data, sim_config)

        # Some succeed, some fail — but total invoke calls = 4
        assert mock_graph.invoke.call_count == 4
