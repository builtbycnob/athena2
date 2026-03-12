# tests/test_meta_agents.py
"""Tests for meta-agents (Red Team and Game Theorist)."""

import pytest
from unittest.mock import patch

from athena.schemas.meta_output import RED_TEAM_SCHEMA, GAME_THEORIST_SCHEMA
from athena.game_theory.schemas import (
    GameTheoryAnalysis, PartyValuations, OutcomeValuation,
    BATNA, SettlementRange, SensitivityResult,
)


# --- Test fixtures (local, specific to meta-agents) ---

def _make_aggregated():
    return {
        "probability_table": {
            ("formalista_pro_cass", "aggressivo"): {
                "n_runs": 5, "p_rejection": 0.6, "p_annulment": 0.1,
                "p_reclassification": 0.3,
                "ci_rejection": (0.23, 0.88),
            },
        },
        "argument_effectiveness": {
            "ARG1": {
                "mean_persuasiveness": 0.85, "std_persuasiveness": 0.1,
                "determinative_rate": 0.7, "n_evaluations": 10,
                "by_judge_profile": {"formalista_pro_cass": 0.6},
            },
        },
        "dominated_strategies": ["aggressivo"],
        "total_runs": 30,
        "failed_runs": 0,
    }


def _make_case_data():
    return {
        "case_id": "test-case",
        "parties": [
            {"id": "opponente", "role": "appellant"},
            {"id": "comune_milano", "role": "respondent"},
        ],
        "seed_arguments": {
            "by_party": {
                "opponente": [
                    {"id": "SEED_ARG1", "claim": "Errata qualificazione"},
                ],
            },
        },
        "stakes": {
            "current_sanction": {"fine_range": [170, 680], "points_deducted": 4},
            "litigation_cost_estimate": 1500,
        },
    }


def _make_game_analysis():
    return GameTheoryAnalysis(
        party_valuations={
            "opponente": PartyValuations(
                party_id="opponente",
                outcomes={
                    "annulment": OutcomeValuation(
                        outcome="annulment", description="Annullamento",
                        fine=0, fine_range=(0, 0), points=0, net_value=425.0,
                    ),
                },
                litigation_cost=1500, status_quo=-425.0,
            ),
        },
        batna={
            "opponente": BATNA(
                party_id="opponente", expected_value=-200.0,
                expected_value_range=(-400.0, 50.0),
                best_strategy="prudente", outcome_probabilities={"annulment": 0.3},
            ),
            "comune_milano": BATNA(
                party_id="comune_milano", expected_value=150.0,
                expected_value_range=(50.0, 300.0),
                best_strategy=None, outcome_probabilities={"rejection": 0.6},
            ),
        },
        settlement=SettlementRange(
            zopa=(100.0, 250.0), nash_solution=175.0,
            surplus=150.0, settlement_exists=True,
        ),
        sensitivity=[
            SensitivityResult(
                parameter="p_annulment", base_value=0.2,
                sweep_values=[0.1, 0.2, 0.3], ev_at_each=[-300, -200, -100],
                threshold=0.25, impact=200.0,
            ),
        ],
        expected_value_by_strategy={
            "prudente": -150.0,
            "aggressivo": -250.0,
            "tecnico": -180.0,
        },
        recommended_strategy="prudente",
        analysis_metadata={"n_runs": 30},
    )


def _make_red_team_output():
    return {
        "vulnerability_assessment": [
            {
                "target_argument_id": "ARG1",
                "attack_vector": "logical",
                "weakness_description": "Argomento testuale debole",
                "counter_argument": "Cassazione equipara le fattispecie",
                "severity": 0.7,
                "defensive_recommendation": "Rafforzare distinzione fattuale",
            },
        ],
        "strategic_vulnerabilities": [],
        "overall_risk_assessment": {
            "level": "medium",
            "reasoning": "Rischio moderato per giurisprudenza sfavorevole",
        },
    }


def _make_game_theorist_output():
    return {
        "strategic_summary": "Posizione negoziale moderatamente favorevole",
        "negotiation_position": {
            "batna_interpretation": "EV negativo, transazione preferibile",
            "zopa_assessment": "Esiste ZOPA tra 100 e 250 EUR",
            "recommended_opening": "Proporre 120 EUR",
        },
        "strategy_ranking": [
            {
                "strategy_id": "prudente",
                "expected_value_eur": -150.0,
                "risk_level": "low",
                "when_to_use": "Giudice formalista",
                "caveats": "EV comunque negativo",
            },
        ],
        "sensitivity_interpretation": "p_annulment è il parametro più influente",
        "settlement_recommendation": {
            "should_settle": True,
            "recommended_price_eur": 175.0,
            "conditions": "Rinuncia reciproca alle spese",
            "reasoning": "Nash solution a 175 EUR, entrambe le parti migliorano",
        },
    }


# --- Red Team Tests ---

class TestRedTeam:
    @patch("athena.agents.meta_agents.invoke_llm")
    def test_returns_structured_output(self, mock_llm):
        from athena.agents.meta_agents import run_red_team
        mock_llm.return_value = _make_red_team_output()
        result = run_red_team(_make_aggregated(), _make_case_data())
        assert "vulnerability_assessment" in result
        assert "strategic_vulnerabilities" in result
        assert "overall_risk_assessment" in result

    @patch("athena.agents.meta_agents.invoke_llm")
    def test_prompt_includes_argument_data(self, mock_llm):
        from athena.agents.meta_agents import run_red_team
        mock_llm.return_value = _make_red_team_output()
        run_red_team(_make_aggregated(), _make_case_data())
        call_args = mock_llm.call_args
        user_prompt = call_args[0][1]
        assert "ARG1" in user_prompt

    @patch("athena.agents.meta_agents.invoke_llm")
    def test_uses_correct_temperature(self, mock_llm):
        from athena.agents.meta_agents import run_red_team
        mock_llm.return_value = _make_red_team_output()
        run_red_team(_make_aggregated(), _make_case_data())
        call_args = mock_llm.call_args
        assert call_args.kwargs.get("temperature") == 0.6

    @patch("athena.agents.meta_agents.invoke_llm")
    def test_passes_json_schema(self, mock_llm):
        from athena.agents.meta_agents import run_red_team
        mock_llm.return_value = _make_red_team_output()
        run_red_team(_make_aggregated(), _make_case_data())
        call_args = mock_llm.call_args
        assert call_args.kwargs.get("json_schema") is not None

    @patch("athena.agents.meta_agents.invoke_llm")
    def test_prompt_includes_game_theory(self, mock_llm):
        from athena.agents.meta_agents import run_red_team
        mock_llm.return_value = _make_red_team_output()
        ga = _make_game_analysis()
        run_red_team(_make_aggregated(), _make_case_data(), game_analysis=ga)
        call_args = mock_llm.call_args
        user_prompt = call_args[0][1]
        assert "BATNA" in user_prompt

    @patch("athena.agents.meta_agents.invoke_llm")
    def test_prompt_includes_kg_insights(self, mock_llm):
        from athena.agents.meta_agents import run_red_team
        mock_llm.return_value = _make_red_team_output()
        kg = {
            "determinative_arguments": [
                {"argument_id": "ARG1", "claim": "Test", "times_determinative": 5, "total_evaluations": 10},
            ],
        }
        run_red_team(_make_aggregated(), _make_case_data(), kg_insights=kg)
        call_args = mock_llm.call_args
        user_prompt = call_args[0][1]
        assert "determinativ" in user_prompt.lower()


# --- Game Theorist Tests ---

class TestGameTheorist:
    @patch("athena.agents.meta_agents.invoke_llm")
    def test_returns_structured_output(self, mock_llm):
        from athena.agents.meta_agents import run_game_theorist
        mock_llm.return_value = _make_game_theorist_output()
        result = run_game_theorist(_make_aggregated(), _make_case_data(), _make_game_analysis())
        assert "strategic_summary" in result
        assert "settlement_recommendation" in result

    @patch("athena.agents.meta_agents.invoke_llm")
    def test_uses_correct_temperature(self, mock_llm):
        from athena.agents.meta_agents import run_game_theorist
        mock_llm.return_value = _make_game_theorist_output()
        run_game_theorist(_make_aggregated(), _make_case_data(), _make_game_analysis())
        call_args = mock_llm.call_args
        assert call_args.kwargs.get("temperature") == 0.3

    @patch("athena.agents.meta_agents.invoke_llm")
    def test_prompt_includes_ev_by_strategy(self, mock_llm):
        from athena.agents.meta_agents import run_game_theorist
        mock_llm.return_value = _make_game_theorist_output()
        run_game_theorist(_make_aggregated(), _make_case_data(), _make_game_analysis())
        call_args = mock_llm.call_args
        user_prompt = call_args[0][1]
        assert "expected_value_by_strategy" in user_prompt

    @patch("athena.agents.meta_agents.invoke_llm")
    def test_prompt_includes_settlement_data(self, mock_llm):
        from athena.agents.meta_agents import run_game_theorist
        mock_llm.return_value = _make_game_theorist_output()
        run_game_theorist(_make_aggregated(), _make_case_data(), _make_game_analysis())
        call_args = mock_llm.call_args
        user_prompt = call_args[0][1]
        # ZOPA/Nash data should be in the prompt (via model_dump)
        assert "zopa" in user_prompt.lower() or "nash" in user_prompt.lower()


# --- Schema Tests ---

class TestSchemaRegistration:
    def test_red_team_schema_registered(self):
        from athena.schemas.structured_output import AGENT_SCHEMAS
        assert "red_team" in AGENT_SCHEMAS

    def test_game_theorist_schema_registered(self):
        from athena.schemas.structured_output import AGENT_SCHEMAS
        assert "game_theorist" in AGENT_SCHEMAS

    def test_red_team_schema_is_valid_json_schema(self):
        assert RED_TEAM_SCHEMA["type"] == "object"
        assert "vulnerability_assessment" in RED_TEAM_SCHEMA["properties"]

    def test_game_theorist_schema_is_valid_json_schema(self):
        assert GAME_THEORIST_SCHEMA["type"] == "object"
        assert "strategic_summary" in GAME_THEORIST_SCHEMA["properties"]
