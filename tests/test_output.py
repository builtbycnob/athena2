# tests/test_output.py
import pytest
from unittest.mock import patch, MagicMock
from athena.output.table import format_probability_table
from athena.output.decision_tree import generate_decision_tree
from athena.output.memo import generate_strategic_memo


def _make_aggregated():
    """Build a realistic aggregated result dict for testing."""
    return {
        "probability_table": {
            ("formalista_pro_cass", "aggressivo"): {
                "n_runs": 5,
                "p_rejection": 0.6,
                "ci_low": 0.23,
                "ci_high": 0.88,
                "p_annulment": 0.1,
                "ci_annulment": (0.02, 0.40),
                "p_reclassification": 0.3,
                "ci_reclassification": (0.10, 0.60),
                "ci_rejection": (0.23, 0.88),
            },
            ("formalista_pro_cass", "prudente"): {
                "n_runs": 5,
                "p_rejection": 0.4,
                "ci_low": 0.12,
                "ci_high": 0.77,
                "p_annulment": 0.2,
                "ci_annulment": (0.05, 0.55),
                "p_reclassification": 0.4,
                "ci_reclassification": (0.12, 0.77),
                "ci_rejection": (0.12, 0.77),
            },
            ("formalista_pro_cass", "tecnico"): {
                "n_runs": 5,
                "p_rejection": 0.5,
                "ci_low": 0.18,
                "ci_high": 0.82,
                "p_annulment": 0.15,
                "ci_annulment": (0.03, 0.45),
                "p_reclassification": 0.35,
                "ci_reclassification": (0.12, 0.65),
                "ci_rejection": (0.18, 0.82),
            },
            ("sostanzialista_anti_cass", "aggressivo"): {
                "n_runs": 5,
                "p_rejection": 0.2,
                "ci_low": 0.04,
                "ci_high": 0.62,
                "p_annulment": 0.35,
                "ci_annulment": (0.12, 0.65),
                "p_reclassification": 0.45,
                "ci_reclassification": (0.18, 0.75),
                "ci_rejection": (0.04, 0.62),
            },
            ("sostanzialista_anti_cass", "prudente"): {
                "n_runs": 5,
                "p_rejection": 0.2,
                "ci_low": 0.04,
                "ci_high": 0.62,
                "p_annulment": 0.2,
                "ci_annulment": (0.05, 0.55),
                "p_reclassification": 0.6,
                "ci_reclassification": (0.23, 0.88),
                "ci_rejection": (0.04, 0.62),
            },
            ("sostanzialista_anti_cass", "tecnico"): {
                "n_runs": 5,
                "p_rejection": 0.2,
                "ci_low": 0.04,
                "ci_high": 0.62,
                "p_annulment": 0.3,
                "ci_annulment": (0.10, 0.60),
                "p_reclassification": 0.5,
                "ci_reclassification": (0.18, 0.82),
                "ci_rejection": (0.04, 0.62),
            },
        },
        "argument_effectiveness": {
            "ARG1": {
                "mean_persuasiveness": 0.85,
                "std_persuasiveness": 0.1,
                "determinative_rate": 0.7,
                "n_evaluations": 10,
                "by_judge_profile": {
                    "formalista_pro_cass": 0.6,
                    "sostanzialista_anti_cass": 0.9,
                },
            },
            "ARG2": {
                "mean_persuasiveness": 0.45,
                "std_persuasiveness": 0.2,
                "determinative_rate": 0.2,
                "n_evaluations": 10,
                "by_judge_profile": {
                    "formalista_pro_cass": 0.5,
                    "sostanzialista_anti_cass": 0.4,
                },
            },
            "ARG3": {
                "mean_persuasiveness": 0.70,
                "std_persuasiveness": 0.15,
                "determinative_rate": 0.5,
                "n_evaluations": 10,
                "by_judge_profile": {
                    "formalista_pro_cass": 0.75,
                    "sostanzialista_anti_cass": 0.65,
                },
            },
        },
        "precedent_analysis": {
            "cass_16515_2005": {
                "followed_rate": 0.6,
                "distinguished_rate": 0.4,
                "by_judge_profile": {
                    "formalista_pro_cass": {"followed_rate": 0.8},
                    "sostanzialista_anti_cass": {"followed_rate": 0.4},
                },
            },
        },
        "total_runs": 30,
        "failed_runs": 0,
        "dominated_strategies": [],
    }


class TestFormatProbabilityTable:
    def test_returns_string(self):
        agg = _make_aggregated()
        result = format_probability_table(agg)
        assert isinstance(result, str)

    def test_contains_markdown_table_headers(self):
        agg = _make_aggregated()
        result = format_probability_table(agg)
        # Should contain pipe-delimited table
        assert "|" in result
        # Should contain judge profile names
        assert "formalista_pro_cass" in result
        assert "sostanzialista_anti_cass" in result

    def test_contains_all_styles_as_columns(self):
        agg = _make_aggregated()
        result = format_probability_table(agg)
        # Header row should have all three styles
        assert "aggressivo" in result.lower() or "Aggressivo" in result
        assert "prudente" in result.lower() or "Prudente" in result
        assert "tecnico" in result.lower() or "Tecnico" in result

    def test_shows_percentages(self):
        agg = _make_aggregated()
        result = format_probability_table(agg)
        # Should show percentage values (e.g. 60%)
        assert "%" in result

    def test_shows_confidence_intervals(self):
        agg = _make_aggregated()
        result = format_probability_table(agg)
        # CI format: [xx-yy%]
        assert "[" in result
        assert "]" in result

    def test_shows_annulment_and_reclassification(self):
        agg = _make_aggregated()
        result = format_probability_table(agg)
        # Should show A: and R: and X: labels
        assert "A:" in result
        assert "R:" in result
        assert "X:" in result

    def test_correct_number_of_rows(self):
        agg = _make_aggregated()
        result = format_probability_table(agg)
        lines = [l for l in result.strip().split("\n") if l.strip().startswith("|")]
        # Header + separator + 2 judge profiles = 4 lines minimum
        assert len(lines) >= 4

    def test_shows_legend(self):
        agg = _make_aggregated()
        result = format_probability_table(agg)
        assert "annullamento" in result.lower() or "A =" in result

    def test_empty_table(self):
        agg = _make_aggregated()
        agg["probability_table"] = {}
        result = format_probability_table(agg)
        assert isinstance(result, str)


class TestGenerateDecisionTree:
    def test_returns_string(self):
        agg = _make_aggregated()
        result = generate_decision_tree(agg)
        assert isinstance(result, str)

    def test_identifies_best_style_per_profile(self):
        agg = _make_aggregated()
        result = generate_decision_tree(agg)
        # For formalista_pro_cass, aggressivo has lowest p_rejection (0.6 vs 0.4 vs 0.5)
        # But "best" = highest p_rejection failure for the opponent, meaning
        # highest chance the appellant wins = lowest p_rejection
        # Actually: p_rejection = probability of rejection (bad for appellant)
        # So best style = lowest p_rejection
        # formalista_pro_cass: aggressivo=0.6, prudente=0.4, tecnico=0.5 → prudente best
        assert "prudente" in result.lower()

    def test_shows_all_judge_profiles(self):
        agg = _make_aggregated()
        result = generate_decision_tree(agg)
        assert "formalista_pro_cass" in result
        assert "sostanzialista_anti_cass" in result

    def test_detects_dominated_strategy(self):
        agg = _make_aggregated()
        agg["dominated_strategies"] = ["aggressivo"]
        result = generate_decision_tree(agg)
        assert "dominat" in result.lower()
        assert "aggressivo" in result.lower()

    def test_no_dominated_strategies(self):
        agg = _make_aggregated()
        agg["dominated_strategies"] = []
        result = generate_decision_tree(agg)
        # Should still generate valid output
        assert isinstance(result, str)
        assert len(result) > 0

    def test_shows_argument_effectiveness_ranking(self):
        agg = _make_aggregated()
        result = generate_decision_tree(agg)
        # Should mention argument IDs
        assert "ARG1" in result
        assert "ARG2" in result

    def test_shows_success_probability(self):
        agg = _make_aggregated()
        result = generate_decision_tree(agg)
        assert "%" in result


class TestGenerateStrategicMemo:
    @patch("athena.output.memo._call_model")
    def test_returns_llm_response(self, mock_call):
        mock_call.return_value = ("Memo strategico generato dal modello.", "stop", 100, 50)
        agg = _make_aggregated()
        case_data = {"case": {"id": "test-case", "title": "Test"}}
        result = generate_strategic_memo(agg, case_data)
        assert result == "Memo strategico generato dal modello."
        mock_call.assert_called_once()

    @patch("athena.output.memo._call_model")
    def test_prompt_includes_probability_data(self, mock_call):
        mock_call.return_value = ("Memo.", "stop", 100, 50)
        agg = _make_aggregated()
        case_data = {"case": {"id": "test-case", "title": "Test"}}
        generate_strategic_memo(agg, case_data)
        call_args = mock_call.call_args
        user_prompt = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("user_prompt", "")
        # Should include probability table data
        assert "formalista_pro_cass" in user_prompt
        assert "probability" in user_prompt.lower() or "probabilist" in user_prompt.lower()

    @patch("athena.output.memo._call_model")
    def test_prompt_includes_argument_effectiveness(self, mock_call):
        mock_call.return_value = ("Memo.", "stop", 100, 50)
        agg = _make_aggregated()
        case_data = {"case": {"id": "test-case", "title": "Test"}}
        generate_strategic_memo(agg, case_data)
        call_args = mock_call.call_args
        user_prompt = call_args[0][1]
        assert "ARG1" in user_prompt
        assert "ARG2" in user_prompt

    @patch("athena.output.memo._call_model")
    def test_prompt_includes_precedent_analysis(self, mock_call):
        mock_call.return_value = ("Memo.", "stop", 100, 50)
        agg = _make_aggregated()
        case_data = {"case": {"id": "test-case", "title": "Test"}}
        generate_strategic_memo(agg, case_data)
        call_args = mock_call.call_args
        user_prompt = call_args[0][1]
        assert "cass_16515_2005" in user_prompt

    @patch("athena.output.memo._call_model")
    def test_prompt_includes_dominated_strategies(self, mock_call):
        mock_call.return_value = ("Memo.", "stop", 100, 50)
        agg = _make_aggregated()
        agg["dominated_strategies"] = ["aggressivo"]
        case_data = {"case": {"id": "test-case", "title": "Test"}}
        generate_strategic_memo(agg, case_data)
        call_args = mock_call.call_args
        user_prompt = call_args[0][1]
        assert "aggressivo" in user_prompt

    @patch("athena.output.memo._call_model")
    def test_prompt_includes_case_data(self, mock_call):
        mock_call.return_value = ("Memo.", "stop", 100, 50)
        agg = _make_aggregated()
        case_data = {"case": {"id": "test-case", "title": "Test opposizione"}}
        generate_strategic_memo(agg, case_data)
        call_args = mock_call.call_args
        user_prompt = call_args[0][1]
        assert "test-case" in user_prompt or "Test opposizione" in user_prompt

    @patch("athena.output.memo._call_model")
    def test_system_prompt_requests_italian(self, mock_call):
        mock_call.return_value = ("Memo.", "stop", 100, 50)
        agg = _make_aggregated()
        case_data = {"case": {"id": "test"}}
        generate_strategic_memo(agg, case_data)
        call_args = mock_call.call_args
        system_prompt = call_args[0][0]
        # System prompt should instruct writing in Italian
        assert "avvocato" in system_prompt.lower() or "italiano" in system_prompt.lower() or "strategico" in system_prompt.lower()

    @patch("athena.output.memo._call_model")
    def test_n_runs_in_prompt(self, mock_call):
        mock_call.return_value = ("Memo.", "stop", 100, 50)
        agg = _make_aggregated()
        case_data = {"case": {"id": "test"}}
        generate_strategic_memo(agg, case_data)
        call_args = mock_call.call_args
        # total_runs should appear in either system or user prompt
        system_prompt = call_args[0][0]
        user_prompt = call_args[0][1]
        combined = system_prompt + user_prompt
        assert "30" in combined  # total_runs = 30
