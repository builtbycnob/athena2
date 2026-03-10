# tests/test_llm_pipeline.py
"""Integration tests for the LLM pipeline using saved/simulated outputs."""

import json
import pytest
from unittest.mock import patch
from athena.agents.llm import invoke_llm, parse_json_response, GenerationResult
from athena.agents.errors import JSONTruncatedError, NonJSONOutputError


class TestPipelineReplay:
    """Test the full parse pipeline with realistic LLM output patterns."""

    def test_clean_appellant_output(self, sample_appellant_brief):
        """Simulates a clean appellant brief output."""
        raw = json.dumps(sample_appellant_brief, ensure_ascii=False)
        result = parse_json_response(raw, finish_reason="stop")
        assert result.data["filed_brief"]["arguments"][0]["id"] == "ARG1"
        assert result.applied_fixes == []

    def test_appellant_with_thinking_block(self, sample_appellant_brief):
        """Model wraps output in thinking block despite enable_thinking=False."""
        raw = (
            "<think>\nLet me construct the legal arguments.\n</think>\n"
            + json.dumps(sample_appellant_brief, ensure_ascii=False)
        )
        result = parse_json_response(raw, finish_reason="stop")
        assert result.data["filed_brief"]["arguments"][0]["id"] == "ARG1"

    def test_respondent_with_trailing_comma(self, sample_respondent_brief):
        """Model adds trailing comma in responses array."""
        raw = json.dumps(sample_respondent_brief, ensure_ascii=False)
        # Inject a trailing comma
        raw = raw.replace("}]}", "}],}")
        result = parse_json_response(raw, finish_reason="stop")
        assert "trailing_commas" in result.applied_fixes

    def test_judge_truncated_and_repaired(self):
        """Judge output truncated mid-reasoning — repair closes it."""
        raw = (
            '{"preliminary_objections_ruling": [], '
            '"case_reaches_merits": true, '
            '"verdict": {"qualification_correct": false, '
            '"qualification_reasoning": "La qualificazione è errat'
        )
        result = parse_json_response(raw, finish_reason="length")
        assert result.was_truncated
        assert result.data["case_reaches_merits"] is True

    def test_non_json_raises_classified_error(self):
        """Model produces free text instead of JSON."""
        raw = (
            "The user wants me to act as the lawyer for the "
            "Comune di Milano in an administrative sanction appeal."
        )
        with pytest.raises(NonJSONOutputError) as exc_info:
            parse_json_response(raw, finish_reason="stop", prompt_tokens=5000, output_tokens=2000)
        assert exc_info.value.output_tokens == 2000

    @patch("athena.agents.llm._call_model")
    def test_invoke_llm_retries_on_truncation(self, mock_call):
        """Full invoke_llm pipeline: truncated → retry → success."""
        # Use output that extract_json cannot repair (no opening brace)
        truncated = '"reasoning": "incomplete analysis of the merits'
        success = '{"key": "value", "score": 7.5}'
        mock_call.side_effect = [
            (truncated, "length", 5000, 16384),
            (success, "stop", 5000, 200),
        ]
        result = invoke_llm("system", "user", temperature=0.5)
        assert result == {"key": "value", "score": 7.5}
        assert mock_call.call_count == 2

    @patch("athena.agents.llm._call_model")
    def test_invoke_llm_repairs_without_retry(self, mock_call):
        """Truncated but repairable — no retry needed."""
        repairable = '{"key": "value", "extra": "trunc'
        mock_call.return_value = (repairable, "length", 5000, 16384)
        result = invoke_llm("system", "user", temperature=0.5)
        assert result["key"] == "value"
        assert mock_call.call_count == 1  # No retry — repair succeeded
