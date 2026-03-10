# tests/test_llm.py
import pytest
import json
from unittest.mock import patch, MagicMock
from athena.agents.llm import invoke_llm, parse_json_response, GenerationResult
from athena.agents.errors import JSONTruncatedError, JSONMalformedError, NonJSONOutputError


class TestParseJsonResponseWithRepair:
    """Test the new parse pipeline with repair."""

    def test_parses_clean_json(self):
        result = parse_json_response('{"key": "value"}', finish_reason="stop")
        assert result.data == {"key": "value"}
        assert result.applied_fixes == []
        assert not result.was_truncated

    def test_repairs_truncated_json(self):
        result = parse_json_response(
            '{"key": "value", "other": "trunc',
            finish_reason="length",
        )
        assert result.data["key"] == "value"
        assert result.was_truncated
        assert "repair_truncated" in result.applied_fixes

    def test_fixes_trailing_comma(self):
        result = parse_json_response('{"key": "value",}', finish_reason="stop")
        assert result.data == {"key": "value"}
        assert "trailing_commas" in result.applied_fixes

    def test_strips_thinking_blocks(self):
        raw = '<think>\nReasoning here\n</think>\n{"key": "value"}'
        result = parse_json_response(raw, finish_reason="stop")
        assert result.data == {"key": "value"}

    def test_raises_classified_error_on_non_json(self):
        with pytest.raises(NonJSONOutputError):
            parse_json_response(
                "The user wants me to act as a lawyer",
                finish_reason="stop",
                prompt_tokens=5000,
                output_tokens=2000,
            )

    def test_raises_truncated_error_when_unrepairable(self):
        with pytest.raises(JSONTruncatedError):
            parse_json_response(
                "completely broken {{{ not json at all",
                finish_reason="length",
                prompt_tokens=5000,
                output_tokens=16384,
            )


class TestInvokeLLMRefactored:
    @patch("athena.agents.llm._call_model")
    def test_returns_parsed_dict(self, mock_call):
        mock_call.return_value = ('{"test": true}', "stop", 100, 50)
        result = invoke_llm("system", "user", temperature=0.5)
        assert result == {"test": True}

    @patch("athena.agents.llm._call_model")
    def test_retries_on_unrepairable_truncation(self, mock_call):
        # First call: truncated with broken JSON that repair can't fix
        # (nested braces confuse the repair heuristic)
        mock_call.side_effect = [
            ("broken {{{ not json at all", "length", 100, 16384),
            ('{"key": "value"}', "stop", 100, 200),
        ]
        result = invoke_llm("system", "user", temperature=0.5)
        assert result == {"key": "value"}
        assert mock_call.call_count == 2

    @patch("athena.agents.llm._call_model")
    def test_saves_failure_artifact(self, mock_call, tmp_path):
        mock_call.return_value = ("not json at all", "stop", 100, 50)
        import athena.agents.llm as llm_mod
        old_dir = llm_mod._FAILURE_DIR
        llm_mod._FAILURE_DIR = str(tmp_path)
        try:
            with pytest.raises(NonJSONOutputError):
                invoke_llm("system", "user", temperature=0.5)
            artifacts = list(tmp_path.iterdir())
            assert len(artifacts) == 1
            content = artifacts[0].read_text()
            assert "not json at all" in content
        finally:
            llm_mod._FAILURE_DIR = old_dir
