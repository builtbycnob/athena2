# tests/test_llm.py
import pytest
import json
from unittest.mock import patch, MagicMock
from athena.agents.llm import invoke_llm, parse_json_response


class TestParseJsonResponse:
    def test_parses_clean_json(self):
        raw = '{"key": "value"}'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parses_json_in_markdown_block(self):
        raw = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parses_json_with_trailing_text(self):
        raw = '{"key": "value"}\n\nSome explanation after.'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_raises_on_invalid_json(self):
        raw = "This is not JSON at all"
        with pytest.raises(ValueError):
            parse_json_response(raw)


class TestInvokeLLM:
    @patch("athena.agents.llm._call_model")
    def test_returns_parsed_dict(self, mock_call):
        mock_call.return_value = '{"test": true}'
        result = invoke_llm("system", "user", temperature=0.5)
        assert result == {"test": True}

    @patch("athena.agents.llm._call_model")
    def test_raises_on_invalid_response(self, mock_call):
        mock_call.return_value = "not json"
        with pytest.raises(ValueError):
            invoke_llm("system", "user", temperature=0.5)
