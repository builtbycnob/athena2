"""Tests for JSON extraction and repair (forked from ARGUS)."""

import json
import pytest
from athena.agents.json_repair import extract_json, repair_truncated_json


class TestRepairTruncatedJson:
    def test_complete_json_returns_as_is(self):
        obj = '{"title": "test", "score": 7.5}'
        result = repair_truncated_json(obj)
        assert result is not None
        assert json.loads(result) == {"title": "test", "score": 7.5}

    def test_truncated_string_value(self):
        fragment = '{"title": "AI for construct'
        result = repair_truncated_json(fragment)
        assert result is not None
        parsed = json.loads(result)
        assert "title" in parsed
        assert parsed["title"].startswith("AI for construct")

    def test_truncated_after_comma(self):
        fragment = '{"title": "test", "score": 7.5,'
        result = repair_truncated_json(fragment)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["title"] == "test"
        assert parsed["score"] == 7.5

    def test_truncated_nested_object(self):
        fragment = '{"outer": {"inner": "val"'
        result = repair_truncated_json(fragment)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "val"

    def test_truncated_array(self):
        fragment = '{"items": ["a", "b"'
        result = repair_truncated_json(fragment)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["items"] == ["a", "b"]

    def test_no_json_start(self):
        result = repair_truncated_json("just plain text")
        assert result is None

    def test_truncated_after_colon(self):
        fragment = '{"title": "test", "score":'
        result = repair_truncated_json(fragment)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["title"] == "test"

    def test_truncated_with_dangling_escape(self):
        fragment = '{"desc": "some text with a \\'
        result = repair_truncated_json(fragment)
        assert result is not None
        json.loads(result)

    def test_real_athena_judge_truncation(self):
        """Simulate realistic judge output truncation."""
        fragment = (
            '{"preliminary_objections_ruling": [], '
            '"case_reaches_merits": true, '
            '"argument_evaluation": [{"argument_id": "ARG1", "party": "appellant", '
            '"persuasiveness": 0.7, "strengths": "Forte argomentazione testuale", '
            '"weaknesses": "Cassazione contraria", "determinative": true}], '
            '"precedent_analysis": {"cass_16515_2005": {"followed": false, '
            '"distinguished": true, "reasoning": "Il caso è distinguibile perché'
        )
        result = repair_truncated_json(fragment)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["case_reaches_merits"] is True
        assert len(parsed["argument_evaluation"]) == 1


class TestExtractJsonMergedKeyColon:
    def test_merged_key_colon(self):
        bad = '{"title": "test", "description:"a service-based agency", "score": 3.5}'
        result = extract_json(bad)
        parsed = json.loads(result)
        assert parsed["title"] == "test"
        assert "description" in parsed
        assert parsed["score"] == 3.5

    def test_multiple_merged_keys(self):
        bad = '{"title:"test title", "description:"test desc", "score": 5.0}'
        result = extract_json(bad)
        parsed = json.loads(result)
        assert parsed["title"] == "test title"
        assert parsed["description"] == "test desc"

    def test_normal_json_not_affected(self):
        good = '{"title": "test", "description": "normal"}'
        result = extract_json(good)
        assert json.loads(result) == {"title": "test", "description": "normal"}

    def test_compact_json_not_affected(self):
        compact = '{"title":"test","score":3.5,"type":"saas"}'
        result = extract_json(compact)
        assert json.loads(result) == {"title": "test", "score": 3.5, "type": "saas"}


class TestExtractJsonTrailingCommas:
    def test_trailing_comma_object(self):
        text = '{"key": "val",}'
        result = extract_json(text)
        assert json.loads(result) == {"key": "val"}

    def test_trailing_comma_array(self):
        text = '{"items": ["a", "b",]}'
        result = extract_json(text)
        assert json.loads(result) == {"items": ["a", "b"]}


class TestExtractJsonSingleQuotes:
    def test_single_quotes(self):
        text = "{'key': 'value'}"
        result = extract_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_single_quotes_with_apostrophe_in_value(self):
        """Italian text with apostrophes must not be corrupted."""
        text = """{'note': "l'articolo 143 nell'ambito del CdS"}"""
        result = extract_json(text)
        parsed = json.loads(result)
        assert "l'articolo" in parsed["note"]

    def test_single_quotes_all_single(self):
        text = "{'key': 'value', 'num': 42}"
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["num"] == 42


class TestExtractJsonNewlines:
    def test_newline_in_value(self):
        text = '{"key": "line one\nline two"}'
        result = extract_json(text)
        parsed = json.loads(result)
        assert "line one" in parsed["key"]

    def test_multiline_json(self):
        text = '{"title": "test",\n  "score": 7.5}'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["title"] == "test"


class TestExtractJsonPreamble:
    def test_json_with_preamble(self):
        text = 'Here is the result:\n{"key": "value"}'
        result = extract_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_thinking_block_preamble(self):
        text = '<think>\nLet me think about this.\n</think>\n{"key": "value"}'
        result = extract_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_markdown_code_block(self):
        text = 'Result:\n```json\n{"key": "value"}\n```'
        result = extract_json(text)
        assert json.loads(result) == {"key": "value"}


class TestExtractJsonWithRepair:
    def test_truncated_json_repaired(self):
        text = '{"title": "test", "score": 7.5, "desc": "truncat'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["title"] == "test"
        assert parsed["score"] == 7.5

    def test_valid_json_skips_all_fixes(self):
        text = '{"a": 1, "b": "two"}'
        result = extract_json(text)
        assert result == text


class TestExtractJsonEdgeCases:
    def test_empty_input(self):
        result = extract_json("", return_metadata=True)
        assert result.applied_fixes == ["none_succeeded"]

    def test_whitespace_only(self):
        result = extract_json("   ", return_metadata=True)
        assert result.applied_fixes == ["none_succeeded"]


class TestExtractJsonReturnsRepairResult:
    def test_clean_json_no_fixes(self):
        text = '{"a": 1}'
        result = extract_json(text, return_metadata=True)
        assert result.text == '{"a": 1}'
        assert result.applied_fixes == []
        assert not result.was_truncated

    def test_truncated_reports_fix(self):
        text = '{"a": "trunc'
        result = extract_json(text, return_metadata=True)
        assert result.was_truncated
        assert "repair_truncated" in result.applied_fixes

    def test_trailing_comma_reports_fix(self):
        text = '{"a": 1,}'
        result = extract_json(text, return_metadata=True)
        assert "trailing_commas" in result.applied_fixes
