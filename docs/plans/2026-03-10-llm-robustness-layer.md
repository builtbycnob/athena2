# LLM Robustness Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate JSON parsing failures by adding truncation detection, JSON repair, error classification, failure artifacts, and fast smoke tests.

**Architecture:** Three-layer defense: (1) `stream_generate()` detects truncation via `finish_reason`, (2) `json_repair.py` fixes malformed/truncated JSON (forked from ARGUS), (3) retry with doubled token budget + conciseness prompt. Error types classify failures. Failure artifacts saved to disk.

**Tech Stack:** mlx_lm `stream_generate()`, Python dataclasses, pytest

---

### Task 1: JSON Repair Module

**Files:**
- Create: `src/athena/agents/json_repair.py`
- Create: `tests/test_json_repair.py`

**Step 1: Write the test file**

```python
# tests/test_json_repair.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_json_repair.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'athena.agents.json_repair'"

**Step 3: Write the json_repair module**

```python
# src/athena/agents/json_repair.py
"""JSON extraction and repair for LLM output.

Forked from ARGUS inference layer. Handles Qwen3.5-specific quirks:
- Merged key-colon ("key:"value" instead of "key": "value")
- Trailing commas
- Single quotes
- Newlines in string values
- Truncated JSON from max_tokens cutoff
"""

import json
import re
from dataclasses import dataclass, field


@dataclass
class RepairResult:
    """Result of JSON extraction with repair metadata."""
    text: str
    applied_fixes: list[str] = field(default_factory=list)
    was_truncated: bool = False


def repair_truncated_json(text: str) -> str | None:
    """Attempt to repair JSON truncated by token limit.

    Finds the first '{', tracks open delimiters via state machine,
    and closes all open strings/arrays/objects.
    Returns repaired JSON string, or None if not repairable.
    """
    start = text.find('{')
    if start == -1:
        return None

    fragment = text[start:].rstrip()

    # State machine: track open delimiters
    in_string = False
    escape_next = False
    stack: list[str] = []
    last_complete_pos = 0

    i = 0
    while i < len(fragment):
        ch = fragment[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == '\\' and in_string:
            escape_next = True
            i += 1
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if ch == '{':
            stack.append('{')
        elif ch == '[':
            stack.append('[')
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
                if not stack:
                    return fragment[:i + 1]
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
        elif ch == ',':
            last_complete_pos = i

        i += 1

    if not stack:
        return None

    repair = fragment

    # Close open string
    if in_string:
        repair = repair.rstrip()
        if repair.endswith('\\'):
            repair = repair[:-1]
        repair += '"'

    # Remove trailing comma or dangling colon
    repair = repair.rstrip()
    if repair.endswith(','):
        repair = repair[:-1]
    elif repair.endswith(':'):
        last_comma = repair.rfind(',')
        last_brace = repair.rfind('{')
        cutpoint = max(last_comma, last_brace)
        if cutpoint > 0:
            repair = repair[:cutpoint + 1]

    # Close all open delimiters
    for delim in reversed(stack):
        repair += '}' if delim == '{' else ']'

    try:
        json.loads(repair)
        return repair
    except json.JSONDecodeError:
        pass

    # Last attempt: truncate to last complete pair
    if last_complete_pos > 0:
        repair = fragment[:last_complete_pos].rstrip().rstrip(',')
        for delim in reversed(stack):
            repair += '}' if delim == '{' else ']'
        try:
            json.loads(repair)
            return repair
        except json.JSONDecodeError:
            pass

    return None


def _clean(s: str) -> str:
    """Remove control chars and fix invalid escape sequences."""
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', s)
    s = re.sub(r'\\([^"\\/bfnrtu])', r'\1', s)
    return s


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def _find_json_block(text: str) -> str:
    """Find the last balanced {...} block in text."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return _clean(stripped)
    depth = 0
    end = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '}':
            if depth == 0:
                end = i
            depth += 1
        elif text[i] == '{':
            depth -= 1
            if depth == 0:
                return _clean(text[i:end + 1])
    return _clean(stripped)


def _extract_from_markdown(text: str) -> str | None:
    """Extract JSON from markdown code block."""
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1))
            return match.group(1)
        except json.JSONDecodeError:
            pass
    return None


def extract_json(text: str, *, return_metadata: bool = False) -> str | RepairResult:
    """Extract and repair JSON from LLM output.

    Applies fixes in priority order with early exit:
    0. Qwen3.5 merged key-colon
    1. Trailing commas
    2. Single quotes
    3. Newlines in values
    4. Unquoted string values
    5. Truncated JSON repair
    6. Combined fixes

    Args:
        text: Raw LLM output text.
        return_metadata: If True, return RepairResult with fix metadata.

    Returns:
        Cleaned JSON string, or RepairResult if return_metadata=True.
    """
    applied_fixes: list[str] = []
    was_truncated = False

    # Strip thinking blocks
    cleaned = _strip_thinking(text.strip())

    # Try markdown code block extraction first
    md_result = _extract_from_markdown(cleaned)
    if md_result is not None:
        applied_fixes.append("markdown_extraction")
        result_text = md_result
        if return_metadata:
            return RepairResult(text=result_text, applied_fixes=applied_fixes, was_truncated=False)
        return result_text

    candidate = _find_json_block(cleaned)

    # Try direct parse
    try:
        json.loads(candidate)
        if return_metadata:
            return RepairResult(text=candidate, applied_fixes=[], was_truncated=False)
        return candidate
    except json.JSONDecodeError:
        pass

    # Fix 0: Qwen3.5 merged key-colon
    fixed0 = re.sub(r'([{,]\s*)"(\w+):', r'\1"\2":', candidate)
    fixed0 = re.sub(r'(":\s*)([A-Za-z$])', r'\1"\2', fixed0)
    try:
        json.loads(fixed0)
        applied_fixes.append("merged_key_colon")
        if return_metadata:
            return RepairResult(text=fixed0, applied_fixes=applied_fixes, was_truncated=False)
        return fixed0
    except json.JSONDecodeError:
        pass

    # Fix 1: Trailing commas
    fixed1 = re.sub(r',\s*([}\]])', r'\1', candidate)
    try:
        json.loads(fixed1)
        applied_fixes.append("trailing_commas")
        if return_metadata:
            return RepairResult(text=fixed1, applied_fixes=applied_fixes, was_truncated=False)
        return fixed1
    except json.JSONDecodeError:
        pass

    # Fix 2: Single quotes
    fixed2 = candidate.replace("'", '"')
    try:
        json.loads(fixed2)
        applied_fixes.append("single_quotes")
        if return_metadata:
            return RepairResult(text=fixed2, applied_fixes=applied_fixes, was_truncated=False)
        return fixed2
    except json.JSONDecodeError:
        pass

    # Fix 3: Newlines in values
    fixed3 = re.sub(r'\n\s*', ' ', candidate)
    try:
        json.loads(fixed3)
        applied_fixes.append("newlines")
        if return_metadata:
            return RepairResult(text=fixed3, applied_fixes=applied_fixes, was_truncated=False)
        return fixed3
    except json.JSONDecodeError:
        pass

    # Fix 4: Unquoted string values
    fixed4 = re.sub(
        r'"(\w+)":\s+([A-Za-z][^"]*?)(?=\s*,?\s*"[a-zA-Z_]+"\s*:|\s*})',
        lambda m: f'"{m.group(1)}": "{m.group(2).strip().rstrip(",")}"',
        candidate,
    )
    fixed4 = re.sub(r'"\s+"', '", "', fixed4)
    try:
        json.loads(fixed4)
        applied_fixes.append("unquoted_values")
        if return_metadata:
            return RepairResult(text=fixed4, applied_fixes=applied_fixes, was_truncated=False)
        return fixed4
    except json.JSONDecodeError:
        pass

    # Fix 5: Truncated JSON repair
    repaired = repair_truncated_json(cleaned)
    if repaired is not None:
        try:
            json.loads(repaired)
            applied_fixes.append("repair_truncated")
            if return_metadata:
                return RepairResult(text=repaired, applied_fixes=applied_fixes, was_truncated=True)
            return repaired
        except json.JSONDecodeError:
            pass

    # Fix 6: Combined fixes
    combined = candidate
    combined = re.sub(r'([{,]\s*)"(\w+):', r'\1"\2":', combined)
    combined = re.sub(r'(":\s*)([A-Za-z$])', r'\1"\2', combined)
    combined = re.sub(r',\s*([}\]])', r'\1', combined)
    combined = re.sub(r'\n\s*', ' ', combined)
    try:
        json.loads(combined)
        applied_fixes.append("combined")
        if return_metadata:
            return RepairResult(text=combined, applied_fixes=applied_fixes, was_truncated=False)
        return combined
    except json.JSONDecodeError:
        pass

    # Nothing worked — return the best candidate
    if return_metadata:
        return RepairResult(text=candidate, applied_fixes=["none_succeeded"], was_truncated=False)
    return candidate
```

**Step 4: Run tests**

Run: `pytest tests/test_json_repair.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/athena/agents/json_repair.py tests/test_json_repair.py
git commit -m "feat: JSON extraction and repair module (forked from ARGUS)"
```

---

### Task 2: Error Classification

**Files:**
- Create: `src/athena/agents/errors.py`
- Create: `tests/test_errors.py`

**Step 1: Write the test file**

```python
# tests/test_errors.py
from athena.agents.errors import (
    LLMError, JSONTruncatedError, JSONMalformedError,
    NonJSONOutputError, classify_error,
)


class TestClassifyError:
    def test_truncated_json(self):
        err = classify_error(
            raw_output='{"key": "val',
            finish_reason="length",
            prompt_tokens=5000,
            output_tokens=16384,
        )
        assert isinstance(err, JSONTruncatedError)
        assert err.finish_reason == "length"

    def test_malformed_json(self):
        err = classify_error(
            raw_output='{"key": value}',
            finish_reason="stop",
            prompt_tokens=5000,
            output_tokens=500,
        )
        assert isinstance(err, JSONMalformedError)

    def test_non_json(self):
        err = classify_error(
            raw_output="The user wants me to act as a lawyer",
            finish_reason="stop",
            prompt_tokens=5000,
            output_tokens=2000,
        )
        assert isinstance(err, NonJSONOutputError)

    def test_all_carry_context(self):
        err = classify_error(
            raw_output="not json",
            finish_reason="stop",
            prompt_tokens=100,
            output_tokens=50,
        )
        assert err.raw_output == "not json"
        assert err.prompt_tokens == 100
        assert err.output_tokens == 50
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_errors.py -v`
Expected: FAIL

**Step 3: Write the errors module**

```python
# src/athena/agents/errors.py
"""Classified LLM errors for structured error handling."""

from dataclasses import dataclass


@dataclass
class LLMError(Exception):
    """Base class for LLM-related errors."""
    raw_output: str
    finish_reason: str | None
    prompt_tokens: int
    output_tokens: int
    message: str = ""

    def __str__(self):
        return self.message or f"{self.__class__.__name__}: {self.raw_output[:100]}..."


@dataclass
class JSONTruncatedError(LLMError):
    """JSON output was truncated by max_tokens limit."""
    message: str = "JSON truncated at max_tokens"


@dataclass
class JSONMalformedError(LLMError):
    """JSON output is structurally invalid (not truncated)."""
    message: str = "JSON structurally malformed"


@dataclass
class NonJSONOutputError(LLMError):
    """Model produced free text instead of JSON."""
    message: str = "Model produced non-JSON output"


def classify_error(
    raw_output: str,
    finish_reason: str | None,
    prompt_tokens: int,
    output_tokens: int,
) -> LLMError:
    """Classify a JSON parsing failure into a specific error type."""
    kwargs = dict(
        raw_output=raw_output,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )

    # Truncated: finish_reason is "length"
    if finish_reason == "length":
        return JSONTruncatedError(**kwargs)

    # Check if it looks like JSON at all
    stripped = raw_output.strip()
    has_brace = '{' in stripped
    if not has_brace:
        return NonJSONOutputError(**kwargs)

    # Has JSON structure but can't parse → malformed
    return JSONMalformedError(**kwargs)
```

**Step 4: Run tests**

Run: `pytest tests/test_errors.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/athena/agents/errors.py tests/test_errors.py
git commit -m "feat: classified LLM error types"
```

---

### Task 3: Refactor LLM Layer — stream_generate + repair + retry

**Files:**
- Modify: `src/athena/agents/llm.py`
- Modify: `tests/test_llm.py`

**Step 1: Write new tests for the refactored LLM layer**

Add to `tests/test_llm.py`:

```python
# Add these imports at top
from athena.agents.llm import parse_json_response, GenerationResult
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
        # Deeply nested truncation that can't be repaired
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
    def test_repairs_truncated_and_retries(self, mock_call):
        # First call: truncated. Second call (retry): succeeds
        mock_call.side_effect = [
            ('{"key": "val', "length", 100, 16384),  # truncated, unrepairable
            ('{"key": "value"}', "stop", 100, 200),   # retry succeeds
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
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL (GenerationResult doesn't exist yet, _call_model signature changed)

**Step 3: Rewrite llm.py**

```python
# src/athena/agents/llm.py
"""LLM integration layer with stream_generate, JSON repair, and retry.

Uses mlx_lm stream_generate() for truncation detection (finish_reason),
json_repair for fixing malformed output, and retry with doubled budget.
"""

import json
import os
import time
from dataclasses import dataclass, field
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from athena.agents.json_repair import extract_json, RepairResult
from athena.agents.errors import (
    LLMError, JSONTruncatedError, JSONMalformedError,
    NonJSONOutputError, classify_error,
)


_MODEL = None
_TOKENIZER = None
_MODEL_PATH = "mlx-community/Qwen3.5-35B-A3B-4bit"
_CONTEXT_WINDOW = 262144  # Qwen3.5 max_position_embeddings
_DEFAULT_MAX_TOKENS = 16384
_FAILURE_DIR = "output/failures"

# Cumulative stats
_stats = {
    "calls": 0, "total_tokens": 0, "total_time": 0.0,
    "repairs": 0, "truncations": 0, "retries": 0,
    "repair_types": {},
}


@dataclass
class GenerationResult:
    """Result of JSON extraction from LLM output."""
    data: dict
    applied_fixes: list[str] = field(default_factory=list)
    was_truncated: bool = False


def _ensure_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        print(f"[LLM] Loading model: {_MODEL_PATH}", flush=True)
        t0 = time.time()
        _MODEL, _TOKENIZER = load(_MODEL_PATH)
        print(f"[LLM] Model loaded in {time.time()-t0:.1f}s", flush=True)


def _call_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> tuple[str, str, int, int]:
    """Call MLX model via stream_generate. Returns (text, finish_reason, prompt_tokens, output_tokens)."""
    _ensure_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    sampler = make_sampler(temp=temperature)
    t0 = time.time()

    text = ""
    finish_reason = None
    prompt_tokens = 0
    output_tokens = 0

    for response in stream_generate(
        _MODEL, _TOKENIZER, prompt=prompt, sampler=sampler, max_tokens=max_tokens,
    ):
        text += response.text
        finish_reason = response.finish_reason
        prompt_tokens = response.prompt_tokens
        output_tokens = response.generation_tokens

    elapsed = time.time() - t0
    tok_s = output_tokens / elapsed if elapsed > 0 else 0

    _stats["calls"] += 1
    _stats["total_tokens"] += output_tokens
    _stats["total_time"] += elapsed

    if finish_reason == "length":
        _stats["truncations"] += 1

    # Token budget warning
    budget_remaining = _CONTEXT_WINDOW - prompt_tokens - output_tokens
    budget_pct = output_tokens / max_tokens * 100 if max_tokens > 0 else 0

    truncation_flag = " TRUNCATED" if finish_reason == "length" else ""
    budget_warn = f" ⚠ {budget_pct:.0f}% of budget used" if budget_pct > 90 else ""

    print(
        f"[LLM] Call #{_stats['calls']}: "
        f"{prompt_tokens} prompt → {output_tokens} output tok, "
        f"{elapsed:.1f}s ({tok_s:.1f} tok/s), "
        f"temp={temperature}{truncation_flag}{budget_warn}",
        flush=True,
    )

    return text, finish_reason, prompt_tokens, output_tokens


def _save_failure_artifact(raw: str, context: str = "") -> None:
    """Save failed LLM output to disk for offline debugging."""
    os.makedirs(_FAILURE_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_FAILURE_DIR, f"{ts}_{_stats['calls']}.txt")
    with open(path, "w") as f:
        if context:
            f.write(f"--- CONTEXT ---\n{context}\n\n")
        f.write(f"--- RAW OUTPUT ({len(raw)} chars) ---\n{raw}\n")
    print(f"[LLM] Failure artifact saved: {path}", flush=True)


def get_stats() -> dict:
    """Return cumulative LLM call statistics."""
    return {
        **_stats,
        "avg_tok_s": _stats["total_tokens"] / _stats["total_time"]
        if _stats["total_time"] > 0 else 0,
    }


def parse_json_response(
    raw: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    output_tokens: int = 0,
) -> GenerationResult:
    """Extract and parse JSON from LLM response with repair.

    Raises classified LLMError subclass on failure.
    """
    repair_result = extract_json(raw, return_metadata=True)

    if repair_result.applied_fixes:
        for fix in repair_result.applied_fixes:
            _stats["repair_types"][fix] = _stats["repair_types"].get(fix, 0) + 1
        if repair_result.applied_fixes != ["none_succeeded"]:
            _stats["repairs"] += 1

    try:
        data = json.loads(repair_result.text)
        return GenerationResult(
            data=data,
            applied_fixes=repair_result.applied_fixes,
            was_truncated=repair_result.was_truncated,
        )
    except json.JSONDecodeError:
        err = classify_error(
            raw_output=raw,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        raise err


def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> dict:
    """Invoke LLM and return parsed JSON dict.

    Pipeline:
    1. Call model via stream_generate
    2. Extract + repair JSON
    3. If truncated and unrepairable: retry with 2x budget + conciseness hint
    4. If still fails: save artifact and raise classified error
    """
    raw, finish_reason, prompt_tokens, output_tokens = _call_model(
        system_prompt, user_prompt, temperature, max_tokens
    )

    try:
        result = parse_json_response(raw, finish_reason, prompt_tokens, output_tokens)
        if result.applied_fixes:
            fixes_str = ", ".join(result.applied_fixes)
            print(f"[LLM] JSON repaired: {fixes_str}", flush=True)
        return result.data
    except JSONTruncatedError:
        # Retry with doubled budget + conciseness hint
        retry_max = max_tokens * 2
        _stats["retries"] += 1
        print(
            f"[LLM] Truncated at {output_tokens} tok, retrying with {retry_max} max_tokens",
            flush=True,
        )
        retry_user = (
            f"{user_prompt}\n\n"
            "## NOTA: il tuo output precedente è stato troncato. "
            "Sii più conciso. Riduci il reasoning a 3-5 frasi per argomento. "
            "Produci l'output JSON completo."
        )
        raw2, fr2, pt2, ot2 = _call_model(
            system_prompt, retry_user, temperature, retry_max
        )
        try:
            result2 = parse_json_response(raw2, fr2, pt2, ot2)
            if result2.applied_fixes:
                print(f"[LLM] Retry repaired: {', '.join(result2.applied_fixes)}", flush=True)
            return result2.data
        except LLMError:
            _save_failure_artifact(raw2, context=f"Retry also failed. Original: {raw[:500]}")
            raise
    except LLMError:
        _save_failure_artifact(raw, context=f"finish_reason={finish_reason}")
        raise
```

**Step 4: Update existing tests in test_llm.py**

Replace the existing `TestParseJsonResponse` and `TestInvokeLLM` classes with the new versions from Step 1 above. Keep imports updated.

**Step 5: Run tests**

Run: `pytest tests/test_llm.py -v`
Expected: ALL PASS

**Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS (the graph tests mock `invoke_llm` which still returns dict)

**Step 7: Commit**

```bash
git add src/athena/agents/llm.py tests/test_llm.py
git commit -m "feat: stream_generate with truncation detection, repair, and retry"
```

---

### Task 4: Smoke Test Configurations

**Files:**
- Create: `simulations/smoke-1.yaml`
- Create: `simulations/smoke-3.yaml`

**Step 1: Create smoke-1 (single run, ~70s)**

```yaml
# simulations/smoke-1.yaml — Quick validation: 1 judge × 1 style × 1 run
simulation:
  case_ref: "gdp-milano-17928-2025"
  language: "it"

  judge_profiles:
    - id: "formalista_pro_cass"
      jurisprudential_orientation: "follows_cassazione"
      formalism: "high"

  appellant_profiles:
    - id: "tecnico"
      style: |
        Concentrati sui vizi formali del verbale e sulla lettera
        della legge. Minimizza il ruolo della giurisprudenza.
        Argomentazione analitica e testuale.

  temperature:
    appellant: 0.5
    respondent: 0.4
    judge: 0.3

  runs_per_combination: 1
```

**Step 2: Create smoke-3 (all styles, ~210s)**

```yaml
# simulations/smoke-3.yaml — Test all advocacy styles: 1 judge × 3 styles × 1 run
simulation:
  case_ref: "gdp-milano-17928-2025"
  language: "it"

  judge_profiles:
    - id: "formalista_pro_cass"
      jurisprudential_orientation: "follows_cassazione"
      formalism: "high"

  appellant_profiles:
    - id: "aggressivo"
      style: |
        Attacca frontalmente la giurisprudenza sfavorevole.
        Obiettivo primario: annullamento. Toni decisi,
        argomentazione assertiva.
    - id: "prudente"
      style: |
        Distingui la giurisprudenza senza attaccarla direttamente.
        Presenta la riqualificazione come esito ragionevole e
        proporzionato. Toni collaborativi con il giudice.
    - id: "tecnico"
      style: |
        Concentrati sui vizi formali del verbale e sulla lettera
        della legge. Minimizza il ruolo della giurisprudenza.
        Argomentazione analitica e testuale.

  temperature:
    appellant: 0.5
    respondent: 0.4
    judge: 0.3

  runs_per_combination: 1
```

**Step 3: Commit**

```bash
git add simulations/smoke-1.yaml simulations/smoke-3.yaml
git commit -m "feat: smoke test configs (1-run and 3-styles)"
```

---

### Task 5: Update Monitor with Repair Stats

**Files:**
- Modify: `scripts/monitor.py`

**Step 1: Add repair tracking to parse_log**

Add to the `parse_log` function, after the LLM call parsing:

```python
        # LLM repair
        m = re.search(r"\[LLM\] JSON repaired: (.+)", line)
        if m:
            repairs.append(m.group(1))

        # LLM truncation + retry
        m = re.search(r"\[LLM\] Truncated at", line)
        if m:
            truncations += 1

        # LLM failure artifact
        m = re.search(r"\[LLM\] Failure artifact saved: (.+)", line)
        if m:
            failure_artifacts.append(m.group(1))
```

Add these counters to the returned dict and display in `format_report`:

```
  Repairs:       3 (trailing_commas: 2, repair_truncated: 1)
  Truncations:   1 (0 retried successfully)
  Artifacts:     output/failures/20260310_... (1 saved)
```

**Step 2: Commit**

```bash
git add scripts/monitor.py
git commit -m "feat: monitor tracks repair stats, truncations, failure artifacts"
```

---

### Task 6: Integration Test with Saved Fixtures

**Files:**
- Create: `tests/fixtures/llm_outputs/` directory
- Create: `tests/test_llm_pipeline.py`

**Step 1: Create fixture directory and save first real outputs**

After running smoke-1, copy the raw outputs from a successful run:

```bash
mkdir -p tests/fixtures/llm_outputs
```

The test will use the existing `sample_appellant_brief` and `sample_respondent_brief` fixtures from conftest.py, serialized as JSON strings (simulating raw LLM output).

**Step 2: Write pipeline replay test**

```python
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
        truncated = '{"key": "val'
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
```

**Step 3: Run tests**

Run: `pytest tests/test_llm_pipeline.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/test_llm_pipeline.py tests/fixtures/
git commit -m "feat: pipeline integration tests with realistic LLM output patterns"
```

---

### Task 7: Verify Full Suite + Smoke Run

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 2: Run smoke-1 with real LLM**

Run: `athena run --case cases/gdp-milano-17928-2025.yaml --simulation simulations/smoke-1.yaml --output output/smoke-1`
Expected: 1/1 succeeded, repair stats in output

**Step 3: Check monitor output**

Run: `python scripts/monitor.py`
Expected: Shows progress, repair stats, no failures

**Step 4: Final commit with any adjustments**

```bash
git add -A
git commit -m "chore: verify full suite + smoke run"
```
