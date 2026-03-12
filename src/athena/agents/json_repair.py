"""JSON extraction and repair for LLM output.

Forked from ARGUS inference layer. Three-layer defense:

1. **Prevention**: Prompt instructions tell the model to avoid unescaped quotes.
2. **Targeted fixes**: Fast regex fixes for known Qwen3.5-specific patterns
   (merged key-colon, trailing commas, single quotes).
3. **Library repair**: `json_repair` — BNF-based recursive descent parser that
   handles embedded quotes, structural errors, and other malformed JSON that
   regex heuristics cannot reliably fix.
4. **Truncation repair**: State-machine bracket closer for max_tokens cutoff.

The `json_repair` library replaces ~150 lines of hand-written embedded-quote
state machine / iterative repair that was fragile and could not handle all
failure modes (e.g. stray brackets from schema-type confusion).
"""

import json
import re
from dataclasses import dataclass, field

from json_repair import repair_json


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
    """Remove control chars, normalize curly quotes, fix invalid escape sequences."""
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', s)
    # Normalize curly/smart quotes to straight quotes (common in Italian LLM output)
    s = s.replace('\u201c', '"').replace('\u201d', '"')  # " "
    s = s.replace('\u2018', "'").replace('\u2019', "'")  # ' '
    s = re.sub(r'\\([^"\\/bfnrtu])', r'\1', s)
    return s


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def _find_json_block(text: str) -> str:
    """Find the last balanced {...} block in text.

    NOTE: The reverse-scan brace counting does not track whether braces
    appear inside JSON string values.  This is intentional — the function
    is only the initial candidate finder and downstream repair stages
    handle edge cases where braces appear inside strings.
    """
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

    Pipeline (early exit on first success):
    1. Markdown code block extraction
    2. Direct parse
    3. Targeted regex fixes (merged key-colon, trailing commas, single quotes)
    4. json_repair library (BNF recursive descent — handles embedded quotes,
       structural errors, and other complex malformations)
    5. Truncated JSON repair (state-machine bracket closer)

    Args:
        text: Raw LLM output text.
        return_metadata: If True, return RepairResult with fix metadata.

    Returns:
        Cleaned JSON string, or RepairResult if return_metadata=True.
    """
    applied_fixes: list[str] = []
    was_truncated = False

    def _ok(text: str) -> str | RepairResult:
        if return_metadata:
            return RepairResult(text=text, applied_fixes=applied_fixes, was_truncated=was_truncated)
        return text

    # Strip thinking blocks
    cleaned = _strip_thinking(text.strip())

    # Try markdown code block extraction first
    md_result = _extract_from_markdown(cleaned)
    if md_result is not None:
        applied_fixes.append("markdown_extraction")
        return _ok(md_result)

    candidate = _find_json_block(cleaned)

    # Try direct parse
    try:
        json.loads(candidate)
        return _ok(candidate)
    except json.JSONDecodeError:
        pass

    # Fix 0: Qwen3.5 merged key-colon ("key:"value" → "key": "value")
    fixed0 = re.sub(r'([{,]\s*)"(\w+):', r'\1"\2":', candidate)
    fixed0 = re.sub(r'(":\s*)([A-Za-z$])', r'\1"\2', fixed0)
    try:
        json.loads(fixed0)
        applied_fixes.append("merged_key_colon")
        return _ok(fixed0)
    except json.JSONDecodeError:
        pass

    # Fix 1: Trailing commas
    fixed1 = re.sub(r',\s*([}\]])', r'\1', candidate)
    try:
        json.loads(fixed1)
        applied_fixes.append("trailing_commas")
        return _ok(fixed1)
    except json.JSONDecodeError:
        pass

    # Fix 2: Single quotes — only replace quotes acting as JSON delimiters,
    # not apostrophes inside text values (e.g. Italian "l'articolo").
    fixed2 = re.sub(
        r"(?<=[\[{,:])\s*'|'\s*(?=[:,\]}])",
        lambda m: m.group().replace("'", '"'),
        candidate,
    )
    try:
        json.loads(fixed2)
        applied_fixes.append("single_quotes")
        return _ok(fixed2)
    except json.JSONDecodeError:
        pass

    # Fix 3: json_repair library — BNF-based recursive descent parser.
    # Handles embedded quotes, structural errors, stray brackets, and other
    # complex malformations that regex heuristics cannot reliably fix.
    try:
        library_repaired = repair_json(candidate, return_objects=False)
        json.loads(library_repaired)
        applied_fixes.append("json_repair_library")
        return _ok(library_repaired)
    except (json.JSONDecodeError, Exception):
        pass

    # Fix 4: Truncated JSON repair (for max_tokens cutoff)
    repaired = repair_truncated_json(cleaned)
    if repaired is not None:
        try:
            json.loads(repaired)
            applied_fixes.append("repair_truncated")
            was_truncated = True
            return _ok(repaired)
        except json.JSONDecodeError:
            pass

    # Fix 5: Library repair on cleaned text (before _find_json_block extraction)
    try:
        library_repaired2 = repair_json(cleaned, return_objects=False)
        json.loads(library_repaired2)
        applied_fixes.append("json_repair_library_raw")
        return _ok(library_repaired2)
    except (json.JSONDecodeError, Exception):
        pass

    # Nothing worked — return the best candidate
    if return_metadata:
        return RepairResult(text=candidate, applied_fixes=["none_succeeded"], was_truncated=False)
    return candidate
