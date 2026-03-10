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


def _fix_embedded_quotes(text: str) -> str:
    """Fix unescaped double quotes used for emphasis inside JSON string values.

    Common in Italian legal text where the model writes e.g.:
        "claim": "...il reato di "contromano", bensì..."
    The inner "contromano" quotes break JSON parsing.

    Uses a state machine: when inside a string and we encounter a `"`,
    we check if what follows looks like valid JSON continuation.
    If not, it's an embedded quote and we escape it.
    """
    result = []
    i = 0
    in_string = False

    while i < len(text):
        ch = text[i]

        # Handle escape sequences inside strings
        if ch == '\\' and in_string and i + 1 < len(text):
            result.append(ch)
            result.append(text[i + 1])
            i += 2
            continue

        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
                i += 1
            else:
                # Potential end of string — look ahead to decide
                j = i + 1
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1

                if j >= len(text) or text[j] in '}]':
                    # End of object/array — real terminator
                    in_string = False
                    result.append(ch)
                    i += 1
                elif text[j] == ':':
                    # Ended a key — real terminator
                    in_string = False
                    result.append(ch)
                    i += 1
                elif text[j] == ',':
                    # After comma: check if next token is valid JSON
                    k = j + 1
                    while k < len(text) and text[k] in ' \t\n\r':
                        k += 1
                    if k < len(text) and text[k] == '"':
                        # Comma then quote — could be next key/value or embedded
                        # Find closing quote of potential key/value
                        m = k + 1
                        while m < len(text):
                            if text[m] == '\\' and m + 1 < len(text):
                                m += 2
                                continue
                            if text[m] == '"':
                                break
                            m += 1
                        if m < len(text):
                            # Check what follows the closing quote
                            n = m + 1
                            while n < len(text) and text[n] in ' \t\n\r':
                                n += 1
                            if n < len(text) and text[n] in ':,}]':
                                # Valid JSON continuation — real terminator
                                in_string = False
                                result.append(ch)
                                i += 1
                            else:
                                # Not valid JSON — embedded quote
                                result.append('\\"')
                                i += 1
                        else:
                            in_string = False
                            result.append(ch)
                            i += 1
                    elif k < len(text) and text[k] in '{[0123456789tfn-':
                        # Comma then value/object/array — real terminator
                        in_string = False
                        result.append(ch)
                        i += 1
                    else:
                        # Comma then bare text — embedded quote
                        result.append('\\"')
                        i += 1
                else:
                    # Followed by letter/other — embedded quote
                    result.append('\\"')
                    i += 1
        else:
            result.append(ch)
            i += 1

    return ''.join(result)


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

    # Fix 0.5: Embedded quotes (e.g. Italian "contromano" inside strings)
    fixed05 = _fix_embedded_quotes(candidate)
    try:
        json.loads(fixed05)
        applied_fixes.append("embedded_quotes")
        if return_metadata:
            return RepairResult(text=fixed05, applied_fixes=applied_fixes, was_truncated=False)
        return fixed05
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
    combined = _fix_embedded_quotes(candidate)
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
