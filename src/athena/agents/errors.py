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
