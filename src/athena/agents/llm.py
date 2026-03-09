# src/athena/agents/llm.py
import json
import re
from mlx_lm import load, generate


_MODEL = None
_TOKENIZER = None
_MODEL_PATH = "mlx-community/Qwen3.5-35B-A3B-4bit"


def _ensure_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        _MODEL, _TOKENIZER = load(_MODEL_PATH)


def _call_model(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call the MLX model and return raw text response."""
    _ensure_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(
        _MODEL,
        _TOKENIZER,
        prompt=prompt,
        temperature=temperature,
        max_tokens=8192,
    )
    return response


def parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown blocks."""
    # Try direct parse first
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def invoke_llm(
    system_prompt: str, user_prompt: str, temperature: float
) -> dict:
    """Invoke LLM and return parsed JSON dict."""
    raw = _call_model(system_prompt, user_prompt, temperature)
    return parse_json_response(raw)
