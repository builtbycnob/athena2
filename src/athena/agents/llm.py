# src/athena/agents/llm.py
"""LLM integration layer with oMLX/MLX backends, JSON repair, and retry."""

import json
import os
import time
from dataclasses import dataclass, field

import httpx
from langfuse import observe, Langfuse

from athena.agents.json_repair import extract_json, RepairResult
from athena.agents.errors import (
    LLMError, JSONTruncatedError, JSONMalformedError,
    NonJSONOutputError, classify_error,
)


langfuse = Langfuse()

_MODEL = None
_TOKENIZER = None
_MODEL_PATH = "nightmedia/Qwen3.5-35B-A3B-Text-qx64-hi-mlx"
_CONTEXT_WINDOW = 262144
_DEFAULT_MAX_TOKENS = 16384
_FAILURE_DIR = "output/failures"

# --- Backend configuration ---
_BACKEND = os.environ.get("ATHENA_BACKEND", "omlx")  # "omlx" | "mlx"
_OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://localhost:8000")
_OMLX_MODEL = os.environ.get("OMLX_MODEL", _MODEL_PATH)
_OMLX_TIMEOUT = 300.0
_OMLX_CLIENT: httpx.Client | None = None

# Lazy-loaded mlx_lm references (set by _ensure_model)
_stream_generate = None
_make_sampler = None

_stats = {
    "calls": 0, "total_tokens": 0, "total_time": 0.0,
    "repairs": 0, "truncations": 0, "retries": 0,
    "repair_types": {},
    "cached_tokens": 0, "ttft_total": 0.0,
}


@dataclass
class GenerationResult:
    """Result of JSON extraction from LLM output."""
    data: dict
    applied_fixes: list[str] = field(default_factory=list)
    was_truncated: bool = False


# --- MLX in-process backend ---

def _ensure_model():
    global _MODEL, _TOKENIZER, _stream_generate, _make_sampler
    if _MODEL is None:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        _stream_generate = stream_generate
        _make_sampler = make_sampler
        print(f"[LLM] Loading model: {_MODEL_PATH}", flush=True)
        t0 = time.time()
        _MODEL, _TOKENIZER = load(_MODEL_PATH)
        print(f"[LLM] Model loaded in {time.time()-t0:.1f}s", flush=True)


def _call_model_mlx(
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

    sampler = _make_sampler(temp=temperature)
    t0 = time.time()

    text = ""
    finish_reason = None
    prompt_tokens = 0
    output_tokens = 0

    for response in _stream_generate(
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


# --- oMLX HTTP backend ---

def _ensure_omlx() -> httpx.Client:
    """Ensure oMLX server is reachable. Returns httpx.Client singleton."""
    global _OMLX_CLIENT
    if _OMLX_CLIENT is not None:
        return _OMLX_CLIENT

    client = httpx.Client(base_url=_OMLX_BASE_URL, timeout=_OMLX_TIMEOUT)
    deadline = time.time() + 30.0
    while True:
        try:
            resp = client.get("/v1/models")
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            print(f"[LLM] oMLX connected: {_OMLX_BASE_URL} — models: {models}", flush=True)
            _OMLX_CLIENT = client
            return client
        except (httpx.ConnectError, httpx.ReadTimeout):
            if time.time() >= deadline:
                client.close()
                raise ConnectionError(
                    f"oMLX server unreachable at {_OMLX_BASE_URL} after 30s"
                )
            time.sleep(2)


def _call_model_omlx(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> tuple[str, str, int, int]:
    """Call oMLX via OpenAI-compatible HTTP API. Returns (text, finish_reason, prompt_tokens, output_tokens)."""
    client = _ensure_omlx()
    payload = {
        "model": _OMLX_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    backoff = [1, 3, 10]
    last_err = None
    t0 = time.time()

    for attempt, delay in enumerate(backoff):
        try:
            resp = client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            break
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            last_err = e
            if attempt < len(backoff) - 1:
                print(f"[LLM] oMLX retry {attempt+1}: {e}", flush=True)
                time.sleep(delay)
            else:
                raise
    else:
        raise last_err  # pragma: no cover

    ttft = time.time() - t0
    body = resp.json()
    choice = body["choices"][0]
    usage = body.get("usage", {})

    text = choice["message"]["content"]
    finish_reason = choice.get("finish_reason", "stop")
    prompt_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)

    elapsed = time.time() - t0
    tok_s = output_tokens / elapsed if elapsed > 0 else 0

    _stats["calls"] += 1
    _stats["total_tokens"] += output_tokens
    _stats["total_time"] += elapsed
    _stats["cached_tokens"] += cached_tokens
    _stats["ttft_total"] += ttft

    if finish_reason == "length":
        _stats["truncations"] += 1

    budget_pct = output_tokens / max_tokens * 100 if max_tokens > 0 else 0
    truncation_flag = " TRUNCATED" if finish_reason == "length" else ""
    budget_warn = f" ⚠ {budget_pct:.0f}% of budget used" if budget_pct > 90 else ""
    cached_flag = f", cached={cached_tokens}" if cached_tokens > 0 else ""

    print(
        f"[LLM] Call #{_stats['calls']}: "
        f"{prompt_tokens} prompt → {output_tokens} output tok, "
        f"{elapsed:.1f}s ({tok_s:.1f} tok/s), "
        f"temp={temperature}{cached_flag}{truncation_flag}{budget_warn}",
        flush=True,
    )

    return text, finish_reason, prompt_tokens, output_tokens


# --- Dispatcher ---

def _call_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> tuple[str, str, int, int]:
    """Dispatch to oMLX or MLX backend. Returns (text, finish_reason, prompt_tokens, output_tokens)."""
    if _BACKEND == "omlx":
        return _call_model_omlx(system_prompt, user_prompt, temperature, max_tokens)
    elif _BACKEND == "mlx":
        return _call_model_mlx(system_prompt, user_prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown ATHENA_BACKEND: {_BACKEND!r}")


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
    avg_ttft = (_stats["ttft_total"] / _stats["calls"]
                if _stats["calls"] > 0 else 0)
    cache_hit_pct = (_stats["cached_tokens"] / _stats["total_tokens"] * 100
                     if _stats["total_tokens"] > 0 else 0)
    return {
        **_stats,
        "avg_tok_s": _stats["total_tokens"] / _stats["total_time"]
        if _stats["total_time"] > 0 else 0,
        "avg_ttft": avg_ttft,
        "cache_hit_pct": cache_hit_pct,
        "backend": _BACKEND,
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
        real_fixes = [f for f in repair_result.applied_fixes if f != "none_succeeded"]
        for fix in real_fixes:
            _stats["repair_types"][fix] = _stats["repair_types"].get(fix, 0) + 1
        if real_fixes:
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


@observe(as_type="generation")
def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> dict:
    """Invoke LLM and return parsed JSON dict.

    Pipeline:
    1. Call model via oMLX or MLX backend
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
        langfuse.update_current_generation(
            model=_MODEL_PATH,
            input={"system": system_prompt[:200], "user": user_prompt[:200]},
            output=result.data,
            usage_details={"input": prompt_tokens, "output": output_tokens},
            metadata={
                "temperature": temperature,
                "finish_reason": finish_reason,
                "applied_fixes": result.applied_fixes,
                "backend": _BACKEND,
            },
        )
        return result.data
    except JSONTruncatedError:
        retry_max = min(max_tokens * 2, _CONTEXT_WINDOW - prompt_tokens)
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
    except LLMError as e:
        langfuse.update_current_generation(
            model=_MODEL_PATH,
            input={"system": system_prompt[:200], "user": user_prompt[:200]},
            output={"error": str(e)},
            usage_details={"input": prompt_tokens, "output": output_tokens},
            metadata={"temperature": temperature, "finish_reason": finish_reason},
            level="ERROR",
        )
        _save_failure_artifact(raw, context=f"finish_reason={finish_reason}")
        raise
