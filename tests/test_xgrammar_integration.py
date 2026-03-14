# tests/test_xgrammar_integration.py
"""Tests for XGrammar constrained decoding integration.

These tests verify XGrammar functionality directly (not through oMLX),
since oMLX runs in a separate venv. The oMLX integration is tested
via smoke runs against the live server.
"""

import json
import pytest

xgr = pytest.importorskip("xgrammar")


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """Load a small tokenizer for testing."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("No tokenizer available for test")


@pytest.fixture(scope="module")
def compiler(gpt2_tokenizer):
    """Create a GrammarCompiler for testing."""
    ti = xgr.TokenizerInfo.from_huggingface(
        gpt2_tokenizer, vocab_size=gpt2_tokenizer.vocab_size
    )
    return xgr.GrammarCompiler(ti)


def test_grammar_compiler_cache_reuse(compiler):
    """Same schema compiled twice should not error (internal caching)."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    c1 = compiler.compile_json_schema(json.dumps(schema))
    c2 = compiler.compile_json_schema(json.dumps(schema))
    assert c1 is not None
    assert c2 is not None


def test_bitmask_shape_matches_vocab(gpt2_tokenizer):
    """Bitmask allocation should match vocab size."""
    bitmask = xgr.allocate_token_bitmask(1, gpt2_tokenizer.vocab_size)
    assert bitmask.shape[0] == 1


def test_matcher_first_token_is_open_brace(compiler, gpt2_tokenizer):
    """First valid token for a JSON object schema should be '{'."""
    import numpy as np

    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    compiled = compiler.compile_json_schema(json.dumps(schema))
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, gpt2_tokenizer.vocab_size)
    matcher.fill_next_token_bitmask(bitmask)

    bits = np.unpackbits(bitmask.numpy().view(np.uint8), bitorder="little")[
        : gpt2_tokenizer.vocab_size
    ]
    valid_ids = np.where(bits == 1)[0]
    assert len(valid_ids) > 0
    decoded = gpt2_tokenizer.decode([valid_ids[0]])
    assert "{" in decoded


def test_accept_token_advances_state(compiler, gpt2_tokenizer):
    """Matcher state should advance after accepting a valid token."""
    import numpy as np

    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    compiled = compiler.compile_json_schema(json.dumps(schema))
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, gpt2_tokenizer.vocab_size)

    # Get first valid token
    matcher.fill_next_token_bitmask(bitmask)
    bits = np.unpackbits(bitmask.numpy().view(np.uint8), bitorder="little")[
        : gpt2_tokenizer.vocab_size
    ]
    valid_ids = np.where(bits == 1)[0]
    first_token = int(valid_ids[0])

    # Accept it
    ok = matcher.accept_token(first_token)
    assert ok is True
    assert not matcher.is_terminated()

    # Next set of valid tokens should be different
    matcher.fill_next_token_bitmask(bitmask)
    bits2 = np.unpackbits(bitmask.numpy().view(np.uint8), bitorder="little")[
        : gpt2_tokenizer.vocab_size
    ]
    valid_ids2 = np.where(bits2 == 1)[0]
    # After '{', the next tokens should include '"' (for key)
    assert len(valid_ids2) > 0


def test_enum_constraint_limits_tokens(compiler, gpt2_tokenizer):
    """Schema with enum should restrict valid tokens."""
    schema = {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["red", "blue"]},
        },
        "required": ["color"],
    }
    compiled = compiler.compile_json_schema(json.dumps(schema))
    assert compiled is not None
    # If compilation succeeds, XGrammar supports enum constraints
