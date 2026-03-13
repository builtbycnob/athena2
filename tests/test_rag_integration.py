# tests/test_rag_integration.py
"""Tests for RAG integration in graph.py — verify RAG context injection."""

import os
import pytest
from unittest.mock import patch, MagicMock

from athena.simulation.graph import _inject_rag_context


class TestInjectRagContext:
    def test_noop_when_disabled(self):
        ctx = {}
        case_data = {"jurisdiction": {"country": "CH"}, "facts": {}, "seed_arguments": {},
                     "legal_texts": []}
        with patch.dict(os.environ, {"ATHENA_RAG_ENABLED": ""}, clear=False):
            _inject_rag_context(ctx, case_data)
        assert "rag_legal_texts" not in ctx

    @patch("athena.rag.retrieve_norms")
    def test_injects_when_enabled(self, mock_retrieve):
        mock_retrieve.return_value = [
            {"sr_number": "210", "article_number": "Art. 1",
             "text": "norm text", "section_breadcrumb": "ZGB"},
        ]
        ctx = {}
        case_data = {
            "jurisdiction": {"country": "CH"},
            "facts": {"disputed": [], "undisputed": []},
            "seed_arguments": {"by_party": {"p1": [{"claim": "test"}]}},
            "legal_texts": [],
        }
        with patch.dict(os.environ, {"ATHENA_RAG_ENABLED": "1"}, clear=False):
            _inject_rag_context(ctx, case_data)

        assert "rag_legal_texts" in ctx
        assert len(ctx["rag_legal_texts"]) == 1
        assert ctx["rag_legal_texts"][0]["sr_number"] == "210"

    @patch("athena.rag.retrieve_norms")
    def test_no_injection_when_empty(self, mock_retrieve):
        mock_retrieve.return_value = []
        ctx = {}
        case_data = {
            "jurisdiction": {"country": "CH"},
            "facts": {}, "seed_arguments": {"by_party": {}},
            "legal_texts": [],
        }
        with patch.dict(os.environ, {"ATHENA_RAG_ENABLED": "1"}, clear=False):
            _inject_rag_context(ctx, case_data)

        assert "rag_legal_texts" not in ctx

    def test_defaults_to_it_jurisdiction(self):
        ctx = {}
        case_data = {"facts": {}, "seed_arguments": {"by_party": {}}, "legal_texts": []}
        with patch.dict(os.environ, {"ATHENA_RAG_ENABLED": ""}, clear=False):
            _inject_rag_context(ctx, case_data)
        # No-op, but should not crash
        assert "rag_legal_texts" not in ctx


class TestRagPublicApi:
    def test_retrieve_norms_disabled(self):
        with patch.dict(os.environ, {"ATHENA_RAG_ENABLED": ""}, clear=False):
            from athena.rag import retrieve_norms
            result = retrieve_norms([], {}, [], "CH")
            assert result == []

    def test_is_rag_enabled(self):
        from athena.rag import is_rag_enabled
        with patch.dict(os.environ, {"ATHENA_RAG_ENABLED": "1"}, clear=False):
            assert is_rag_enabled() is True
        with patch.dict(os.environ, {"ATHENA_RAG_ENABLED": ""}, clear=False):
            assert is_rag_enabled() is False
