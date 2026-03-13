# tests/test_rag_retriever.py
"""Tests for RAG retriever (mocked — no real embedder or store)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from athena.rag.retriever import retrieve_relevant_norms, _build_queries, _norm_matches_existing


class TestBuildQueries:
    def test_extracts_claims(self):
        seeds = [{"claim": "Art. 1 violated"}, {"claim": "Damages owed"}]
        queries = _build_queries(seeds, {"disputed": [], "undisputed": []})
        assert "Art. 1 violated" in queries
        assert "Damages owed" in queries

    def test_extracts_facts(self):
        facts = {
            "disputed": [{"description": "Contract was breached"}],
            "undisputed": [{"description": "Contract signed 2020"}],
        }
        queries = _build_queries([], facts)
        assert "Contract was breached" in queries
        assert "Contract signed 2020" in queries

    def test_empty_input(self):
        queries = _build_queries([], {"disputed": [], "undisputed": []})
        assert queries == []

    def test_string_undisputed_facts(self):
        facts = {"disputed": [], "undisputed": ["fact1", "fact2"]}
        queries = _build_queries([], facts)
        assert "fact1" in queries


class TestNormMatchesExisting:
    def test_matches_by_sr_number(self):
        chunk = {"sr_number": "210", "article_number": "Art. 1"}
        existing = [{"reference": "SR 210 Art. 1"}]
        assert _norm_matches_existing(chunk, existing) is True

    def test_no_match(self):
        chunk = {"sr_number": "220", "article_number": "Art. 5"}
        existing = [{"reference": "SR 210 Art. 1"}]
        assert _norm_matches_existing(chunk, existing) is False

    def test_empty_sr(self):
        chunk = {"sr_number": "", "article_number": "Art. 1"}
        existing = [{"reference": "SR 210 Art. 1"}]
        assert _norm_matches_existing(chunk, existing) is False


class TestRetrieveRelevantNorms:
    @patch("athena.rag.store.search_norms")
    @patch("athena.rag.embedder.embed_dense")
    def test_basic_retrieval(self, mock_embed, mock_search):
        mock_embed.return_value = np.random.randn(1, 1024).astype(np.float32)
        mock_search.return_value = [
            {"chunk_id": "a", "sr_number": "220", "article_number": "Art. 1",
             "text": "test norm", "token_count": 10, "_distance": 0.1},
        ]

        result = retrieve_relevant_norms(
            seed_arguments=[{"claim": "test claim"}],
            facts={"disputed": [], "undisputed": []},
            existing_legal_texts=[],
            jurisdiction="CH",
        )
        assert len(result) == 1
        assert result[0]["chunk_id"] == "a"

    @patch("athena.rag.store.search_norms")
    @patch("athena.rag.embedder.embed_dense")
    def test_deduplicates(self, mock_embed, mock_search):
        mock_embed.return_value = np.random.randn(2, 1024).astype(np.float32)
        mock_search.return_value = [
            {"chunk_id": "a", "text": "norm", "token_count": 10, "_distance": 0.1},
        ]

        result = retrieve_relevant_norms(
            seed_arguments=[{"claim": "q1"}, {"claim": "q2"}],
            facts={"disputed": [], "undisputed": []},
            existing_legal_texts=[],
            jurisdiction="CH",
        )
        # Same chunk_id from both queries → should be deduplicated
        assert len(result) == 1

    @patch("athena.rag.store.search_norms")
    @patch("athena.rag.embedder.embed_dense")
    def test_token_budget(self, mock_embed, mock_search):
        mock_embed.return_value = np.random.randn(1, 1024).astype(np.float32)
        mock_search.return_value = [
            {"chunk_id": f"c{i}", "text": f"norm {i}", "token_count": 500,
             "_distance": 0.1 * i}
            for i in range(10)
        ]

        result = retrieve_relevant_norms(
            seed_arguments=[{"claim": "test"}],
            facts={"disputed": [], "undisputed": []},
            existing_legal_texts=[],
            jurisdiction="CH",
            token_budget=1500,
        )
        # 1500 / 500 = 3 chunks max
        assert len(result) == 3

    @patch("athena.rag.store.search_norms")
    @patch("athena.rag.embedder.embed_dense")
    def test_filters_existing(self, mock_embed, mock_search):
        mock_embed.return_value = np.random.randn(1, 1024).astype(np.float32)
        mock_search.return_value = [
            {"chunk_id": "a", "sr_number": "210", "article_number": "Art. 1",
             "text": "existing norm", "token_count": 10, "_distance": 0.1},
        ]

        result = retrieve_relevant_norms(
            seed_arguments=[{"claim": "test"}],
            facts={"disputed": [], "undisputed": []},
            existing_legal_texts=[{"reference": "SR 210 Art. 1"}],
            jurisdiction="CH",
        )
        assert len(result) == 0

    def test_empty_input(self):
        result = retrieve_relevant_norms(
            seed_arguments=[],
            facts={"disputed": [], "undisputed": []},
            existing_legal_texts=[],
            jurisdiction="CH",
        )
        assert result == []
