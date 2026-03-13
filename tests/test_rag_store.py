# tests/test_rag_store.py
"""Tests for RAG LanceDB vector store (mocked — no real LanceDB)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from athena.rag.store import NormChunk, search_norms, upsert_chunks, hybrid_search, reset_db


@pytest.fixture(autouse=True)
def _reset():
    reset_db()
    yield
    reset_db()


class TestNormChunk:
    def test_valid_chunk(self):
        chunk = NormChunk(
            chunk_id="abc123",
            jurisdiction="CH",
            sr_number="210",
            article_number="Art. 1",
            section_breadcrumb="ZGB > Einleitung",
            language="de",
            text="Das Recht findet auf alle Rechtsfragen Anwendung.",
            token_count=8,
        )
        assert chunk.chunk_id == "abc123"
        assert chunk.jurisdiction == "CH"

    def test_minimal_chunk(self):
        chunk = NormChunk(chunk_id="x", jurisdiction="CH")
        assert chunk.text == ""
        assert chunk.token_count == 0

    def test_temporal_fields(self):
        chunk = NormChunk(
            chunk_id="t1", jurisdiction="CH",
            valid_from="2000-01-01", valid_until="2025-12-31",
        )
        assert chunk.valid_from == "2000-01-01"
        assert chunk.valid_until == "2025-12-31"


class TestSearchNorms:
    @patch("athena.rag.store.get_table")
    def test_search_returns_results(self, mock_get_table):
        mock_table = MagicMock()
        mock_query = MagicMock()
        mock_query.limit.return_value = mock_query
        mock_query.to_list.return_value = [
            {"chunk_id": "a", "text": "Article 1", "_distance": 0.1},
        ]
        mock_table.search.return_value = mock_query
        mock_get_table.return_value = mock_table

        results = search_norms(np.zeros(1024), "CH", limit=5)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "a"

    @patch("athena.rag.store.get_table")
    def test_search_no_table(self, mock_get_table):
        mock_get_table.return_value = None
        results = search_norms(np.zeros(1024), "CH")
        assert results == []

    @patch("athena.rag.store.get_table")
    def test_search_with_language(self, mock_get_table):
        mock_table = MagicMock()
        mock_query = MagicMock()
        mock_query.limit.return_value = mock_query
        mock_query.where.return_value = mock_query
        mock_query.to_list.return_value = []
        mock_table.search.return_value = mock_query
        mock_get_table.return_value = mock_table

        search_norms(np.zeros(1024), "CH", language="de")
        mock_query.where.assert_called_once_with("language = 'de'")


class TestHybridSearch:
    @patch("athena.rag.store.search_norms")
    def test_falls_back_to_dense_only(self, mock_search):
        mock_search.return_value = [{"chunk_id": "a", "_distance": 0.1}]
        results = hybrid_search(np.zeros(1024), {}, "CH", limit=5)
        assert len(results) == 1


class TestUpsertChunks:
    @patch("athena.rag.store.get_table")
    @patch("athena.rag.store.create_table")
    def test_upsert_creates_table(self, mock_create, mock_get):
        mock_get.return_value = None
        chunks = [NormChunk(chunk_id="a", jurisdiction="CH", text="test")]
        embeddings = np.random.randn(1, 1024).astype(np.float32)

        count = upsert_chunks(chunks, embeddings, "CH")
        assert count == 1
        mock_create.assert_called_once()

    @patch("athena.rag.store.get_table")
    def test_upsert_adds_to_existing(self, mock_get):
        mock_table = MagicMock()
        mock_get.return_value = mock_table
        chunks = [
            NormChunk(chunk_id="a", jurisdiction="CH", text="test1"),
            NormChunk(chunk_id="b", jurisdiction="CH", text="test2"),
        ]
        embeddings = np.random.randn(2, 1024).astype(np.float32)

        count = upsert_chunks(chunks, embeddings, "CH")
        assert count == 2
        mock_table.add.assert_called_once()

    def test_upsert_empty(self):
        assert upsert_chunks([], np.empty((0, 1024)), "CH") == 0
