# tests/test_rag_ingestion.py
"""Tests for Swiss corpus ingestion (mocked — no real HuggingFace or embedder)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from athena.rag.ingestion.swiss import (
    _split_into_articles,
    _split_html_into_articles,
    _split_pdf_into_articles,
    _make_chunk_id,
    _estimate_tokens,
)


class TestSplitPdfIntoArticles:
    def test_splits_articles_double_space(self):
        # PDF format: articles preceded by double-space
        text = "Preamble text.  Art. 1 First article text.  Art. 2 Second article text."
        articles = _split_pdf_into_articles(text)
        assert len(articles) == 2
        assert articles[0]["article_number"] == "Art. 1"
        assert "First article text" in articles[0]["text"]
        assert articles[1]["article_number"] == "Art. 2"

    def test_paragraph_sign(self):
        text = "Preamble.  §  1 Gegenstand  1 Dieses Gesetz regelt...  §  2 Geltungsbereich  1 Dieses Gesetz findet Anwendung auf..."
        articles = _split_pdf_into_articles(text)
        assert len(articles) == 2
        assert "§" in articles[0]["article_number"]

    def test_bis_article(self):
        text = "Preamble.  Art. 1 First article of the code.  Art. 1bis Added article text here.  Art. 2 Second article of the code."
        articles = _split_pdf_into_articles(text)
        assert len(articles) == 3
        assert articles[1]["article_number"] == "Art. 1bis"

    def test_no_articles(self):
        text = "Just a paragraph without any article markers."
        articles = _split_pdf_into_articles(text)
        assert len(articles) == 0

    def test_ignores_cross_references(self):
        # "Art. 46 des Bundesgesetzes" mid-sentence — no double-space before
        text = "gestützt auf Art. 46 des Bundesgesetzes  Art. 1 Real article content here."
        articles = _split_pdf_into_articles(text)
        assert len(articles) == 1
        assert articles[0]["article_number"] == "Art. 1"


class TestSplitHtmlIntoArticles:
    def test_splits_html_articles(self):
        html = """
        <div class="document">
        <div class="article">
            <div class="article_number"><span class="article_symbol">Art.</span> 1</div>
            <div class="article_title"><span class="title_text">Scope</span></div>
            <div class="paragraph"><p>This law regulates...</p></div>
        </div>
        <div class="article">
            <div class="article_number"><span class="article_symbol">Art.</span> 2</div>
            <div class="article_title"><span class="title_text">Definitions</span></div>
            <div class="paragraph"><p>The following definitions apply...</p></div>
        </div>
        </div>
        """
        articles = _split_html_into_articles(html)
        assert len(articles) == 2
        assert articles[0]["article_number"] == "Art. 1"
        assert "This law regulates" in articles[0]["text"]
        assert articles[1]["article_number"] == "Art. 2"

    def test_empty_html(self):
        articles = _split_html_into_articles("")
        assert len(articles) == 0


class TestSplitIntoArticles:
    def test_prefers_html_over_pdf(self):
        html = """
        <div class="article">
            <div class="article_number"><span class="article_symbol">Art.</span> 1</div>
            <div class="paragraph"><p>HTML content</p></div>
        </div>
        """
        pdf = "  Art. 1 PDF content."
        articles = _split_into_articles(html, pdf, "210")
        assert len(articles) == 1
        assert "HTML content" in articles[0]["text"]

    def test_falls_back_to_pdf(self):
        articles = _split_into_articles("", "Preamble.  Art. 1 PDF article text here for real.", "210")
        assert len(articles) == 1
        assert articles[0]["article_number"] == "Art. 1"

    def test_full_fallback(self):
        articles = _split_into_articles("", "No articles here.", "210")
        assert len(articles) == 1
        assert articles[0]["article_number"] == "full"

    def test_empty_returns_empty(self):
        articles = _split_into_articles("", "", "210")
        assert len(articles) == 0


class TestMakeChunkId:
    def test_deterministic(self):
        id1 = _make_chunk_id("210", "Art. 1", "de")
        id2 = _make_chunk_id("210", "Art. 1", "de")
        assert id1 == id2

    def test_different_for_different_input(self):
        id1 = _make_chunk_id("210", "Art. 1", "de")
        id2 = _make_chunk_id("210", "Art. 2", "de")
        assert id1 != id2


class TestEstimateTokens:
    def test_basic(self):
        tokens = _estimate_tokens("one two three four")
        assert tokens == 3  # 4 * 0.75 = 3

    def test_empty(self):
        tokens = _estimate_tokens("")
        assert tokens == 1  # min 1


class TestIngestSwissCorpus:
    @patch("athena.rag.store.upsert_chunks")
    @patch("athena.rag.embedder.embed_dense")
    @patch("datasets.load_dataset")
    def test_ingestion(self, mock_load, mock_embed, mock_upsert):
        # Simulate dataset with 2 laws using actual field names
        mock_ds = [
            {"sr_number": "210", "language": "de", "title": "ZGB",
             "html_content": "", "pdf_content": "Preamble.  Art. 1 First rule of the civil code.  Art. 2 Second rule of the civil code.",
             "version_active_since": "1912-01-01"},
            {"sr_number": "220", "language": "de", "title": "OR",
             "html_content": "", "pdf_content": "Preamble.  Art. 1 Contract law applies to all obligations.  Art. 2 Performance and delivery.",
             "version_active_since": "1912-01-01"},
        ]
        mock_load.return_value = mock_ds
        mock_embed.return_value = np.random.randn(4, 1024).astype(np.float32)
        mock_upsert.return_value = 4

        from athena.rag.ingestion.swiss import ingest_swiss_corpus
        stats = ingest_swiss_corpus(batch_size=10)

        assert stats["status"] == "ok"
        assert stats["laws_processed"] == 2
        assert stats["chunks_created"] == 4

    @patch("athena.rag.store.upsert_chunks")
    @patch("athena.rag.embedder.embed_dense")
    @patch("datasets.load_dataset")
    def test_skips_empty_texts(self, mock_load, mock_embed, mock_upsert):
        mock_ds = [
            {"sr_number": "000", "language": "de", "title": "Empty",
             "html_content": "", "pdf_content": "",
             "version_active_since": None},
        ]
        mock_load.return_value = mock_ds
        mock_embed.return_value = np.empty((0, 1024))

        from athena.rag.ingestion.swiss import ingest_swiss_corpus
        stats = ingest_swiss_corpus()
        assert stats["laws_processed"] == 0
        assert stats["chunks_created"] == 0
