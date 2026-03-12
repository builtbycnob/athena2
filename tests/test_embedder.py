# tests/test_embedder.py
"""Tests for the local embedder (all mocked, no model download)."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestEmbedderAvailability:
    @patch.dict("sys.modules", {"sentence_transformers": None})
    def test_unavailable_when_no_package(self):
        import athena.knowledge.embedder as emb
        emb._available = None  # reset cached state
        assert emb.is_embedder_available() is False
        emb._available = None  # cleanup

    def test_embed_text_returns_none_when_unavailable(self):
        import athena.knowledge.embedder as emb
        old = emb._available
        emb._available = False
        result = emb.embed_text("test")
        assert result is None
        emb._available = old

    def test_embed_texts_returns_none_when_unavailable(self):
        import athena.knowledge.embedder as emb
        old = emb._available
        emb._available = False
        result = emb.embed_texts(["test"])
        assert result is None
        emb._available = old


class TestEmbedderWithMock:
    def _mock_model(self):
        model = MagicMock()
        model.encode.return_value = np.random.randn(1, 768).astype(np.float32)
        return model

    @patch("athena.knowledge.embedder._ensure_model")
    @patch("athena.knowledge.embedder.is_embedder_available", return_value=True)
    def test_embed_text_returns_vector(self, mock_avail, mock_ensure):
        from athena.knowledge.embedder import embed_text
        model = self._mock_model()
        mock_ensure.return_value = model
        result = embed_text("test legal argument about article 143")
        assert result is not None
        assert len(result) == 768
        assert isinstance(result[0], float)

    @patch("athena.knowledge.embedder._ensure_model")
    @patch("athena.knowledge.embedder.is_embedder_available", return_value=True)
    def test_embed_texts_batch(self, mock_avail, mock_ensure):
        from athena.knowledge.embedder import embed_texts
        model = self._mock_model()
        model.encode.return_value = np.random.randn(3, 768).astype(np.float32)
        mock_ensure.return_value = model
        result = embed_texts(["a", "b", "c"])
        assert result is not None
        assert len(result) == 3
        assert len(result[0]) == 768

    @patch("athena.knowledge.embedder._ensure_model")
    @patch("athena.knowledge.embedder.is_embedder_available", return_value=True)
    def test_short_text_padding(self, mock_avail, mock_ensure):
        from athena.knowledge.embedder import embed_text
        model = self._mock_model()
        mock_ensure.return_value = model
        embed_text("AI")  # short text
        call_args = model.encode.call_args[0][0]
        # Should be padded
        assert len(call_args[0]) > 20

    def test_embedding_dimension(self):
        from athena.knowledge.embedder import EMBEDDING_DIM
        assert EMBEDDING_DIM == 768

    @patch("athena.knowledge.embedder.is_embedder_available", return_value=True)
    def test_embed_texts_empty_list(self, mock_avail):
        from athena.knowledge.embedder import embed_texts
        result = embed_texts([])
        assert result == []
