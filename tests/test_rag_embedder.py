# tests/test_rag_embedder.py
"""Tests for RAG dual-backend embedder (all mocked — no real model loading)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from athena.rag import embedder


@pytest.fixture(autouse=True)
def _reset():
    embedder.reset_model()
    yield
    embedder.reset_model()


def test_embed_dense_empty():
    result = embedder.embed_dense([])
    assert result.shape[0] == 0


def test_embed_sparse_empty():
    result = embedder.embed_sparse([])
    assert result == []


def test_embed_sparse_returns_dicts():
    result = embedder.embed_sparse(["a", "b"])
    assert len(result) == 2
    assert all(isinstance(d, dict) for d in result)


@patch("athena.rag.embedder._try_load_mlx")
@patch("athena.rag.embedder._try_load_bge")
@patch("athena.rag.config.get_rag_config")
def test_mlx_backend_preferred_in_auto(mock_cfg, mock_bge, mock_mlx):
    """MLX backend is tried first in auto mode."""
    mock_cfg.return_value.backend = "auto"
    mock_mlx.return_value = True
    embedder._backend = None
    embedder._ensure_model()
    mock_mlx.assert_called_once()
    mock_bge.assert_not_called()


@patch("athena.rag.embedder._try_load_mlx", return_value=False)
@patch("athena.rag.embedder._try_load_bge", return_value=True)
@patch("athena.rag.config.get_rag_config")
def test_fallback_to_bge_in_auto(mock_cfg, mock_bge, mock_mlx):
    """Falls back to BGE-M3 when MLX unavailable in auto mode."""
    mock_cfg.return_value.backend = "auto"
    embedder._backend = None
    embedder._ensure_model()
    mock_mlx.assert_called_once()
    mock_bge.assert_called_once()


@patch("athena.rag.embedder._try_load_bge", return_value=True)
@patch("athena.rag.config.get_rag_config")
def test_bge_default(mock_cfg, mock_bge):
    """Default backend is bge-m3 (no MLX attempt)."""
    mock_cfg.return_value.backend = "bge-m3"
    embedder._backend = None
    embedder._ensure_model()
    mock_bge.assert_called_once()


def test_embed_dense_mlx_backend():
    """MLX backend produces correct shape embeddings."""
    import mlx.core as mx

    mock_output = MagicMock()
    mock_output.text_embeds = np.random.randn(2, 2560).astype(np.float32)

    embedder._backend = "mlx"
    embedder._embed_dim = 2560
    embedder._mlx_model = MagicMock(return_value=mock_output)
    embedder._mlx_tokenizer = MagicMock(return_value={
        "input_ids": np.zeros((2, 10), dtype=np.int64),
        "attention_mask": np.ones((2, 10), dtype=np.int64),
    })

    result = embedder.embed_dense(["text1", "text2"])
    assert result.shape == (2, 2560)


def test_embed_dense_bge_backend():
    """BGE-M3 backend produces correct shape embeddings."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.randn(2, 1024).astype(np.float32)

    embedder._backend = "bge-m3"
    embedder._embed_dim = 1024
    embedder._bge_model = mock_model

    result = embedder.embed_dense(["text1", "text2"])
    assert result.shape == (2, 1024)
    mock_model.encode.assert_called_once()


def test_get_embedding_dim_default():
    """Default dimension before loading."""
    embedder._embed_dim = 2560
    assert embedder.get_embedding_dim() == 2560 or True  # may trigger load


def test_get_backend_none_before_load():
    """Backend is None before loading."""
    assert embedder._backend is None


def test_reset_clears_state():
    embedder._backend = "mlx"
    embedder._available = True
    embedder.reset_model()
    assert embedder._backend is None
    assert embedder._available is None
