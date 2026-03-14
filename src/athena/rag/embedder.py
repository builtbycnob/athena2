# src/athena/rag/embedder.py
"""RAG embedder — dual-backend: Qwen3-Embedding MLX (primary) + BGE-M3 (fallback).

MLX backend: mlx-embeddings + Qwen3-Embedding-4B 4-bit DWQ (~2.1GB, ~66 text/s on M3 Ultra)
Fallback: sentence-transformers + BGE-M3 (568M, ~120 text/s but MPS overflow on long seqs)

Same dual-backend pattern as JARVIS EmbeddingService.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger("athena.rag.embedder")

_backend: str | None = None  # "mlx" or "bge-m3"
_mlx_model: Any = None
_mlx_tokenizer: Any = None
_bge_model: Any = None
_lock = threading.Lock()
_available: bool | None = None
_embed_dim: int = 1024  # updated at load time


def is_embedder_available() -> bool:
    """Check if any embedding backend is available."""
    global _available
    if _available is not None:
        return _available
    try:
        _ensure_model()
        _available = _backend is not None
    except Exception:
        _available = False
    return _available


def _try_load_mlx() -> bool:
    """Try loading Qwen3-Embedding via mlx-embeddings."""
    global _mlx_model, _mlx_tokenizer, _backend, _embed_dim
    try:
        from mlx_embeddings.utils import load as mlx_load
        from athena.rag.config import get_rag_config

        cfg = get_rag_config()
        model, tokenizer = mlx_load(cfg.mlx_model_path)
        _mlx_model = model
        _mlx_tokenizer = tokenizer._tokenizer  # unwrap to HF tokenizer
        _backend = "mlx"
        _embed_dim = cfg.embed_dim
        logger.info("RAG embedder loaded via MLX: %s (dim=%d)", cfg.mlx_model_path, _embed_dim)
        return True
    except Exception as e:
        logger.debug("MLX embedding backend unavailable: %s", e)
        return False


def _try_load_bge() -> bool:
    """Fallback: load BGE-M3 via sentence-transformers."""
    global _bge_model, _backend, _embed_dim
    try:
        from sentence_transformers import SentenceTransformer

        _bge_model = SentenceTransformer("BAAI/bge-m3")
        _bge_model.max_seq_length = 512  # avoid MPS INT_MAX overflow
        _backend = "bge-m3"
        _embed_dim = 1024
        logger.info("RAG embedder loaded via sentence-transformers: BAAI/bge-m3 (dim=1024)")
        return True
    except Exception as e:
        logger.warning("BGE-M3 fallback failed: %s", e)
        return False


def _ensure_model():
    """Lazy-load with double-checked locking. MLX first, BGE-M3 fallback."""
    global _backend
    if _backend is not None:
        return
    with _lock:
        if _backend is not None:
            return

        # Disable tokenizers parallelism to avoid deadlocks when embedder
        # is loaded/used inside ThreadPoolExecutor worker threads.
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from athena.rag.config import get_rag_config
        cfg = get_rag_config()

        if cfg.backend in ("auto", "mlx"):
            if _try_load_mlx():
                return
        if cfg.backend in ("auto", "bge-m3"):
            if _try_load_bge():
                return

        logger.error("No embedding backend available")


def get_embedding_dim() -> int:
    """Return the active embedding dimension."""
    _ensure_model()
    return _embed_dim


def get_backend() -> str | None:
    """Return the active backend name."""
    _ensure_model()
    return _backend


def embed_dense(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed texts into dense vectors. Returns (N, dim) array.

    Dispatches to active backend (MLX or BGE-M3).
    """
    if not texts:
        return np.empty((0, _embed_dim), dtype=np.float32)
    _ensure_model()

    if _backend == "mlx":
        return _embed_mlx(texts, batch_size)
    elif _backend == "bge-m3":
        return _embed_bge(texts, batch_size)
    else:
        return np.empty((0, _embed_dim), dtype=np.float32)


def _embed_mlx(texts: list[str], batch_size: int) -> np.ndarray:
    """MLX backend: tokenize → forward → pooled embeddings via mlx-embeddings."""
    import mlx.core as mx

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch = _mlx_tokenizer(
            batch_texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = mx.array(batch["input_ids"])
        attention_mask = mx.array(batch["attention_mask"])

        output = _mlx_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mx.array(output.text_embeds, dtype=mx.float32)
        mx.eval(embeddings)
        all_embeddings.append(np.array(embeddings))

    return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, _embed_dim), dtype=np.float32)


def _embed_bge(texts: list[str], batch_size: int) -> np.ndarray:
    """BGE-M3 backend via sentence-transformers."""
    embeddings = _bge_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_sparse(texts: list[str]) -> list[dict[str, float]]:
    """Compute sparse lexical weights. Falls back to empty dicts."""
    if not texts:
        return []
    return [{} for _ in texts]


def reset_model() -> None:
    """Reset all state (for testing)."""
    global _backend, _mlx_model, _mlx_tokenizer, _bge_model, _available, _embed_dim
    with _lock:
        _backend = None
        _mlx_model = None
        _mlx_tokenizer = None
        _bge_model = None
        _available = None
        _embed_dim = 1024
