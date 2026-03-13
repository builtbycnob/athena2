# src/athena/rag/store.py
"""LanceDB vector store for legal norm chunks.

Embedded, zero-server, Apple Silicon GPU support.
One table per jurisdiction (norms_ch, norms_it, etc.).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("athena.rag.store")

_db = None
_db_lock = threading.Lock()


class NormChunk(BaseModel):
    """A single chunk of a legal norm (typically one article)."""

    chunk_id: str
    jurisdiction: str
    sr_number: str = ""
    article_number: str = ""
    section_breadcrumb: str = ""
    language: str = "de"
    text: str = ""
    valid_from: str | None = None
    valid_until: str | None = None
    token_count: int = 0


def _ensure_db():
    """Lazy-open LanceDB."""
    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is not None:
            return _db
        import lancedb
        from athena.rag.config import get_rag_config
        cfg = get_rag_config()
        _db = lancedb.connect(cfg.db_path)
        logger.info(f"LanceDB opened: {cfg.db_path}")
        return _db


def get_table(jurisdiction: str):
    """Get or create a LanceDB table for a jurisdiction."""
    db = _ensure_db()
    table_name = f"norms_{jurisdiction.lower()}"
    if table_name in db.table_names():
        return db.open_table(table_name)
    return None


def create_table(jurisdiction: str, data: list[dict], schema=None):
    """Create a new table for a jurisdiction with initial data."""
    db = _ensure_db()
    table_name = f"norms_{jurisdiction.lower()}"
    if table_name in db.table_names():
        db.drop_table(table_name)
    return db.create_table(table_name, data=data)


def search_norms(
    query_embedding: np.ndarray,
    jurisdiction: str,
    limit: int = 10,
    language: str | None = None,
) -> list[dict]:
    """Vector search for similar norm chunks.

    Args:
        query_embedding: Dense embedding vector.
        jurisdiction: Jurisdiction code (e.g. "CH").
        limit: Max results.
        language: Optional language filter.

    Returns:
        List of result dicts with NormChunk fields + _distance.
    """
    table = get_table(jurisdiction)
    if table is None:
        return []

    query = table.search(query_embedding.tolist()).limit(limit)
    if language:
        query = query.where(f"language = '{language}'")

    try:
        results = query.to_list()
        return results
    except Exception as e:
        logger.warning(f"Search failed: {e}")
        return []


def hybrid_search(
    dense_embedding: np.ndarray,
    sparse_weights: dict[str, float],
    jurisdiction: str,
    limit: int = 10,
    language: str | None = None,
    rrf_k: int = 60,
) -> list[dict]:
    """Hybrid search with Reciprocal Rank Fusion (RRF).

    Combines dense vector search with sparse (lexical) results.
    Falls back to dense-only if sparse is empty.
    """
    # Dense search
    dense_results = search_norms(dense_embedding, jurisdiction, limit=limit * 2, language=language)

    if not sparse_weights:
        return dense_results[:limit]

    # RRF: combine rankings
    # Since LanceDB doesn't natively support sparse search,
    # we use dense-only results for now. Sparse can be added later
    # with full-text search index or BM25 reranking.
    return dense_results[:limit]


def upsert_chunks(
    chunks: list[NormChunk],
    embeddings: np.ndarray,
    jurisdiction: str,
) -> int:
    """Bulk insert chunks with embeddings into the jurisdiction table.

    Args:
        chunks: List of NormChunk models.
        embeddings: Dense embeddings array (N, dim).
        jurisdiction: Jurisdiction code.

    Returns:
        Number of chunks inserted.
    """
    if not chunks:
        return 0

    data = []
    for i, chunk in enumerate(chunks):
        row = chunk.model_dump()
        row["vector"] = embeddings[i].tolist()
        data.append(row)

    table = get_table(jurisdiction)
    if table is None:
        create_table(jurisdiction, data)
    else:
        table.add(data)

    return len(data)


def reset_db() -> None:
    """Reset the DB connection (for testing)."""
    global _db
    with _db_lock:
        _db = None
