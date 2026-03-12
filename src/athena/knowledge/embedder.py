# src/athena/knowledge/embedder.py
"""Local sentence-transformers embedder for semantic search.

Sync-only (no asyncio), lazy-load with double-checked locking.
Graceful degradation: returns None if sentence-transformers not installed.
"""

import logging
import threading

logger = logging.getLogger("athena.knowledge.embedder")

NOMIC_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

_model = None
_model_lock = threading.Lock()
_available: bool | None = None


def is_embedder_available() -> bool:
    """Check if sentence-transformers is installed."""
    global _available
    if _available is not None:
        return _available
    try:
        import sentence_transformers  # noqa: F401
        _available = True
    except ImportError:
        _available = False
    return _available


def _ensure_model():
    """Lazy-load the embedding model with double-checked locking."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(NOMIC_MODEL_ID, trust_remote_code=True)
        logger.info(f"Embedder loaded: {NOMIC_MODEL_ID} (dim={EMBEDDING_DIM})")
        return _model


def _pad_short_text(text: str) -> str:
    """Pad short texts to avoid rotary embedding crash on nomic."""
    if len(text) < 20:
        return f"search query: {text} entity description"
    return text


def embed_text(text: str) -> list[float] | None:
    """Embed a single text. Returns None if unavailable."""
    if not is_embedder_available():
        return None
    model = _ensure_model()
    safe_text = _pad_short_text(text)
    embedding = model.encode(
        [safe_text], normalize_embeddings=True, show_progress_bar=False,
    )
    return embedding[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch of texts. Returns None if unavailable."""
    if not is_embedder_available():
        return None
    if not texts:
        return []
    model = _ensure_model()
    safe_texts = [_pad_short_text(t) for t in texts]
    embeddings = model.encode(
        safe_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    return [emb.tolist() for emb in embeddings]
