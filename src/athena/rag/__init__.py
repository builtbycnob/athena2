# src/athena/rag/__init__.py
"""ATHENA RAG Legal Corpus — public API.

All functions are no-ops when RAG is disabled. Import is always safe
(lancedb/sentence-transformers only imported lazily when RAG is enabled).

Usage:
    from athena.rag import is_rag_enabled, retrieve_norms, ingest_corpus
"""

from athena.rag.config import is_rag_enabled


def retrieve_norms(
    seed_arguments: list[dict],
    facts: dict,
    existing_legal_texts: list[dict],
    jurisdiction: str,
    language: str = "de",
    limit: int = 10,
    token_budget: int | None = None,
) -> list[dict]:
    """Retrieve relevant norms for judge context. Empty list if RAG disabled."""
    if not is_rag_enabled():
        return []
    from athena.rag.retriever import retrieve_relevant_norms
    from athena.rag.config import get_rag_config
    cfg = get_rag_config()
    budget = token_budget or cfg.token_budget
    return retrieve_relevant_norms(
        seed_arguments, facts, existing_legal_texts,
        jurisdiction, language=language, limit=limit, token_budget=budget,
    )


def ingest_corpus(jurisdiction: str, source: str = "huggingface") -> dict:
    """Ingest a legal corpus. No-op if RAG disabled."""
    if not is_rag_enabled():
        return {"status": "disabled"}
    if jurisdiction.upper() == "CH":
        from athena.rag.ingestion.swiss import ingest_swiss_corpus
        return ingest_swiss_corpus()
    return {"status": "error", "message": f"No corpus for jurisdiction '{jurisdiction}'"}
