# src/athena/rag/retriever.py
"""Retrieve relevant legal norms for judge context enrichment.

Builds queries from seed arguments + facts, runs hybrid search,
deduplicates, filters out already-present norms, and truncates to token budget.
"""

from __future__ import annotations

import logging

from athena.rag.config import get_rag_config
from athena.rag.store import NormChunk

logger = logging.getLogger("athena.rag.retriever")


def _build_queries(
    seed_arguments: list[dict],
    facts: dict,
) -> list[str]:
    """Build search queries from seed arguments and facts."""
    queries = []

    # Extract claims from seed arguments
    for arg in seed_arguments:
        claim = arg.get("claim", arg.get("text", ""))
        if claim:
            queries.append(claim)

    # Extract fact descriptions
    for fact in facts.get("disputed", []):
        desc = fact.get("description", "")
        if desc:
            queries.append(desc)

    # Undisputed facts can also reference norms
    for fact in facts.get("undisputed", []):
        if isinstance(fact, dict):
            desc = fact.get("description", fact.get("text", ""))
        else:
            desc = str(fact)
        if desc:
            queries.append(desc)

    return queries


def _norm_matches_existing(
    chunk: dict,
    existing_legal_texts: list[dict],
) -> bool:
    """Check if a retrieved norm chunk is already present in the case's legal_texts."""
    sr = chunk.get("sr_number", "")
    art = chunk.get("article_number", "")
    if not sr:
        return False

    for lt in existing_legal_texts:
        lt_ref = lt.get("reference", lt.get("id", ""))
        # Match by SR number + article
        if sr in str(lt_ref):
            if not art or art in str(lt_ref):
                return True
    return False


def retrieve_relevant_norms(
    seed_arguments: list[dict],
    facts: dict,
    existing_legal_texts: list[dict],
    jurisdiction: str,
    language: str = "de",
    limit: int = 10,
    token_budget: int = 2000,
) -> list[dict]:
    """Retrieve relevant norms not already in the case.

    Args:
        seed_arguments: List of seed argument dicts (with 'claim' or 'text' keys).
        facts: Facts dict with 'disputed' and 'undisputed' lists.
        existing_legal_texts: Legal texts already in the case file.
        jurisdiction: Jurisdiction code (e.g. "CH").
        language: Preferred language for results.
        limit: Max results per query.
        token_budget: Max total tokens of retrieved norms.

    Returns:
        List of NormChunk dicts, deduplicated and within token budget.
    """
    from athena.rag.embedder import embed_dense
    from athena.rag.store import search_norms

    queries = _build_queries(seed_arguments, facts)
    if not queries:
        return []

    # Embed all queries
    query_embeddings = embed_dense(queries)

    # Search per query and collect results
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for i, query in enumerate(queries):
        results = search_norms(
            query_embeddings[i],
            jurisdiction,
            limit=limit,
            language=language,
        )
        for r in results:
            cid = r.get("chunk_id", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                candidates.append(r)

    # Filter out norms already in case
    filtered = [
        c for c in candidates
        if not _norm_matches_existing(c, existing_legal_texts)
    ]

    # Sort by search distance (lower is better)
    filtered.sort(key=lambda c: c.get("_distance", float("inf")))

    # Truncate to token budget
    result = []
    total_tokens = 0
    for chunk in filtered:
        tokens = chunk.get("token_count", 0)
        if total_tokens + tokens > token_budget and result:
            break
        result.append(chunk)
        total_tokens += tokens

    logger.info(
        f"RAG: {len(queries)} queries → {len(candidates)} candidates "
        f"→ {len(filtered)} after dedup → {len(result)} within budget "
        f"({total_tokens}/{token_budget} tokens)"
    )
    return result
