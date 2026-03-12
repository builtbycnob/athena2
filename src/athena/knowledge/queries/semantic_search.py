# src/athena/knowledge/queries/semantic_search.py
"""Semantic search over the knowledge graph using vector indexes.

Requires Neo4j vector indexes + embedder. Graceful degradation:
returns empty list if either is unavailable.
"""

from athena.knowledge.config import get_session
from athena.knowledge.embedder import embed_text, is_embedder_available


def search_similar_arguments(query_text: str, limit: int = 10) -> list[dict]:
    """Find arguments with semantically similar claims.

    Returns list of dicts with argument_id, claim, legal_reasoning, score.
    """
    if not is_embedder_available():
        return []
    embedding = embed_text(query_text)
    if not embedding:
        return []

    with get_session() as session:
        try:
            result = session.run(
                "CALL db.index.vector.queryNodes("
                "'argument_claim_embedding', $limit, $embedding) "
                "YIELD node, score "
                "RETURN node.argument_id AS argument_id, "
                "node.claim AS claim, "
                "node.legal_reasoning AS legal_reasoning, "
                "score",
                limit=limit, embedding=embedding,
            )
            return [dict(record) for record in result]
        except Exception:
            return []


def search_similar_legal_texts(query_text: str, limit: int = 5) -> list[dict]:
    """Find legal texts semantically similar to query.

    Returns list of dicts with legal_text_id, norm, score.
    """
    if not is_embedder_available():
        return []
    embedding = embed_text(query_text)
    if not embedding:
        return []

    with get_session() as session:
        try:
            result = session.run(
                "CALL db.index.vector.queryNodes("
                "'legal_text_embedding', $limit, $embedding) "
                "YIELD node, score "
                "RETURN node.legal_text_id AS legal_text_id, "
                "node.norm AS norm, "
                "score",
                limit=limit, embedding=embedding,
            )
            return [dict(record) for record in result]
        except Exception:
            return []


def search_similar_seed_arguments(query_text: str, limit: int = 10) -> list[dict]:
    """Find seed arguments with semantically similar claims.

    Returns list of dicts with seed_arg_id, claim, direction, score.
    """
    if not is_embedder_available():
        return []
    embedding = embed_text(query_text)
    if not embedding:
        return []

    with get_session() as session:
        try:
            result = session.run(
                "CALL db.index.vector.queryNodes("
                "'seed_arg_claim_embedding', $limit, $embedding) "
                "YIELD node, score "
                "RETURN node.seed_arg_id AS seed_arg_id, "
                "node.claim AS claim, "
                "node.direction AS direction, "
                "score",
                limit=limit, embedding=embedding,
            )
            return [dict(record) for record in result]
        except Exception:
            return []
