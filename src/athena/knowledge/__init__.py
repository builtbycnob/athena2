# src/athena/knowledge/__init__.py
"""ATHENA Knowledge Graph — public API.

All functions are no-ops when KG is disabled. Import is always safe
(neo4j/graphiti only imported lazily when KG is enabled).

Usage:
    from athena.knowledge import ingest_case, store_run_result, ...
"""

from athena.knowledge.config import is_kg_enabled, health_check


def ingest_case(case_data: dict) -> dict:
    """Load case entities into graph. No-op if KG disabled."""
    if not is_kg_enabled():
        return {"nodes": 0, "edges": 0}
    from athena.knowledge.ingestion.case_loader import ingest_case as _ingest
    return _ingest(case_data)


def store_run_result(case_id: str, result: dict) -> dict:
    """Store one simulation run result. No-op if KG disabled."""
    if not is_kg_enabled():
        return {"nodes": 0, "edges": 0}
    from athena.knowledge.ingestion.result_loader import store_run_result as _store
    return _store(case_id, result)


def store_aggregation(case_id: str, aggregated: dict) -> dict:
    """Store aggregated stats. No-op if KG disabled."""
    if not is_kg_enabled():
        return {"updated_args": 0, "updated_precs": 0}
    from athena.knowledge.ingestion.stats_loader import store_aggregation as _store
    return _store(case_id, aggregated)


def store_game_theory(case_id: str, game_analysis) -> dict:
    """Store game theory analysis. No-op if KG disabled."""
    if not is_kg_enabled():
        return {"nodes": 0, "edges": 0}
    from athena.knowledge.ingestion.stats_loader import store_game_theory as _store
    return _store(case_id, game_analysis)


def get_enrichment(case_id: str, judge_profile_id: str) -> dict | None:
    """Get pre-simulation context enrichment. None if KG disabled."""
    if not is_kg_enabled():
        return None
    from athena.knowledge.queries.context_enrichment import get_enrichment_for_run
    return get_enrichment_for_run(case_id, judge_profile_id)


def get_post_analysis(case_id: str) -> dict | None:
    """Get post-simulation analysis for memo. None if KG disabled."""
    if not is_kg_enabled():
        return None
    from athena.knowledge.queries.post_analysis import get_post_analysis as _get
    return _get(case_id)


def search_arguments(query_text: str, limit: int = 10) -> list[dict]:
    """Semantic search over arguments. Empty list if KG disabled."""
    if not is_kg_enabled():
        return []
    from athena.knowledge.queries.semantic_search import search_similar_arguments
    return search_similar_arguments(query_text, limit)


def store_irac(case_id: str, irac_output: dict) -> dict:
    """Store IRAC analyses. No-op if KG disabled."""
    if not is_kg_enabled():
        return {"nodes": 0, "edges": 0}
    from athena.knowledge.ingestion.stats_loader import store_irac as _store
    return _store(case_id, irac_output)
