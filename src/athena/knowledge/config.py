# src/athena/knowledge/config.py
"""Neo4j connection management with lazy initialization and health checks.

All configuration via environment variables (all optional, KG off by default):
  ATHENA_KG_ENABLED  — "1" to enable (default: off)
  NEO4J_URI          — default: bolt://localhost:7687
  NEO4J_USER         — default: neo4j
  NEO4J_PASSWORD     — required when KG enabled
  NEO4J_DATABASE     — default: athena
"""

import os
import threading

_driver = None
_driver_lock = threading.Lock()


def is_kg_enabled() -> bool:
    """Check if knowledge graph is enabled via env or CLI flag."""
    return os.environ.get("ATHENA_KG_ENABLED", "") == "1"


def _get_neo4j_config() -> dict:
    """Read Neo4j connection config from environment."""
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", ""),
        "database": os.environ.get("NEO4J_DATABASE", "athena"),
    }


def _ensure_driver():
    """Lazy-init Neo4j driver with double-checked locking."""
    global _driver
    if _driver is not None:
        return _driver

    with _driver_lock:
        if _driver is not None:
            return _driver

        from neo4j import GraphDatabase

        cfg = _get_neo4j_config()
        if not cfg["password"]:
            raise RuntimeError(
                "NEO4J_PASSWORD is required when ATHENA_KG_ENABLED=1"
            )

        _driver = GraphDatabase.driver(
            cfg["uri"],
            auth=(cfg["user"], cfg["password"]),
        )
        # Verify connectivity with timeout
        _driver.verify_connectivity(timeout=5)

        # Run schema migration
        _ensure_schema(_driver, cfg["database"])

        return _driver


def _ensure_schema(driver, database: str) -> None:
    """Create uniqueness constraints on first connect (idempotent)."""
    constraints = [
        ("case_id_unique", "CaseNode", "case_id"),
        ("party_id_unique", "PartyNode", "party_id"),
        ("fact_id_unique", "FactNode", "fact_id"),
        ("evidence_id_unique", "EvidenceNode", "evidence_id"),
        ("legal_text_id_unique", "LegalTextNode", "legal_text_id"),
        ("precedent_id_unique", "PrecedentNode", "precedent_id"),
        ("seed_arg_id_unique", "SeedArgumentNode", "seed_arg_id"),
        ("argument_id_unique", "ArgumentNode", "argument_id"),
        ("sim_run_id_unique", "SimRunNode", "run_id"),
        ("judge_decision_run_unique", "JudgeDecisionNode", "run_id"),
        ("batna_key_unique", "BATNANode", "key"),
        ("settlement_key_unique", "SettlementNode", "key"),
    ]
    with driver.session(database=database) as session:
        for name, label, prop in constraints:
            session.run(
                f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )


def get_session():
    """Get a Neo4j session. Caller must close it."""
    driver = _ensure_driver()
    cfg = _get_neo4j_config()
    return driver.session(database=cfg["database"])


def get_driver():
    """Get the Neo4j driver (lazy-initialized)."""
    return _ensure_driver()


def health_check() -> dict:
    """Check Neo4j connectivity and return status info."""
    try:
        driver = _ensure_driver()
        cfg = _get_neo4j_config()
        with driver.session(database=cfg["database"]) as session:
            result = session.run(
                "MATCH (n) RETURN count(n) AS node_count"
            ).single()
            edge_result = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS edge_count"
            ).single()
        return {
            "status": "ok",
            "uri": cfg["uri"],
            "database": cfg["database"],
            "node_count": result["node_count"],
            "edge_count": edge_result["edge_count"],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def reset_driver() -> None:
    """Close and reset the driver (for testing)."""
    global _driver
    with _driver_lock:
        if _driver is not None:
            _driver.close()
            _driver = None
