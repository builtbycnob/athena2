# src/athena/knowledge/queries/context_enrichment.py
"""Pre-simulation context enrichment queries.

Provides empirical insights from previous simulation runs to seed
more effective arguments in subsequent runs.
"""

from athena.knowledge.config import get_session


def query_seed_argument_ranking(case_id: str, judge_profile_id: str) -> list[dict]:
    """Rank seed arguments by average persuasiveness with a specific judge profile.

    Returns list of dicts: [{seed_arg_id, claim, avg_persuasiveness, n_evaluations}]
    sorted by persuasiveness descending.
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (sa:SeedArgumentNode)<-[:DERIVES_FROM]-(a:ArgumentNode)
                  <-[eval:EVALUATES]-(jd:JudgeDecisionNode)
                  -[:PRODUCED_IN]->(r:SimRunNode {judge_profile_id: $judge_id, case_id: $case_id})
            RETURN sa.seed_arg_id AS seed_arg_id,
                   sa.claim AS claim,
                   avg(eval.persuasiveness) AS avg_persuasiveness,
                   count(*) AS n_evaluations
            ORDER BY avg_persuasiveness DESC
            """,
            judge_id=judge_profile_id,
            case_id=case_id,
        )
        return [dict(r) for r in result]


def query_best_precedent_strategy(case_id: str, judge_profile_id: str) -> list[dict]:
    """Find most effective precedent strategy per judge profile.

    Returns list of dicts: [{precedent_id, strategy, effectiveness, n}]
    sorted by effectiveness descending.
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (pr:PrecedentNode)<-[addr:ADDRESSES_PRECEDENT]-(a:ArgumentNode)
                  <-[eval:EVALUATES]-(jd:JudgeDecisionNode)
                  -[:PRODUCED_IN]->(r:SimRunNode {judge_profile_id: $judge_id, case_id: $case_id})
            RETURN pr.precedent_id AS precedent_id,
                   addr.strategy AS strategy,
                   avg(eval.persuasiveness) AS effectiveness,
                   count(*) AS n
            ORDER BY effectiveness DESC
            """,
            judge_id=judge_profile_id,
            case_id=case_id,
        )
        return [dict(r) for r in result]


def query_expected_counters(case_id: str, seed_arg_id: str) -> list[dict]:
    """Find expected counter-strategies for a seed argument.

    Returns list of dicts: [{counter_strategy, frequency}]
    sorted by frequency descending.
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (sa:SeedArgumentNode {seed_arg_id: $seed_id})
                  <-[:DERIVES_FROM]-(a:ArgumentNode)
                  <-[:RESPONDS_TO]-(resp:ResponseNode)
            WHERE a.run_id STARTS WITH $case_prefix
            RETURN resp.counter_strategy AS counter_strategy,
                   count(*) AS frequency
            ORDER BY frequency DESC
            """,
            seed_id=seed_arg_id,
            case_prefix="",  # No prefix filter needed — seed_arg_id is unique
        )
        return [dict(r) for r in result]


def get_enrichment_for_run(case_id: str, judge_profile_id: str) -> dict:
    """Gather all KG insights for a simulation run.

    Returns dict suitable for injection into party context:
    {
        "seed_arg_ranking": [...],
        "precedent_strategies": [...],
        "counter_strategies": {seed_arg_id: [...]},
    }
    """
    ranking = query_seed_argument_ranking(case_id, judge_profile_id)
    prec_strategies = query_best_precedent_strategy(case_id, judge_profile_id)

    # Get counters for top-ranked seed arguments
    counter_strategies = {}
    for item in ranking[:5]:  # Top 5
        seed_id = item["seed_arg_id"]
        counters = query_expected_counters(case_id, seed_id)
        if counters:
            counter_strategies[seed_id] = counters

    return {
        "seed_arg_ranking": ranking,
        "precedent_strategies": prec_strategies,
        "counter_strategies": counter_strategies,
    }
