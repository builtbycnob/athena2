# src/athena/knowledge/queries/post_analysis.py
"""Post-simulation analysis queries for memo generation.

Provides cross-judge argument trajectories and determinative argument
identification for the strategic memo.
"""

from athena.knowledge.config import get_session


def query_argument_trajectory(case_id: str) -> list[dict]:
    """Track argument persuasiveness across judge profiles.

    Returns list of dicts: [{seed_arg_id, claim, judge_profile_id, mean_persuasiveness}]
    grouped by seed argument and judge profile.
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (sa:SeedArgumentNode)<-[:DERIVES_FROM]-(a:ArgumentNode)
                  <-[eval:EVALUATES]-(jd:JudgeDecisionNode)
                  -[:PRODUCED_IN]->(r:SimRunNode {case_id: $case_id})
            RETURN sa.seed_arg_id AS seed_arg_id,
                   sa.claim AS claim,
                   r.judge_profile_id AS judge_profile_id,
                   avg(eval.persuasiveness) AS mean_persuasiveness,
                   count(*) AS n_evaluations
            ORDER BY sa.seed_arg_id, r.judge_profile_id
            """,
            case_id=case_id,
        )
        return [dict(r) for r in result]


def query_determinative_arguments(case_id: str) -> list[dict]:
    """Find arguments most frequently rated as determinative.

    Returns list of dicts: [{argument_original_id, claim, times_determinative, total_evaluations}]
    sorted by times_determinative descending.
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (a:ArgumentNode)<-[eval:EVALUATES]-(jd:JudgeDecisionNode)
                  -[:PRODUCED_IN]->(r:SimRunNode {case_id: $case_id})
            WITH a.original_id AS arg_id,
                 collect(DISTINCT a.claim)[0] AS claim,
                 count(*) AS total_evals,
                 sum(CASE WHEN eval.determinative THEN 1 ELSE 0 END) AS times_det
            WHERE times_det > 0
            RETURN arg_id AS argument_id,
                   claim,
                   times_det AS times_determinative,
                   total_evals AS total_evaluations
            ORDER BY times_determinative DESC
            """,
            case_id=case_id,
        )
        return [dict(r) for r in result]


def query_precedent_follow_rates(case_id: str) -> list[dict]:
    """Get precedent follow/distinguish rates across judge profiles.

    Returns list of dicts: [{precedent_id, judge_profile_id, followed_rate, n}]
    """
    with get_session() as session:
        result = session.run(
            """
            MATCH (jd:JudgeDecisionNode)-[fp:FOLLOWS_PRECEDENT]->(pr:PrecedentNode),
                  (jd)-[:PRODUCED_IN]->(r:SimRunNode {case_id: $case_id})
            RETURN pr.precedent_id AS precedent_id,
                   r.judge_profile_id AS judge_profile_id,
                   sum(CASE WHEN fp.followed THEN 1 ELSE 0 END) AS n_followed,
                   sum(CASE WHEN fp.distinguished THEN 1 ELSE 0 END) AS n_distinguished,
                   count(*) AS n_total
            ORDER BY pr.precedent_id, r.judge_profile_id
            """,
            case_id=case_id,
        )
        return [dict(r) for r in result]


def get_post_analysis(case_id: str) -> dict:
    """Gather all post-analysis KG insights for memo generation.

    Returns dict:
    {
        "argument_trajectories": [...],
        "determinative_arguments": [...],
        "precedent_follow_rates": [...],
    }
    """
    return {
        "argument_trajectories": query_argument_trajectory(case_id),
        "determinative_arguments": query_determinative_arguments(case_id),
        "precedent_follow_rates": query_precedent_follow_rates(case_id),
    }
