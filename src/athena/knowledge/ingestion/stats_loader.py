# src/athena/knowledge/ingestion/stats_loader.py
"""Ingest aggregated statistics and game theory analysis into the graph.

Updates existing nodes with computed properties (persuasiveness, follow rates)
and creates game theory nodes (BATNA, Settlement, Sensitivity).
"""

from athena.knowledge.config import get_session


def store_aggregation(case_id: str, aggregated: dict) -> dict:
    """Update graph with aggregated stats from Monte Carlo runs.

    Updates SeedArgumentNode and PrecedentNode properties with
    mean_persuasiveness, determinative_rate, followed_rate, etc.

    Returns:
        Summary dict with update counts.
    """
    counts = {"updated_args": 0, "updated_precs": 0}

    with get_session() as session:
        # Update argument effectiveness on SeedArgumentNodes
        # The aggregator keys by argument_id (which maps to seed_arg original ids)
        for arg_id, stats in aggregated.get("argument_effectiveness", {}).items():
            result = session.run(
                "MATCH (sa:SeedArgumentNode {seed_arg_id: $said}) "
                "SET sa.mean_persuasiveness = $mean_p, "
                "sa.std_persuasiveness = $std_p, "
                "sa.determinative_rate = $det_rate "
                "RETURN sa.seed_arg_id AS id",
                said=arg_id,
                mean_p=stats.get("mean_persuasiveness", 0.0),
                std_p=stats.get("std_persuasiveness", 0.0),
                det_rate=stats.get("determinative_rate", 0.0),
            )
            if result.single():
                counts["updated_args"] += 1

        # Update precedent analysis on PrecedentNodes
        for prec_id, stats in aggregated.get("precedent_analysis", {}).items():
            result = session.run(
                "MATCH (pr:PrecedentNode {precedent_id: $pid}) "
                "SET pr.followed_rate = $fr, "
                "pr.distinguished_rate = $dr "
                "RETURN pr.precedent_id AS id",
                pid=prec_id,
                fr=stats.get("followed_rate", 0.0),
                dr=stats.get("distinguished_rate", 0.0),
            )
            if result.single():
                counts["updated_precs"] += 1

        # Store dominated strategies as property on CaseNode
        dominated = aggregated.get("dominated_strategies", [])
        session.run(
            "MATCH (c:CaseNode {case_id: $case_id}) "
            "SET c.dominated_strategies = $dominated, "
            "c.total_runs = $total, c.failed_runs = $failed",
            case_id=case_id,
            dominated=dominated,
            total=aggregated.get("total_runs", 0),
            failed=aggregated.get("failed_runs", 0),
        )

    return counts


def store_game_theory(case_id: str, game_analysis) -> dict:
    """Store game theory analysis nodes (BATNA, Settlement, Sensitivity).

    Args:
        case_id: Case identifier.
        game_analysis: GameTheoryAnalysis instance.

    Returns:
        Summary dict with node counts.
    """
    counts = {"nodes": 0, "edges": 0}

    with get_session() as session:
        # BATNA nodes per party
        for party_id, batna in game_analysis.batna.items():
            key = f"{case_id}__{party_id}"
            session.run(
                "MERGE (b:BATNANode {key: $key}) "
                "SET b.case_id = $case_id, b.party_id = $party_id, "
                "b.expected_value = $ev, "
                "b.expected_value_range_low = $ev_low, "
                "b.expected_value_range_high = $ev_high, "
                "b.best_strategy = $best",
                key=key,
                case_id=case_id,
                party_id=party_id,
                ev=batna.expected_value,
                ev_low=batna.expected_value_range[0],
                ev_high=batna.expected_value_range[1],
                best=batna.best_strategy,
            )
            session.run(
                "MATCH (c:CaseNode {case_id: $case_id}), "
                "(b:BATNANode {key: $key}) "
                "MERGE (c)-[:HAS_BATNA]->(b)",
                case_id=case_id,
                key=key,
            )
            counts["nodes"] += 1
            counts["edges"] += 1

        # Settlement node
        s = game_analysis.settlement
        skey = f"{case_id}__settlement"
        props = {
            "case_id": case_id,
            "settlement_exists": s.settlement_exists,
            "surplus": s.surplus,
        }
        if s.zopa:
            props["zopa_low"] = s.zopa[0]
            props["zopa_high"] = s.zopa[1]
        if s.nash_solution is not None:
            props["nash_solution"] = s.nash_solution

        session.run(
            "MERGE (sn:SettlementNode {key: $key}) SET sn += $props",
            key=skey,
            props=props,
        )
        session.run(
            "MATCH (c:CaseNode {case_id: $case_id}), "
            "(sn:SettlementNode {key: $key}) "
            "MERGE (c)-[:HAS_SETTLEMENT]->(sn)",
            case_id=case_id,
            key=skey,
        )
        counts["nodes"] += 1
        counts["edges"] += 1

        # Sensitivity nodes (top 5 by impact for tornado ranking)
        for sr in game_analysis.sensitivity[:5]:
            sens_key = f"{case_id}__{sr.parameter}"
            session.run(
                "MERGE (sn:SensitivityNode {key: $key}) "
                "SET sn.case_id = $case_id, sn.parameter = $param, "
                "sn.impact = $impact, sn.threshold = $threshold, "
                "sn.base_value = $base",
                key=sens_key,
                case_id=case_id,
                param=sr.parameter,
                impact=sr.impact,
                threshold=sr.threshold,
                base=sr.base_value,
            )
            session.run(
                "MATCH (c:CaseNode {case_id: $case_id}), "
                "(sn:SensitivityNode {key: $key}) "
                "MERGE (c)-[:HAS_SENSITIVITY]->(sn)",
                case_id=case_id,
                key=sens_key,
            )
            counts["nodes"] += 1
            counts["edges"] += 1

    return counts


def store_irac(case_id: str, irac_output: dict) -> dict:
    """Store IRAC analyses as IracNodes linked to SeedArgumentNodes.

    Args:
        case_id: Case identifier.
        irac_output: Dict with "irac_analyses" list from IRAC meta-agent.

    Returns:
        Summary dict with node/edge counts.
    """
    counts = {"nodes": 0, "edges": 0}
    analyses = irac_output.get("irac_analyses", [])
    if not analyses:
        return counts

    with get_session() as session:
        for item in analyses:
            seed_arg_id = item.get("seed_arg_id", "")
            irac_id = f"{case_id}__{seed_arg_id}"

            session.run(
                "MERGE (i:IracNode {irac_id: $irac_id}) "
                "SET i.seed_arg_id = $seed_arg_id, "
                "i.case_id = $case_id, "
                "i.issue = $issue, "
                "i.rule = $rule, "
                "i.application = $application, "
                "i.conclusion = $conclusion",
                irac_id=irac_id,
                seed_arg_id=seed_arg_id,
                case_id=case_id,
                issue=item.get("issue", ""),
                rule=item.get("rule", ""),
                application=item.get("application", ""),
                conclusion=item.get("conclusion", ""),
            )
            counts["nodes"] += 1

            # HAS_IRAC edge: SeedArgumentNode → IracNode
            session.run(
                "MATCH (sa:SeedArgumentNode {seed_arg_id: $seed_arg_id}), "
                "(i:IracNode {irac_id: $irac_id}) "
                "MERGE (sa)-[:HAS_IRAC]->(i)",
                seed_arg_id=seed_arg_id,
                irac_id=irac_id,
            )
            counts["edges"] += 1

    return counts
