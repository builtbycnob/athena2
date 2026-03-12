# src/athena/knowledge/ingestion/result_loader.py
"""Ingest per-run simulation results into the knowledge graph.

Maps structured output (JSON Schema–enforced) directly to graph nodes/edges.
No LLM calls needed — data is already structured.
"""

from athena.knowledge.config import get_session
from athena.knowledge.embedder import embed_text, is_embedder_available


def store_run_result(case_id: str, result: dict) -> dict:
    """Store one simulation run's arguments and evaluations in graph.

    Args:
        case_id: Case identifier for linking.
        result: Single run result dict with run_id, judge_profile,
            appellant_brief, respondent_brief, judge_decision.

    Returns:
        Summary dict with node/edge counts.
    """
    run_id = result["run_id"]
    judge_profile_id = result.get("judge_profile", "unknown")
    party_profile_ids = result.get("party_profiles", {})
    counts = {"nodes": 0, "edges": 0}

    with get_session() as session:
        # 1. SimRunNode
        session.run(
            "MERGE (r:SimRunNode {run_id: $run_id}) "
            "SET r.judge_profile_id = $jp, r.case_id = $case_id",
            run_id=run_id,
            jp=judge_profile_id,
            case_id=case_id,
        )
        # Link to case
        session.run(
            "MATCH (c:CaseNode {case_id: $case_id}), "
            "(r:SimRunNode {run_id: $run_id}) "
            "MERGE (c)-[:HAS_RUN]->(r)",
            case_id=case_id,
            run_id=run_id,
        )
        counts["nodes"] += 1
        counts["edges"] += 1

        # 2. Appellant arguments
        appellant_brief = result.get("appellant_brief", {})
        if appellant_brief:
            filed = appellant_brief.get("filed_brief", appellant_brief)
            arguments = filed.get("arguments", [])
            # Determine appellant party_id
            app_party_id = ""
            for pid, prof_id in party_profile_ids.items():
                # Heuristic: first party is typically appellant
                if not app_party_id:
                    app_party_id = pid

            for arg in arguments:
                arg_id = f"{run_id}__{arg['id']}"
                session.run(
                    "MERGE (a:ArgumentNode {argument_id: $aid}) "
                    "SET a.type = $type, a.claim = $claim, "
                    "a.legal_reasoning = $reasoning, a.run_id = $run_id, "
                    "a.party_id = $party_id, a.original_id = $orig_id",
                    aid=arg_id,
                    type=arg.get("type", ""),
                    claim=arg.get("claim", ""),
                    reasoning=arg.get("legal_reasoning", ""),
                    run_id=run_id,
                    party_id=app_party_id,
                    orig_id=arg["id"],
                )
                counts["nodes"] += 1

                # Embedding
                claim = arg.get("claim", "")
                legal_reasoning = arg.get("legal_reasoning", "")
                if is_embedder_available() and claim:
                    emb = embed_text(f"{claim} {legal_reasoning}")
                    if emb:
                        session.run(
                            "MATCH (a:ArgumentNode {argument_id: $aid}) "
                            "SET a.claim_embedding = $emb",
                            aid=arg_id, emb=emb,
                        )

                # PRODUCED_IN → SimRun
                session.run(
                    "MATCH (a:ArgumentNode {argument_id: $aid}), "
                    "(r:SimRunNode {run_id: $run_id}) "
                    "MERGE (a)-[:PRODUCED_IN]->(r)",
                    aid=arg_id,
                    run_id=run_id,
                )
                counts["edges"] += 1

                # DERIVES_FROM → SeedArgument
                derived_from = arg.get("derived_from")
                if derived_from and arg.get("type") == "derived":
                    session.run(
                        "MATCH (a:ArgumentNode {argument_id: $aid}), "
                        "(sa:SeedArgumentNode {seed_arg_id: $said}) "
                        "MERGE (a)-[:DERIVES_FROM]->(sa)",
                        aid=arg_id,
                        said=derived_from,
                    )
                    counts["edges"] += 1

                # CITES_NORM → LegalText
                for norm_id in arg.get("norm_text_cited", []):
                    session.run(
                        "MATCH (a:ArgumentNode {argument_id: $aid}), "
                        "(l:LegalTextNode {legal_text_id: $lid}) "
                        "MERGE (a)-[:CITES_NORM]->(l)",
                        aid=arg_id,
                        lid=norm_id,
                    )
                    counts["edges"] += 1

                # REFERENCES_FACT → Fact
                for fact_id in arg.get("facts_referenced", []):
                    session.run(
                        "MATCH (a:ArgumentNode {argument_id: $aid}), "
                        "(f:FactNode {fact_id: $fid}) "
                        "MERGE (a)-[:REFERENCES_FACT]->(f)",
                        aid=arg_id,
                        fid=fact_id,
                    )
                    counts["edges"] += 1

                # CITES_EVIDENCE → Evidence
                for ev_id in arg.get("evidence_cited", []):
                    session.run(
                        "MATCH (a:ArgumentNode {argument_id: $aid}), "
                        "(e:EvidenceNode {evidence_id: $eid}) "
                        "MERGE (a)-[:CITES_EVIDENCE]->(e)",
                        aid=arg_id,
                        eid=ev_id,
                    )
                    counts["edges"] += 1

                # ADDRESSES_PRECEDENT
                for prec in arg.get("precedents_addressed", []):
                    session.run(
                        "MATCH (a:ArgumentNode {argument_id: $aid}), "
                        "(pr:PrecedentNode {precedent_id: $pid}) "
                        "MERGE (a)-[r:ADDRESSES_PRECEDENT]->(pr) "
                        "SET r.strategy = $strategy, r.reasoning = $reasoning",
                        aid=arg_id,
                        pid=prec["id"],
                        strategy=prec.get("strategy", ""),
                        reasoning=prec.get("reasoning", ""),
                    )
                    counts["edges"] += 1

        # 3. Respondent responses
        respondent_brief = result.get("respondent_brief", {})
        if respondent_brief:
            filed = respondent_brief.get("filed_brief", respondent_brief)
            responses = filed.get("responses_to_opponent", [])

            for resp in responses:
                to_arg = resp.get("to_argument", "")
                resp_id = f"{run_id}__resp__{to_arg}"
                session.run(
                    "MERGE (resp:ResponseNode {response_id: $rid}) "
                    "SET resp.to_argument = $to_arg, "
                    "resp.counter_strategy = $strategy, "
                    "resp.counter_reasoning = $reasoning, "
                    "resp.run_id = $run_id",
                    rid=resp_id,
                    to_arg=to_arg,
                    strategy=resp.get("counter_strategy", ""),
                    reasoning=resp.get("counter_reasoning", ""),
                    run_id=run_id,
                )
                counts["nodes"] += 1

                # RESPONDS_TO → ArgumentNode (match by original_id within same run)
                session.run(
                    "MATCH (resp:ResponseNode {response_id: $rid}), "
                    "(a:ArgumentNode {run_id: $run_id, original_id: $orig_id}) "
                    "MERGE (resp)-[:RESPONDS_TO]->(a)",
                    rid=resp_id,
                    run_id=run_id,
                    orig_id=to_arg,
                )
                counts["edges"] += 1

                # PRODUCED_IN → SimRun
                session.run(
                    "MATCH (resp:ResponseNode {response_id: $rid}), "
                    "(r:SimRunNode {run_id: $run_id}) "
                    "MERGE (resp)-[:PRODUCED_IN]->(r)",
                    rid=resp_id,
                    run_id=run_id,
                )
                counts["edges"] += 1

        # 4. JudgeDecisionNode
        decision = result.get("judge_decision", {})
        if decision:
            verdict = decision.get("verdict", {})
            qual_correct = verdict.get("qualification_correct", True)
            consequence = None
            if not qual_correct:
                if_inc = verdict.get("if_incorrect", {})
                if if_inc:
                    consequence = if_inc.get("consequence")

            session.run(
                "MERGE (jd:JudgeDecisionNode {run_id: $run_id}) "
                "SET jd.qualification_correct = $qc, "
                "jd.consequence = $cons, "
                "jd.reasoning = $reasoning, "
                "jd.case_id = $case_id",
                run_id=run_id,
                qc=qual_correct,
                cons=consequence,
                reasoning=decision.get("reasoning", ""),
                case_id=case_id,
            )
            counts["nodes"] += 1

            # PRODUCED_IN → SimRun
            session.run(
                "MATCH (jd:JudgeDecisionNode {run_id: $run_id}), "
                "(r:SimRunNode {run_id: $run_id}) "
                "MERGE (jd)-[:PRODUCED_IN]->(r)",
                run_id=run_id,
            )
            counts["edges"] += 1

            # EVALUATES → ArgumentNodes
            for eval_item in decision.get("argument_evaluation", []):
                orig_arg_id = eval_item.get("argument_id", "")
                arg_node_id = f"{run_id}__{orig_arg_id}"
                session.run(
                    "MATCH (jd:JudgeDecisionNode {run_id: $run_id}), "
                    "(a:ArgumentNode {argument_id: $aid}) "
                    "MERGE (jd)-[r:EVALUATES]->(a) "
                    "SET r.persuasiveness = $p, r.determinative = $d, "
                    "r.strengths = $s, r.weaknesses = $w",
                    run_id=run_id,
                    aid=arg_node_id,
                    p=eval_item.get("persuasiveness", 0.0),
                    d=eval_item.get("determinative", False),
                    s=eval_item.get("strengths", ""),
                    w=eval_item.get("weaknesses", ""),
                )
                counts["edges"] += 1

            # FOLLOWS_PRECEDENT
            for prec_id, analysis in decision.get("precedent_analysis", {}).items():
                session.run(
                    "MATCH (jd:JudgeDecisionNode {run_id: $run_id}), "
                    "(pr:PrecedentNode {precedent_id: $pid}) "
                    "MERGE (jd)-[r:FOLLOWS_PRECEDENT]->(pr) "
                    "SET r.followed = $followed, r.distinguished = $distinguished, "
                    "r.reasoning = $reasoning",
                    run_id=run_id,
                    pid=prec_id,
                    followed=analysis.get("followed", False),
                    distinguished=analysis.get("distinguished", False),
                    reasoning=analysis.get("reasoning", ""),
                )
                counts["edges"] += 1

    return counts
