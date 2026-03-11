# src/athena/knowledge/ingestion/case_loader.py
"""Ingest case YAML data into the knowledge graph.

Creates entity nodes (Case, Party, Fact, Evidence, LegalText, Precedent,
SeedArgument) and structural edges. Idempotent via MERGE on unique IDs.
"""

from athena.knowledge.config import get_session
from athena.knowledge.ontology import (
    CaseNode,
    PartyNode,
    FactNode,
    EvidenceNode,
    LegalTextNode,
    PrecedentNode,
    SeedArgumentNode,
)


def ingest_case(case_data: dict) -> dict:
    """Load case entities into graph. Idempotent (upsert on case_id + entity id).

    Returns:
        Summary dict with node/edge counts created.
    """
    case_id = case_data["case_id"]
    counts = {"nodes": 0, "edges": 0}

    with get_session() as session:
        # 1. CaseNode
        case_node = CaseNode.from_case_data(case_data)
        session.run(
            "MERGE (c:CaseNode {case_id: $case_id}) "
            "SET c += $props",
            case_id=case_id,
            props=case_node.model_dump(),
        )
        counts["nodes"] += 1

        # 2. PartyNodes + HAS_PARTY edges
        for party in case_data.get("parties", []):
            pn = PartyNode.from_party(party)
            session.run(
                "MERGE (p:PartyNode {party_id: $party_id}) "
                "SET p += $props",
                party_id=pn.party_id,
                props=pn.model_dump(),
            )
            session.run(
                "MATCH (c:CaseNode {case_id: $case_id}), "
                "(p:PartyNode {party_id: $party_id}) "
                "MERGE (c)-[:HAS_PARTY]->(p)",
                case_id=case_id,
                party_id=pn.party_id,
            )
            counts["nodes"] += 1
            counts["edges"] += 1

        # 3. Undisputed FactNodes + HAS_FACT edges
        facts = case_data.get("facts", {})
        for fact in facts.get("undisputed", []):
            fn = FactNode(
                fact_id=fact["id"],
                description=fact["description"],
                is_disputed=False,
            )
            session.run(
                "MERGE (f:FactNode {fact_id: $fact_id}) "
                "SET f += $props",
                fact_id=fn.fact_id,
                props=fn.model_dump(exclude={"positions"}),
            )
            session.run(
                "MATCH (c:CaseNode {case_id: $case_id}), "
                "(f:FactNode {fact_id: $fact_id}) "
                "MERGE (c)-[:HAS_FACT]->(f)",
                case_id=case_id,
                fact_id=fn.fact_id,
            )
            counts["nodes"] += 1
            counts["edges"] += 1

        # 4. Disputed FactNodes + HAS_FACT + POSITION edges + DEPENDS_ON
        for df in facts.get("disputed", []):
            positions = df.get("positions", {})
            fn = FactNode(
                fact_id=df["id"],
                description=df["description"],
                is_disputed=True,
                positions=positions,
            )
            session.run(
                "MERGE (f:FactNode {fact_id: $fact_id}) "
                "SET f.description = $desc, f.is_disputed = true",
                fact_id=fn.fact_id,
                desc=fn.description,
            )
            session.run(
                "MATCH (c:CaseNode {case_id: $case_id}), "
                "(f:FactNode {fact_id: $fact_id}) "
                "MERGE (c)-[:HAS_FACT]->(f)",
                case_id=case_id,
                fact_id=fn.fact_id,
            )
            counts["nodes"] += 1
            counts["edges"] += 1

            # POSITION edges to parties
            for party_id, position_text in positions.items():
                session.run(
                    "MATCH (f:FactNode {fact_id: $fact_id}), "
                    "(p:PartyNode {party_id: $party_id}) "
                    "MERGE (f)-[r:POSITION]->(p) "
                    "SET r.text = $text",
                    fact_id=fn.fact_id,
                    party_id=party_id,
                    text=position_text,
                )
                counts["edges"] += 1

            # DEPENDS_ON edges to undisputed facts
            for dep_id in df.get("depends_on_facts", []):
                session.run(
                    "MATCH (f1:FactNode {fact_id: $from_id}), "
                    "(f2:FactNode {fact_id: $to_id}) "
                    "MERGE (f1)-[:DEPENDS_ON]->(f2)",
                    from_id=fn.fact_id,
                    to_id=dep_id,
                )
                counts["edges"] += 1

        # 5. EvidenceNodes + SUPPORTS_FACT + PRODUCED_BY edges
        for ev in case_data.get("evidence", []):
            en = EvidenceNode(
                evidence_id=ev["id"],
                type=ev["type"],
                description=ev["description"],
                produced_by=ev["produced_by"],
                admissibility=ev.get("admissibility", ""),
            )
            session.run(
                "MERGE (e:EvidenceNode {evidence_id: $eid}) "
                "SET e += $props",
                eid=en.evidence_id,
                props=en.model_dump(),
            )
            session.run(
                "MATCH (c:CaseNode {case_id: $case_id}), "
                "(e:EvidenceNode {evidence_id: $eid}) "
                "MERGE (c)-[:HAS_EVIDENCE]->(e)",
                case_id=case_id,
                eid=en.evidence_id,
            )
            counts["nodes"] += 1
            counts["edges"] += 1

            # SUPPORTS_FACT
            for fact_id in ev.get("supports_facts", []):
                session.run(
                    "MATCH (e:EvidenceNode {evidence_id: $eid}), "
                    "(f:FactNode {fact_id: $fid}) "
                    "MERGE (e)-[:SUPPORTS_FACT]->(f)",
                    eid=en.evidence_id,
                    fid=fact_id,
                )
                counts["edges"] += 1

            # PRODUCED_BY
            session.run(
                "MATCH (e:EvidenceNode {evidence_id: $eid}), "
                "(p:PartyNode {party_id: $pid}) "
                "MERGE (e)-[:PRODUCED_BY]->(p)",
                eid=en.evidence_id,
                pid=ev["produced_by"],
            )
            counts["edges"] += 1

        # 6. LegalTextNodes
        for lt in case_data.get("legal_texts", []):
            ln = LegalTextNode(
                legal_text_id=lt["id"],
                norm=lt["norm"],
                text=lt["text"],
            )
            session.run(
                "MERGE (l:LegalTextNode {legal_text_id: $lid}) "
                "SET l += $props",
                lid=ln.legal_text_id,
                props=ln.model_dump(),
            )
            session.run(
                "MATCH (c:CaseNode {case_id: $case_id}), "
                "(l:LegalTextNode {legal_text_id: $lid}) "
                "MERGE (c)-[:HAS_LEGAL_TEXT]->(l)",
                case_id=case_id,
                lid=ln.legal_text_id,
            )
            counts["nodes"] += 1
            counts["edges"] += 1

        # 7. PrecedentNodes
        for prec in case_data.get("key_precedents", []):
            pn = PrecedentNode(
                precedent_id=prec["id"],
                citation=prec["citation"],
                holding=prec["holding"],
                weight=prec.get("weight", ""),
            )
            session.run(
                "MERGE (pr:PrecedentNode {precedent_id: $pid}) "
                "SET pr += $props",
                pid=pn.precedent_id,
                props=pn.model_dump(exclude_none=True),
            )
            session.run(
                "MATCH (c:CaseNode {case_id: $case_id}), "
                "(pr:PrecedentNode {precedent_id: $pid}) "
                "MERGE (c)-[:HAS_PRECEDENT]->(pr)",
                case_id=case_id,
                pid=pn.precedent_id,
            )
            counts["nodes"] += 1
            counts["edges"] += 1

        # 8. SeedArgumentNodes
        seed_args = case_data.get("seed_arguments", {})
        by_party = seed_args.get("by_party", {})
        for party_id, args in by_party.items():
            for arg in args:
                san = SeedArgumentNode(
                    seed_arg_id=arg["id"],
                    claim=arg["claim"],
                    direction=arg["direction"],
                    party_id=party_id,
                    references_facts=arg.get("references_facts", []),
                )
                session.run(
                    "MERGE (sa:SeedArgumentNode {seed_arg_id: $said}) "
                    "SET sa += $props",
                    said=san.seed_arg_id,
                    props=san.model_dump(),
                )
                session.run(
                    "MATCH (c:CaseNode {case_id: $case_id}), "
                    "(sa:SeedArgumentNode {seed_arg_id: $said}) "
                    "MERGE (c)-[:HAS_SEED_ARGUMENT]->(sa)",
                    case_id=case_id,
                    said=san.seed_arg_id,
                )
                counts["nodes"] += 1
                counts["edges"] += 1

                # REFERENCES_FACT edges
                for fact_id in arg.get("references_facts", []):
                    session.run(
                        "MATCH (sa:SeedArgumentNode {seed_arg_id: $said}), "
                        "(f:FactNode {fact_id: $fid}) "
                        "MERGE (sa)-[:REFERENCES_FACT]->(f)",
                        said=san.seed_arg_id,
                        fid=fact_id,
                    )
                    counts["edges"] += 1

    return counts
