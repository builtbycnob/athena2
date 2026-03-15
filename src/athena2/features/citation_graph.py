"""Citation graph construction from Swiss legal decisions.

Builds a directed graph where:
- Nodes = decisions (by decision_id)
- Edges = citations (decision A cites decision B)
- Edge types: FOLLOWS (agrees), DISTINGUISHES (disagrees), CITES (neutral)

Sources:
1. rcds/swiss_citation_extraction NER tags (primary — 127K cases)
2. Regex extraction from considerations text (secondary — 329K cases)

Output: NetworkX DiGraph serialized as GraphML + adjacency Parquet.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── BGE Reference Resolution ──────────────────────────────────────

# Pattern: "BGE 144 III 120" → normalized key "BGE_144_III_120"
BGE_NORMALIZE = re.compile(
    r"\b(BGE|DTF|ATF)\s+(\d{1,3})\s+(I{1,3}|IV|V)\s+(\d+)",
    re.IGNORECASE,
)


def normalize_bge_ref(text: str) -> str | None:
    """Extract and normalize a BGE reference to a canonical key."""
    m = BGE_NORMALIZE.search(text)
    if m:
        return f"BGE_{m.group(2)}_{m.group(3)}_{m.group(4)}"
    return None


# ── Citation Extraction from NER ───────────────────────────────────

def extract_citations_from_ner(
    tokens: list[str],
    labels: list[int],
) -> list[dict[str, str]]:
    """Extract citation spans from IOB-tagged tokens.

    Label scheme: 0=O, 1=B-CITATION, 2=I-CITATION, 3=B-LAW, 4=I-LAW

    Returns list of dicts with 'type' (CITATION or LAW) and 'text'.
    """
    citations = []
    current_type = None
    current_tokens = []

    for token, label in zip(tokens, labels):
        if label == 1:  # B-CITATION
            if current_tokens:
                citations.append({"type": current_type, "text": " ".join(current_tokens)})
            current_type = "CITATION"
            current_tokens = [token]
        elif label == 2 and current_type == "CITATION":  # I-CITATION
            current_tokens.append(token)
        elif label == 3:  # B-LAW
            if current_tokens:
                citations.append({"type": current_type, "text": " ".join(current_tokens)})
            current_type = "LAW"
            current_tokens = [token]
        elif label == 4 and current_type == "LAW":  # I-LAW
            current_tokens.append(token)
        else:  # O
            if current_tokens:
                citations.append({"type": current_type, "text": " ".join(current_tokens)})
                current_type = None
                current_tokens = []

    if current_tokens:
        citations.append({"type": current_type, "text": " ".join(current_tokens)})

    return citations


# ── Graph Construction ─────────────────────────────────────────────

@dataclass
class CitationGraph:
    """Directed citation graph between Swiss Federal Supreme Court decisions."""
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, str]] = field(default_factory=list)
    bge_to_decisions: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def add_decision(self, decision_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a decision node."""
        if decision_id not in self.nodes:
            self.nodes[decision_id] = metadata or {}

    def add_citation(self, source_id: str, target_ref: str, citation_type: str = "CITES") -> None:
        """Add a citation edge from source decision to target reference."""
        self.edges.append({
            "source": source_id,
            "target": target_ref,
            "type": citation_type,
        })

    def build_from_regex(self, rows: list[dict[str, Any]]) -> None:
        """Build graph from regex-extracted BGE citations across all cases.

        Args:
            rows: List of dicts with decision_id, facts, considerations.
        """
        for row in rows:
            decision_id = row.get("decision_id", "")
            self.add_decision(decision_id, {
                "law_area": row.get("law_area", ""),
                "year": row.get("year"),
                "language": row.get("language", ""),
                "label": row.get("label"),
            })

            # Extract BGE citations from considerations
            text = row.get("considerations", "") or ""
            for m in BGE_NORMALIZE.finditer(text):
                bge_key = f"BGE_{m.group(2)}_{m.group(3)}_{m.group(4)}"
                self.add_citation(decision_id, bge_key)
                self.bge_to_decisions[bge_key].append(decision_id)

        logger.info(f"Graph: {self.n_nodes:,} nodes, {self.n_edges:,} edges, "
                     f"{len(self.bge_to_decisions):,} unique BGE references")

    def build_from_ner_dataset(self, ner_data: list[dict[str, Any]]) -> None:
        """Build graph from NER-tagged citation extraction dataset.

        Args:
            ner_data: Rows from rcds/swiss_citation_extraction with
                      'decision_id', 'considerations' (token list), 'NER_labels'.
        """
        for row in ner_data:
            decision_id = row.get("decision_id", "")
            tokens = row.get("considerations", [])
            labels = row.get("NER_labels", [])

            if not tokens or not labels:
                continue

            self.add_decision(decision_id, {
                "law_area": row.get("law_area", ""),
                "year": row.get("year"),
                "language": row.get("language", ""),
            })

            citations = extract_citations_from_ner(tokens, labels)
            for cit in citations:
                if cit["type"] == "CITATION":
                    bge_key = normalize_bge_ref(cit["text"])
                    if bge_key:
                        self.add_citation(decision_id, bge_key)
                        self.bge_to_decisions[bge_key].append(decision_id)

        logger.info(f"NER graph: {self.n_nodes:,} nodes, {self.n_edges:,} edges")

    def compute_statistics(self) -> dict[str, Any]:
        """Compute graph statistics."""
        in_degree: Counter = Counter()
        out_degree: Counter = Counter()

        for edge in self.edges:
            out_degree[edge["source"]] += 1
            in_degree[edge["target"]] += 1

        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_unique_bge": len(self.bge_to_decisions),
            "avg_citations_per_case": self.n_edges / max(self.n_nodes, 1),
            "most_cited_bge": in_degree.most_common(20),
            "most_citing_decisions": out_degree.most_common(10),
            "citation_distribution": {
                "0": sum(1 for n in self.nodes if out_degree.get(n, 0) == 0),
                "1-5": sum(1 for n in self.nodes if 1 <= out_degree.get(n, 0) <= 5),
                "6-10": sum(1 for n in self.nodes if 6 <= out_degree.get(n, 0) <= 10),
                "11-20": sum(1 for n in self.nodes if 11 <= out_degree.get(n, 0) <= 20),
                "20+": sum(1 for n in self.nodes if out_degree.get(n, 0) > 20),
            },
        }

    def to_networkx(self):
        """Convert to NetworkX DiGraph."""
        import networkx as nx

        G = nx.DiGraph()

        for node_id, metadata in self.nodes.items():
            G.add_node(node_id, **metadata)

        for edge in self.edges:
            G.add_edge(edge["source"], edge["target"], type=edge.get("type", "CITES"))

        return G

    def save(self, output_dir: Path) -> None:
        """Save graph as GraphML + adjacency Parquet."""
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save edges as Parquet
        edges_df = pd.DataFrame(self.edges)
        edges_df.to_parquet(output_dir / "citation_edges.parquet", index=False)

        # Save node metadata as Parquet
        nodes_data = [{"decision_id": k, **v} for k, v in self.nodes.items()]
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_parquet(output_dir / "citation_nodes.parquet", index=False)

        # Save statistics
        import json
        stats = self.compute_statistics()
        (output_dir / "citation_stats.json").write_text(json.dumps(stats, indent=2, default=str))

        # Save as GraphML if NetworkX available
        try:
            G = self.to_networkx()
            import networkx as nx
            nx.write_graphml(G, str(output_dir / "citation_graph.graphml"))
            logger.info(f"Saved GraphML with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except ImportError:
            logger.info("NetworkX not available, skipping GraphML export")

        logger.info(f"Citation graph saved to {output_dir}")
