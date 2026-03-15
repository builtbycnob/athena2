#!/usr/bin/env python3
"""ATHENA2 Phase 4: Citation Graph + GAT Integration.

Builds full citation graph, trains GAT, integrates with dynamics MLP.

Usage:
    uv run python scripts/phase5_citation.py                    # Full pipeline
    uv run python scripts/phase5_citation.py --step graph       # Build graph only
    uv run python scripts/phase5_citation.py --step gat         # Train GAT only
    uv run python scripts/phase5_citation.py --step integrate   # Integrate + retrain

Expected improvement: +2-5 points macro F1 (DEXA 2025).

Requires: pip install athena[worldmodel]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("athena2.phase5")


def step_build_graph(data_dir: Path, output_dir: Path) -> dict:
    """Phase 4.1: Build full citation graph from 329K cases."""
    import pandas as pd
    from athena2.features.citation_graph import CitationGraph

    logger.info("=" * 60)
    logger.info("PHASE 4.1: Full Citation Graph Construction")
    logger.info("=" * 60)

    sjp_path = data_dir / "sjp_xl.parquet"
    df = pd.read_parquet(sjp_path)
    logger.info(f"Loaded {len(df):,} cases")

    rows = df.to_dict(orient="records")

    graph = CitationGraph()
    graph.build_from_regex(rows)

    # Compute PageRank via NetworkX
    try:
        import networkx as nx
        G = graph.to_networkx()
        pagerank = nx.pagerank(G, alpha=0.85)
        logger.info(f"PageRank computed for {len(pagerank):,} nodes")
    except ImportError:
        pagerank = {}
        logger.warning("NetworkX not available, skipping PageRank")

    # Cross-reference with Citation Extraction dataset
    citation_dir = data_dir
    for split in ["train", "validation", "test"]:
        ce_path = citation_dir / f"citation_extraction_{split}.parquet"
        if ce_path.exists():
            ce_df = pd.read_parquet(ce_path)
            logger.info(f"Citation extraction {split}: {len(ce_df):,} rows")
            # Build NER graph from this data
            ner_rows = ce_df.to_dict(orient="records")
            graph.build_from_ner_dataset(ner_rows)

    # Add criticality labels
    crit_path = data_dir / "criticality.parquet"
    if crit_path.exists():
        crit_df = pd.read_parquet(crit_path)
        for _, row in crit_df.iterrows():
            did = row.get("decision_id", "")
            if did in graph.nodes:
                graph.nodes[did]["bge_label"] = row.get("bge_label")
                graph.nodes[did]["citation_label"] = row.get("citation_label")
        logger.info(f"Added criticality labels for {len(crit_df):,} cases")

    # Statistics
    stats = graph.compute_statistics()
    logger.info(f"\nCitation Graph Statistics:")
    logger.info(f"  Nodes: {stats['n_nodes']:,}")
    logger.info(f"  Edges: {stats['n_edges']:,}")
    logger.info(f"  Unique BGE: {stats['n_unique_bge']:,}")
    logger.info(f"  Avg citations/case: {stats['avg_citations_per_case']:.1f}")

    # Save
    graph_dir = output_dir / "citation_graph"
    graph.save(graph_dir)

    if pagerank:
        np.save(graph_dir / "pagerank.npy", pagerank)

    return stats


def step_train_gat(graph_dir: Path, output_dir: Path) -> dict:
    """Phase 4.2: Train GAT on citation graph."""
    import torch
    import pandas as pd
    from athena2.models.citation_gat import (
        CitationGAT, build_node_features, build_edge_index,
    )

    logger.info("=" * 60)
    logger.info("PHASE 4.2: GAT Training on Citation Graph")
    logger.info("=" * 60)

    # Load graph
    nodes_df = pd.read_parquet(graph_dir / "citation_nodes.parquet")
    edges_df = pd.read_parquet(graph_dir / "citation_edges.parquet")

    nodes = nodes_df.to_dict(orient="records")
    edges = edges_df.to_dict(orient="records")

    # Load PageRank if available
    pagerank = None
    pr_path = graph_dir / "pagerank.npy"
    if pr_path.exists():
        pagerank = dict(np.load(pr_path, allow_pickle=True).item())

    # Build features and edge index
    x, node_map = build_node_features(nodes, pagerank)
    edge_index = build_edge_index(edges, node_map)

    logger.info(f"Node features: {x.shape}")
    logger.info(f"Edges: {edge_index.shape}")

    # Train GAT (self-supervised: predict label from graph structure)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    gat = CitationGAT(
        node_feature_dim=x.shape[1],
        hidden_dim=16,
        output_dim=64,
        n_heads=4,
        dropout=0.1,
    ).to(device)

    x = x.to(device)
    edge_index = edge_index.to(device)

    # Self-supervised training: link prediction
    optimizer = torch.optim.Adam(gat.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(100):
        gat.train()
        embeddings = gat(x, edge_index)

        # Simple link prediction loss
        # Positive edges: actual citations
        if edge_index.shape[1] > 0:
            src_emb = embeddings[edge_index[0]]
            dst_emb = embeddings[edge_index[1]]
            pos_score = (src_emb * dst_emb).sum(dim=-1)

            # Negative edges: random pairs
            neg_dst = torch.randint(0, len(x), (edge_index.shape[1],), device=device)
            neg_emb = embeddings[neg_dst]
            neg_score = (src_emb * neg_emb).sum(dim=-1)

            loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
            loss += -torch.log(1 - torch.sigmoid(neg_score) + 1e-10).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"  GAT epoch {epoch+1}/100: loss={loss.item():.4f}")

    # Save GAT model
    gat_dir = output_dir / "gat_model"
    gat_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gat.state_dict(), gat_dir / "gat.pt")

    # Extract and save embeddings for all nodes
    gat.eval()
    with torch.no_grad():
        all_embeddings = gat(x, edge_index).cpu().numpy()
    np.save(gat_dir / "node_embeddings.npy", all_embeddings)

    # Save node mapping
    node_map_inv = {v: k for k, v in node_map.items()}
    (gat_dir / "node_map.json").write_text(json.dumps(node_map, indent=2))

    logger.info(f"GAT trained, embeddings: {all_embeddings.shape}")

    # Semantic coherence check
    # Find nearest neighbors for a sample node
    from sklearn.metrics.pairwise import cosine_similarity
    sample_idx = 0
    sims = cosine_similarity(all_embeddings[sample_idx:sample_idx+1], all_embeddings)[0]
    top_5 = np.argsort(sims)[-6:-1][::-1]
    logger.info(f"\nNearest neighbors for node {node_map_inv.get(sample_idx, 'unknown')}:")
    for idx in top_5:
        logger.info(f"  {node_map_inv.get(idx, 'unknown')}: similarity={sims[idx]:.4f}")

    return {"n_nodes": len(all_embeddings), "embedding_dim": all_embeddings.shape[1]}


def step_integrate(gat_dir: Path, model_dir: Path, output_dir: Path) -> dict:
    """Phase 4.3: Integrate GAT features with dynamics MLP and retrain."""
    logger.info("=" * 60)
    logger.info("PHASE 4.3: GAT Integration + Dynamics MLP Retrain")
    logger.info("=" * 60)

    # Load GAT embeddings
    embeddings = np.load(gat_dir / "node_embeddings.npy")
    node_map = json.loads((gat_dir / "node_map.json").read_text())

    logger.info(f"GAT embeddings: {embeddings.shape}")
    logger.info(f"Node map: {len(node_map):,} nodes")

    # The full integration would:
    # 1. Load the trained model from Phase 2
    # 2. Add GAT features to the dynamics MLP input
    # 3. Retrain the dynamics MLP (freeze encoder + feature heads)
    # 4. Evaluate improvement

    logger.info("Integration requires Phase 2 trained model + GAT embeddings")
    logger.info("Run scripts/phase3_multitask.py first with --gat flag")

    return {"status": "ready_for_integration", "gat_dim": embeddings.shape[1]}


def main():
    parser = argparse.ArgumentParser(description="ATHENA2 Phase 4: Citation Graph + GAT")
    parser.add_argument("--step", choices=["graph", "gat", "integrate", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/phase4"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/phase2/best_model"))
    args = parser.parse_args()

    t0 = time.time()
    results = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.step in ("graph", "all"):
        results["graph"] = step_build_graph(args.data_dir, args.output_dir)

    if args.step in ("gat", "all"):
        graph_dir = args.output_dir / "citation_graph"
        results["gat"] = step_train_gat(graph_dir, args.output_dir)

    if args.step in ("integrate", "all"):
        gat_dir = args.output_dir / "gat_model"
        results["integrate"] = step_integrate(gat_dir, args.model_dir, args.output_dir)

    elapsed = time.time() - t0
    logger.info(f"\nPhase 4 complete in {elapsed/60:.1f} min")
    (args.output_dir / "phase4_results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
