#!/usr/bin/env python3
"""ATHENA2 Phase 1: Data Foundation.

Downloads all datasets, runs feature extraction, builds citation graph,
generates comprehensive EDA report.

Usage:
    uv run python scripts/phase1_data_foundation.py              # Full pipeline
    uv run python scripts/phase1_data_foundation.py --stats-only # Just statistics
    uv run python scripts/phase1_data_foundation.py --step ingest # Just ingestion
    uv run python scripts/phase1_data_foundation.py --step features # Just features
    uv run python scripts/phase1_data_foundation.py --step citation # Just citation graph

Requires: pip install athena[validation,worldmodel]
Estimated time: ~30 min for ingestion, ~10 min for regex features, ~2h for citation graph
"""

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("athena2.phase1")


def step_ingest(stats_only: bool = False) -> None:
    """Step 1: Download and process all datasets."""
    from athena2.data.ingestion import (
        generate_eda_report,
        ingest_criticality,
        ingest_primary,
        ingest_secondary,
        load_config,
    )

    config = load_config()
    output_dir = Path(config["paths"]["processed"])

    all_stats = []

    logger.info("=" * 60)
    logger.info("STEP 1: Dataset Ingestion")
    logger.info("=" * 60)

    t0 = time.time()

    logger.info("\n[1/4] Primary dataset (SJP-XL, 329K cases)...")
    stats = ingest_primary(config, output_dir, stats_only)
    all_stats.append(stats)
    print(stats.summary())

    logger.info("\n[2/4] Secondary dataset (SJP, 85K cases)...")
    stats = ingest_secondary(config, output_dir, stats_only)
    all_stats.append(stats)
    print(stats.summary())

    logger.info("\n[3/4] Criticality dataset (139K cases)...")
    stats = ingest_criticality(config, output_dir, stats_only)
    all_stats.append(stats)
    print(stats.summary())

    # Citation extraction requires tokenized data — skip for now if it fails
    try:
        from athena2.data.ingestion import ingest_citation_extraction
        logger.info("\n[4/4] Citation extraction dataset (127K cases)...")
        stats = ingest_citation_extraction(config, output_dir, stats_only)
        all_stats.append(stats)
        print(stats.summary())
    except Exception as e:
        logger.warning(f"Citation extraction skipped: {e}")

    elapsed = time.time() - t0
    logger.info(f"\nIngestion complete in {elapsed/60:.1f} min")

    # Generate EDA report
    report_path = Path(config["paths"]["reports"]) / "eda_report.md"
    generate_eda_report(all_stats, report_path)
    logger.info(f"EDA report: {report_path}")


def step_regex_features() -> None:
    """Step 2: Run regex-based feature extraction on all cases."""
    import pandas as pd
    from athena2.data.ingestion import load_config
    from athena2.features.regex_features import extract_batch

    config = load_config()
    input_path = Path(config["paths"]["processed"]) / "sjp_xl.parquet"
    output_path = Path(config["paths"]["features"]) / "regex_features.parquet"

    logger.info("=" * 60)
    logger.info("STEP 2: Regex Feature Extraction")
    logger.info("=" * 60)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    t0 = time.time()
    rows = df.to_dict(orient="records")

    # Process in chunks for memory efficiency
    chunk_size = 10000
    all_results = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        results = extract_batch(chunk)
        all_results.extend(results)
        logger.info(f"  Processed {min(i + chunk_size, len(rows)):,}/{len(rows):,}")

    elapsed = time.time() - t0
    logger.info(f"Extraction complete in {elapsed:.1f}s ({len(rows)/elapsed:.0f} cases/s)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df = pd.DataFrame(all_results)
    features_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(features_df):,} feature rows to {output_path}")

    # Summary statistics
    logger.info(f"\nFeature Summary:")
    logger.info(f"  Cases with BGE citations: {(features_df['n_bge_citations'] > 0).sum():,} "
                f"({(features_df['n_bge_citations'] > 0).mean():.1%})")
    logger.info(f"  Cases with SR references: {(features_df['n_sr_references'] > 0).sum():,} "
                f"({(features_df['n_sr_references'] > 0).mean():.1%})")
    logger.info(f"  Cases with article refs: {(features_df['n_article_references'] > 0).sum():,} "
                f"({(features_df['n_article_references'] > 0).mean():.1%})")
    logger.info(f"  Avg BGE citations/case: {features_df['n_bge_citations'].mean():.1f}")
    logger.info(f"  Avg unique laws/case: {features_df['n_unique_laws'].mean():.1f}")


def step_citation_graph() -> None:
    """Step 3: Build citation graph from regex-extracted references."""
    import pandas as pd
    from athena2.data.ingestion import load_config
    from athena2.features.citation_graph import CitationGraph

    config = load_config()
    input_path = Path(config["paths"]["processed"]) / "sjp_xl.parquet"
    output_dir = Path(config["paths"]["features"]) / "citation_graph"

    logger.info("=" * 60)
    logger.info("STEP 3: Citation Graph Construction")
    logger.info("=" * 60)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows")

    t0 = time.time()
    rows = df.to_dict(orient="records")

    graph = CitationGraph()
    graph.build_from_regex(rows)

    elapsed = time.time() - t0
    logger.info(f"Graph built in {elapsed:.1f}s")

    # Statistics
    stats = graph.compute_statistics()
    logger.info(f"\nCitation Graph Statistics:")
    logger.info(f"  Nodes: {stats['n_nodes']:,}")
    logger.info(f"  Edges: {stats['n_edges']:,}")
    logger.info(f"  Unique BGE references: {stats['n_unique_bge']:,}")
    logger.info(f"  Avg citations/case: {stats['avg_citations_per_case']:.1f}")
    logger.info(f"\n  Most cited BGE references:")
    for ref, count in stats["most_cited_bge"][:10]:
        logger.info(f"    {ref}: {count:,} citations")

    # Save
    graph.save(output_dir)
    logger.info(f"\nGraph saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ATHENA2 Phase 1: Data Foundation")
    parser.add_argument("--step", choices=["ingest", "features", "citation", "all"],
                        default="all")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    total_start = time.time()

    if args.step in ("ingest", "all"):
        step_ingest(args.stats_only)

    if args.step in ("features", "all") and not args.stats_only:
        step_regex_features()

    if args.step in ("citation", "all") and not args.stats_only:
        step_citation_graph()

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1 complete in {total_elapsed/60:.1f} min")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
