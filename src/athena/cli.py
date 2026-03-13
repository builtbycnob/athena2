# src/athena/cli.py
"""CLI entry point for running ATHENA simulations.

Usage:
    athena run --case CASE_YAML --simulation SIM_YAML --output OUTPUT_DIR
    athena serve --host HOST --port PORT
"""

import argparse
import os
import sys

import yaml

from athena.schemas.simulation import SimulationConfig


def migrate_case_v1(case_data: dict) -> dict:
    """Detect old-format case YAML and convert to N-party format.

    Old format: seed_arguments.appellant/respondent, disputed facts with
    appellant_position/respondent_position.
    New format: seed_arguments.by_party, disputed facts with positions dict.
    """
    # Build role→id mapping once
    party_id_by_role = {}
    for p in case_data.get("parties", []):
        party_id_by_role[p["role"]] = p["id"]

    sa = case_data.get("seed_arguments", {})
    # Detect old format: has 'appellant' key but not 'by_party'
    if "by_party" not in sa and ("appellant" in sa or "respondent" in sa):
        sa = dict(sa)  # copy to avoid mutating caller's dict
        by_party = {}
        if "appellant" in sa:
            pid = party_id_by_role.get("appellant", "appellant")
            by_party[pid] = sa.pop("appellant")
        if "respondent" in sa:
            pid = party_id_by_role.get("respondent", "respondent")
            by_party[pid] = sa.pop("respondent")
        sa["by_party"] = by_party
        case_data["seed_arguments"] = sa

    # Migrate disputed facts: appellant_position/respondent_position → positions dict
    facts = case_data.get("facts", {})
    for i, df in enumerate(facts.get("disputed", [])):
        if "positions" not in df and ("appellant_position" in df or "respondent_position" in df):
            df = dict(df)  # copy to avoid mutating caller's dict
            positions = {}
            if "appellant_position" in df:
                pid = party_id_by_role.get("appellant", "appellant")
                positions[pid] = df.pop("appellant_position")
            if "respondent_position" in df:
                pid = party_id_by_role.get("respondent", "respondent")
                positions[pid] = df.pop("respondent_position")
            df["positions"] = positions
            facts["disputed"][i] = df

    return case_data


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="athena",
        description="ATHENA — Adversarial Tactical Hearing & Equilibrium Navigation Agent",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a Monte Carlo simulation")
    run_parser.add_argument(
        "--case", required=True, help="Path to case YAML file"
    )
    run_parser.add_argument(
        "--simulation", required=True, help="Path to simulation config YAML file"
    )
    run_parser.add_argument(
        "--output", required=True, help="Output directory for results"
    )
    run_parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Concurrent simulation runs (default: $ATHENA_CONCURRENCY or 4)",
    )
    run_parser.add_argument(
        "--kg", action="store_true", default=False,
        help="Enable knowledge graph (requires Neo4j, default: off)",
    )
    run_parser.add_argument(
        "--rag", action="store_true", default=False,
        help="Enable RAG legal corpus retrieval (default: off)",
    )

    # serve subcommand
    serve_parser = sub.add_parser("serve", help="Start the FastAPI API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")

    # kg status subcommand
    sub.add_parser("kg-status", help="Show knowledge graph status (node/edge counts)")

    # ingest-corpus subcommand
    ingest_parser = sub.add_parser("ingest-corpus", help="Ingest legal corpus into RAG vector store")
    ingest_parser.add_argument(
        "--jurisdiction", required=True,
        help="Jurisdiction code (e.g. CH)",
    )
    ingest_parser.add_argument(
        "--source", default="huggingface",
        help="Corpus source (default: huggingface)",
    )

    # validation subcommands
    fetch_parser = sub.add_parser("fetch-cases", help="Fetch validation cases from HuggingFace")
    fetch_parser.add_argument(
        "--legal-area", default="civil_law",
        help="Legal area filter (civil_law, penal_law, public_law, social_law, or 'all')",
    )
    fetch_parser.add_argument("--n-rejection", type=int, default=5, help="Number of rejection cases")
    fetch_parser.add_argument("--n-approval", type=int, default=5, help="Number of approval cases")
    fetch_parser.add_argument("--min-year", type=int, default=2000, help="Minimum year")
    fetch_parser.add_argument("--max-words", type=int, default=2000, help="Maximum words in text")
    fetch_parser.add_argument("--cases-dir", default="cases/validation", help="Output directory for case YAML files")
    fetch_parser.add_argument("--ground-truth-dir", default="ground_truth", help="Output directory for ground truth")
    fetch_parser.add_argument("--no-llm", action="store_true", help="Skip LLM extraction (template-only)")
    fetch_parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    validate_parser = sub.add_parser("validate", help="Score ATHENA results against ground truth")
    validate_parser.add_argument("--results-dir", required=True, help="Directory with ATHENA results")
    validate_parser.add_argument("--ground-truth", default="ground_truth", help="Ground truth directory")
    validate_parser.add_argument("--output", default=None, help="Output path for validation report")

    return parser.parse_args(argv)


def _kg_status() -> None:
    """Show knowledge graph status."""
    try:
        os.environ.setdefault("ATHENA_KG_ENABLED", "1")
        from athena.knowledge.config import health_check
        status = health_check()
        if status["status"] == "ok":
            print(f"[KG] Status: OK")
            print(f"[KG] URI: {status['uri']}")
            print(f"[KG] Database: {status['database']}")
            print(f"[KG] Nodes: {status['node_count']}")
            print(f"[KG] Edges: {status['edge_count']}")
        else:
            print(f"[KG] Status: ERROR — {status['error']}")
    except Exception as e:
        print(f"[KG] Cannot connect to Neo4j: {e}")


def _fetch_cases(args) -> None:
    """Fetch validation cases from HuggingFace and convert to ATHENA YAML."""
    from athena.validation.dataset_fetcher import fetch_swiss_cases
    from athena.validation.case_extractor import extract_and_save

    legal_area = None if args.legal_area == "all" else args.legal_area
    print(f"[VALIDATION] Fetching Swiss cases: {args.n_rejection} rejection + {args.n_approval} approval")
    print(f"[VALIDATION] Filters: legal_area={legal_area}, min_year={args.min_year}, max_words={args.max_words}")

    records = fetch_swiss_cases(
        legal_area=legal_area,
        min_year=args.min_year,
        max_words=args.max_words,
        n_rejection=args.n_rejection,
        n_approval=args.n_approval,
        seed=args.seed,
    )
    print(f"[VALIDATION] Fetched {len(records)} records")

    use_llm = not args.no_llm
    for i, record in enumerate(records):
        label_str = "rejection" if record["label"] == 0 else "approval"
        print(f"[VALIDATION] [{i+1}/{len(records)}] Converting {record['id']} ({label_str})...")
        try:
            yaml_path, gt_path = extract_and_save(
                record, args.cases_dir, args.ground_truth_dir, use_llm=use_llm,
            )
            print(f"  → {yaml_path}")
        except Exception as e:
            print(f"  → FAILED: {e}")

    print(f"[VALIDATION] Done. Cases in {args.cases_dir}/, ground truth in {args.ground_truth_dir}/")


def _validate(args) -> None:
    """Score ATHENA results against ground truth."""
    from athena.validation.scorer import score_results

    print(f"[VALIDATION] Scoring results in {args.results_dir} against {args.ground_truth}")
    report = score_results(args.results_dir, args.ground_truth)

    if report.n == 0:
        print("[VALIDATION] No scored cases found. Check that result directories match ground truth case IDs.")
        return

    ci_low, ci_high = report.accuracy_ci
    print(f"[VALIDATION] Cases scored: {report.n}")
    print(f"[VALIDATION] Accuracy: {report.accuracy:.1%} [{ci_low:.1%}, {ci_high:.1%}]")
    print(f"[VALIDATION] Log Loss: {report.log_loss:.3f}")
    print(f"[VALIDATION] ECE: {report.ece:.3f}")

    errors = report.error_analysis()
    if errors:
        print(f"[VALIDATION] Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e['case_id']}: expected={e['expected']}, predicted={e['predicted']} "
                  f"(p_rej={e['p_rejection']:.2f}, p_ann={e['p_annulment']:.2f})")

    md = report.to_markdown()
    output_path = args.output or os.path.join(args.results_dir, "validation_report.md")
    with open(output_path, "w") as f:
        f.write(md)
    print(f"[VALIDATION] Report saved: {output_path}")


def _ingest_corpus(args) -> None:
    """Ingest legal corpus into RAG vector store."""
    jurisdiction = args.jurisdiction.upper()
    print(f"[RAG] Ingesting corpus for jurisdiction: {jurisdiction}")

    if jurisdiction == "CH":
        from athena.rag.ingestion.swiss import ingest_swiss_corpus
        stats = ingest_swiss_corpus()
        print(f"[RAG] Done: {stats['laws_processed']} laws, {stats['chunks_created']} chunks ingested")
    else:
        print(f"[RAG] Error: no corpus ingestion available for jurisdiction '{jurisdiction}'")
        sys.exit(1)


def _run(args) -> None:
    """Run the pipeline via the extracted API layer."""
    from athena.api.models import PipelineOptions, ProgressEvent
    from athena.api.pipeline import (
        prepare_case_data,
        prepare_sim_config,
        run_pipeline,
        write_pipeline_outputs,
    )

    # --- Load inputs ---
    print(f"[ATHENA] Loading case file: {args.case}")
    with open(args.case) as f:
        case_raw = yaml.safe_load(f)
    case_data = prepare_case_data(case_raw)

    print(f"[ATHENA] Loading simulation config: {args.simulation}")
    with open(args.simulation) as f:
        sim_raw = yaml.safe_load(f)
    sim_config_dict = prepare_sim_config(sim_raw)

    # Print plan summary
    sim_config = SimulationConfig(**sim_config_dict)
    print(f"[ATHENA] Total runs planned: {sim_config.total_runs}")
    party_counts = " x ".join(
        f"{len(profiles)} {pid} profiles"
        for pid, profiles in sim_config.party_profiles.items()
    )
    print(
        f"[ATHENA] Combinations: {len(sim_config.judge_profiles)} judge profiles "
        f"x {party_counts} "
        f"x {sim_config.runs_per_combination} runs"
    )

    # --- Build options ---
    rag_enabled = args.rag or os.environ.get("ATHENA_RAG_ENABLED") == "1"
    options = PipelineOptions(
        concurrency=args.concurrency,
        kg_enabled=args.kg or os.environ.get("ATHENA_KG_ENABLED") == "1",
        rag_enabled=rag_enabled,
    )

    if rag_enabled:
        os.environ["ATHENA_RAG_ENABLED"] = "1"

    # --- Progress callback → print ---
    def _on_progress(event: ProgressEvent) -> None:
        print(f"[ATHENA] [{event.stage}] {event.message}")

    # --- Run ---
    result = run_pipeline(case_data, sim_config_dict, options, _on_progress)

    # --- Write outputs ---
    written = write_pipeline_outputs(result, args.output)
    for path in written:
        print(f"[ATHENA] Saved: {path}")

    # --- Stats ---
    from athena.simulation.orchestrator import _get_concurrency
    stats = result.stats
    print(f"[ATHENA] LLM stats: {stats.get('calls', 0)} calls, "
          f"{stats.get('total_tokens', 0)} tokens, "
          f"{stats.get('total_time', 0):.0f}s total, "
          f"{stats.get('avg_tok_s', 0):.1f} avg tok/s, "
          f"concurrency={_get_concurrency()}")

    print(f"[ATHENA] Done. {len(result.results)} runs, outputs in {args.output}/")

    from athena.agents.llm import langfuse
    langfuse.flush()


def _serve(args) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install athena[api]")
        sys.exit(1)

    from athena.api.app import create_app

    app = create_app()
    print(f"[ATHENA] Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.command == "kg-status":
        _kg_status()
        return

    if args.command == "fetch-cases":
        _fetch_cases(args)
        return

    if args.command == "validate":
        _validate(args)
        return

    if args.command == "serve":
        _serve(args)
        return

    if args.command == "ingest-corpus":
        _ingest_corpus(args)
        return

    if args.command != "run":
        print("Usage: athena run --case CASE.yaml --simulation SIM.yaml --output DIR")
        sys.exit(1)

    _run(args)
