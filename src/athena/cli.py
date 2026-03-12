# src/athena/cli.py
"""CLI entry point for running ATHENA simulations.

Usage:
    athena run --case CASE_YAML --simulation SIM_YAML --output OUTPUT_DIR
"""

import argparse
import json
import os
import sys

import yaml

from athena.agents.llm import get_stats
from athena.schemas.simulation import SimulationConfig, migrate_simulation_v1
from athena.simulation.orchestrator import run_monte_carlo


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
from athena.simulation.aggregator import aggregate_results
from athena.output.table import format_probability_table
from athena.output.decision_tree import generate_decision_tree
from athena.output.memo import generate_strategic_memo


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

    # kg status subcommand
    sub.add_parser("kg-status", help="Show knowledge graph status (node/edge counts)")

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


def _init_kg(args) -> bool:
    """Initialize KG if --kg flag set or ATHENA_KG_ENABLED=1. Returns True if enabled."""
    kg_enabled = getattr(args, "kg", False) or os.environ.get("ATHENA_KG_ENABLED") == "1"
    if not kg_enabled:
        return False

    os.environ["ATHENA_KG_ENABLED"] = "1"
    try:
        from athena.knowledge.config import get_driver
        get_driver()
        print("[KG] Knowledge graph connected")
        return True
    except Exception as e:
        print(f"[KG] Warning: knowledge graph unavailable ({e}), continuing without KG")
        os.environ["ATHENA_KG_ENABLED"] = "0"
        return False


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

    if args.command != "run":
        print("Usage: athena run --case CASE.yaml --simulation SIM.yaml --output DIR")
        sys.exit(1)

    # --- Load inputs ---
    print(f"[ATHENA] Loading case file: {args.case}")
    with open(args.case) as f:
        case_raw = yaml.safe_load(f)
    # Unwrap top-level 'case' key if present
    case_data = case_raw.get("case", case_raw)
    # Map YAML field names to schema field names
    if "id" in case_data and "case_id" not in case_data:
        case_data["case_id"] = case_data.pop("id")
    # Promote key_precedents from jurisdiction to top level if missing
    if "key_precedents" not in case_data:
        jur = case_data.get("jurisdiction")
        if isinstance(jur, dict):
            case_data["key_precedents"] = jur.get("key_precedents", [])
    # Migrate v1 case format (hardcoded appellant/respondent) to v2 (N-party)
    case_data = migrate_case_v1(case_data)

    # --- Knowledge Graph init + case ingestion ---
    kg_active = _init_kg(args)
    if kg_active:
        try:
            from athena.knowledge import ingest_case
            counts = ingest_case(case_data)
            print(f"[KG] Case ingested: {counts['nodes']} nodes, {counts['edges']} edges")
        except Exception as e:
            print(f"[KG] Warning: case ingestion failed ({e})")

    print(f"[ATHENA] Loading simulation config: {args.simulation}")
    with open(args.simulation) as f:
        sim_raw = yaml.safe_load(f)

    # YAML has a top-level 'simulation' key wrapper
    sim_data = sim_raw.get("simulation", sim_raw)
    sim_data = migrate_simulation_v1(sim_data)
    sim_config = SimulationConfig(**sim_data)
    sim_config_dict = sim_config.model_dump()

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

    # --- Concurrency ---
    if args.concurrency is not None:
        os.environ["ATHENA_CONCURRENCY"] = str(args.concurrency)

    # --- Run simulations ---
    print("[ATHENA] Starting Monte Carlo simulation...")
    results = run_monte_carlo(case_data, sim_config_dict)
    print(f"[ATHENA] Completed: {len(results)}/{sim_config.total_runs} runs succeeded")

    # --- Aggregate ---
    print("[ATHENA] Aggregating results...")
    aggregated = aggregate_results(results, sim_config.total_runs)

    # --- Game theory analysis ---
    game_analysis = None
    if "stakes" in case_data:
        print("[ATHENA] Running game theory analysis...")
        from athena.game_theory import analyze as gt_analyze
        game_analysis = gt_analyze(aggregated, case_data, results)
    else:
        print("[ATHENA] Skipping game theory analysis (no stakes in case data)")

    # --- KG: store aggregation + game theory ---
    if kg_active:
        try:
            from athena.knowledge import store_aggregation, store_game_theory
            store_aggregation(case_data["case_id"], aggregated)
            if game_analysis:
                store_game_theory(case_data["case_id"], game_analysis)
            print("[KG] Aggregation and game theory stored")
        except Exception as e:
            print(f"[KG] Warning: stats ingestion failed ({e})")

    # --- KG: post-analysis for memo ---
    kg_post = None
    if kg_active:
        try:
            from athena.knowledge import get_post_analysis
            kg_post = get_post_analysis(case_data["case_id"])
            if kg_post:
                print("[KG] Post-analysis retrieved for memo")
        except Exception as e:
            print(f"[KG] Warning: post-analysis query failed ({e})")

    # --- Meta-agents ---
    red_team_output = None
    game_theorist_output = None

    print("[ATHENA] Running red team analysis...")
    try:
        from athena.agents.meta_agents import run_red_team
        red_team_output = run_red_team(
            aggregated, case_data,
            game_analysis=game_analysis, kg_insights=kg_post,
        )
    except Exception as e:
        print(f"[ATHENA] Warning: red team analysis failed ({e})")

    if game_analysis is not None:
        print("[ATHENA] Running game theorist analysis...")
        try:
            from athena.agents.meta_agents import run_game_theorist
            game_theorist_output = run_game_theorist(
                aggregated, case_data, game_analysis,
            )
        except Exception as e:
            print(f"[ATHENA] Warning: game theorist analysis failed ({e})")

    # --- IRAC extraction ---
    irac_output = None
    print("[ATHENA] Running IRAC extraction...")
    try:
        from athena.agents.meta_agents import run_irac_extraction
        irac_output = run_irac_extraction(results, case_data)
    except Exception as e:
        print(f"[ATHENA] Warning: IRAC extraction failed ({e})")

    # KG: store IRAC
    if kg_active and irac_output and irac_output.get("irac_analyses"):
        try:
            from athena.knowledge import store_irac
            store_irac(case_data["case_id"], irac_output)
        except Exception:
            pass

    # --- Generate outputs ---
    print("[ATHENA] Generating probability table...")
    table_md = format_probability_table(aggregated)

    print("[ATHENA] Generating decision tree...")
    tree_txt = generate_decision_tree(aggregated)

    gt_summary_md = None
    if game_analysis is not None:
        print("[ATHENA] Generating game theory summary...")
        from athena.output.game_theory_summary import format_game_theory_summary
        gt_summary_md = format_game_theory_summary(game_analysis)

    print("[ATHENA] Generating strategic memo (requires LLM)...")
    try:
        memo_md = generate_strategic_memo(
            aggregated, case_data, game_analysis=game_analysis, kg_insights=kg_post,
            red_team_output=red_team_output, game_theorist_output=game_theorist_output,
            irac_output=irac_output,
        )
    except Exception as e:
        memo_md = f"# Strategic Memo\n\nMemo generation failed: {e}\n"
        print(f"[ATHENA] Warning: memo generation failed ({e}), saved placeholder")

    # --- Save outputs ---
    os.makedirs(args.output, exist_ok=True)

    if game_analysis is not None:
        gt_json_path = os.path.join(args.output, "game_theory.json")
        with open(gt_json_path, "w") as f:
            json.dump(game_analysis.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"[ATHENA] Saved: {gt_json_path}")

        gt_summary_path = os.path.join(args.output, "game_theory_summary.md")
        with open(gt_summary_path, "w") as f:
            f.write(gt_summary_md)
        print(f"[ATHENA] Saved: {gt_summary_path}")

    if red_team_output:
        rt_path = os.path.join(args.output, "red_team.json")
        with open(rt_path, "w") as f:
            json.dump(red_team_output, f, indent=2, ensure_ascii=False)
        print(f"[ATHENA] Saved: {rt_path}")

    if game_theorist_output:
        gta_path = os.path.join(args.output, "game_theorist_agent.json")
        with open(gta_path, "w") as f:
            json.dump(game_theorist_output, f, indent=2, ensure_ascii=False)
        print(f"[ATHENA] Saved: {gta_path}")

    if irac_output and irac_output.get("irac_analyses"):
        irac_path = os.path.join(args.output, "irac_analysis.json")
        with open(irac_path, "w") as f:
            json.dump(irac_output, f, indent=2, ensure_ascii=False)
        print(f"[ATHENA] Saved: {irac_path}")

    table_path = os.path.join(args.output, "probability_table.md")
    with open(table_path, "w") as f:
        f.write(table_md)
    print(f"[ATHENA] Saved: {table_path}")

    tree_path = os.path.join(args.output, "decision_tree.txt")
    with open(tree_path, "w") as f:
        f.write(tree_txt)
    print(f"[ATHENA] Saved: {tree_path}")

    memo_path = os.path.join(args.output, "strategic_memo.md")
    with open(memo_path, "w") as f:
        f.write(memo_md)
    print(f"[ATHENA] Saved: {memo_path}")

    raw_path = os.path.join(args.output, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[ATHENA] Saved: {raw_path}")

    # --- LLM stats ---
    stats = get_stats()
    from athena.simulation.orchestrator import _get_concurrency
    print(f"[ATHENA] LLM stats: {stats['calls']} calls, "
          f"{stats['total_tokens']} tokens, "
          f"{stats['total_time']:.0f}s total, "
          f"{stats['avg_tok_s']:.1f} avg tok/s, "
          f"concurrency={_get_concurrency()}")

    print(f"[ATHENA] Done. {len(results)} runs, outputs in {args.output}/")

    from athena.agents.llm import langfuse
    langfuse.flush()
