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

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

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

    # --- Generate outputs ---
    print("[ATHENA] Generating probability table...")
    table_md = format_probability_table(aggregated)

    print("[ATHENA] Generating decision tree...")
    tree_txt = generate_decision_tree(aggregated)

    print("[ATHENA] Generating strategic memo (requires LLM)...")
    try:
        memo_md = generate_strategic_memo(aggregated, case_data)
    except Exception as e:
        memo_md = f"# Strategic Memo\n\nMemo generation failed: {e}\n"
        print(f"[ATHENA] Warning: memo generation failed ({e}), saved placeholder")

    # --- Save outputs ---
    os.makedirs(args.output, exist_ok=True)

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
