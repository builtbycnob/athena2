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

from athena.schemas.simulation import SimulationConfig
from athena.simulation.orchestrator import run_monte_carlo
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

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.command != "run":
        print("Usage: athena run --case CASE.yaml --simulation SIM.yaml --output DIR")
        sys.exit(1)

    # --- Load inputs ---
    print(f"[ATHENA] Loading case file: {args.case}")
    with open(args.case) as f:
        case_data = yaml.safe_load(f)

    print(f"[ATHENA] Loading simulation config: {args.simulation}")
    with open(args.simulation) as f:
        sim_raw = yaml.safe_load(f)

    sim_config = SimulationConfig(**sim_raw)
    sim_config_dict = sim_config.model_dump()

    print(f"[ATHENA] Total runs planned: {sim_config.total_runs}")
    print(
        f"[ATHENA] Combinations: {len(sim_config.judge_profiles)} judge profiles "
        f"x {len(sim_config.appellant_profiles)} appellant profiles "
        f"x {sim_config.runs_per_combination} runs"
    )

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

    print(f"[ATHENA] Done. {len(results)} runs, outputs in {args.output}/")
