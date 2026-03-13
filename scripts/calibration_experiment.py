#!/usr/bin/env python3
"""Calibration experiment: isolate CH judge bias variables.

Runs controlled experiments varying temperature, minItems, enum ordering,
and measures their effect on lower_court_correct distribution and severity
distribution under constrained decoding.

Usage:
    python scripts/calibration_experiment.py \
        --case cases/validation/ch-247.yaml \
        --runs-per-config 5 \
        [--configs A,B,C,D,E,F]
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from athena.schemas.structured_output import JUDGE_CH_SCHEMA
from athena.simulation.graph import (
    build_bilateral_phases,
    build_graph_from_phases,
)


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

def _baseline_schema() -> dict:
    """Return a deep copy of the current JUDGE_CH_SCHEMA."""
    return copy.deepcopy(JUDGE_CH_SCHEMA)


def _schema_min_items_zero(schema: dict) -> dict:
    """Set minItems=0 on identified_errors (allow zero errors)."""
    schema["properties"]["verdict"]["properties"]["identified_errors"]["minItems"] = 0
    return schema


def _schema_reorder_enums(schema: dict) -> dict:
    """Reorder severity enum: none first, decisive last."""
    errors_item = schema["properties"]["verdict"]["properties"]["identified_errors"]["items"]
    errors_item["properties"]["severity"]["enum"] = ["none", "minor", "significant", "decisive"]
    errors_item["properties"]["error_type"]["enum"] = [
        "none_found", "procedural", "fact_finding",
        "legal_interpretation", "proportionality",
    ]
    return schema


EXPERIMENT_CONFIGS = {
    "A": {
        "label": "Baseline (temp=0.4, minItems=1, current enums)",
        "temp_judge": 0.4,
        "schema_patches": [],
    },
    "B": {
        "label": "Temperature 0.7 only",
        "temp_judge": 0.7,
        "schema_patches": [],
    },
    "C": {
        "label": "Temperature 1.0 only",
        "temp_judge": 1.0,
        "schema_patches": [],
    },
    "D": {
        "label": "minItems=0 + temp=0.7",
        "temp_judge": 0.7,
        "schema_patches": ["min_items_zero"],
    },
    "E": {
        "label": "Enum reordered + temp=0.7",
        "temp_judge": 0.7,
        "schema_patches": ["reorder_enums"],
    },
    "F": {
        "label": "Combined: minItems=0 + enum reordered + temp=0.7",
        "temp_judge": 0.7,
        "schema_patches": ["min_items_zero", "reorder_enums"],
    },
}

SCHEMA_PATCH_FNS = {
    "min_items_zero": _schema_min_items_zero,
    "reorder_enums": _schema_reorder_enums,
}


# ---------------------------------------------------------------------------
# Schema injection
# ---------------------------------------------------------------------------

def build_patched_schema(patches: list[str]) -> dict:
    """Build JUDGE_CH_SCHEMA with the specified patches applied."""
    schema = _baseline_schema()
    for patch_name in patches:
        fn = SCHEMA_PATCH_FNS[patch_name]
        schema = fn(schema)
    return schema


def _monkey_patch_agent_schemas(schema: dict) -> None:
    """Temporarily replace AGENT_SCHEMAS["judge_ch"] with patched schema."""
    from athena.schemas import structured_output
    structured_output.AGENT_SCHEMAS["judge_ch"] = schema


def _restore_agent_schemas() -> None:
    """Restore the original judge_ch schema."""
    from athena.schemas import structured_output
    structured_output.AGENT_SCHEMAS["judge_ch"] = _baseline_schema()


# ---------------------------------------------------------------------------
# Case loading (simplified from cli.py)
# ---------------------------------------------------------------------------

def load_case(case_path: str) -> dict:
    """Load and prepare case data from YAML."""
    with open(case_path) as f:
        case_raw = yaml.safe_load(f)
    case_data = case_raw.get("case", case_raw)
    if "id" in case_data and "case_id" not in case_data:
        case_data["case_id"] = case_data.pop("id")
    if "key_precedents" not in case_data:
        jur = case_data.get("jurisdiction")
        if isinstance(jur, dict):
            case_data["key_precedents"] = jur.get("key_precedents", [])

    # Migrate v1 → v2 if needed
    from athena.cli import migrate_case_v1
    case_data = migrate_case_v1(case_data)
    return case_data


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------

def run_single_experiment(
    case_data: dict,
    judge_temp: float,
    run_id: str,
) -> dict:
    """Run one full simulation (appellant→respondent→judge) and return the result."""
    run_params = {
        "run_id": run_id,
        "judge_profile": {"id": "formalista_neutro", "jurisprudential_orientation": "neutral", "formalism": "high"},
        "party_profiles": {},
        "temperatures": {
            "appellant": 0.5,
            "respondent": 0.4,
            "judge": judge_temp,
        },
        "temperature": {
            "appellant": 0.5,
            "respondent": 0.4,
            "judge": judge_temp,
        },
        "appellant_profile": {
            "style": (
                "Concentrati sui vizi giuridici della decisione impugnata e sulla lettera "
                "della legge. Argomentazione analitica e testuale."
            ),
        },
        "language": "it",
    }

    from athena.simulation.graph import run_single
    result = run_single(case_data, run_params)
    return result


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def extract_metrics(result: dict) -> dict | None:
    """Extract key metrics from a simulation result."""
    decision = result.get("judge_decision")
    if decision is None:
        return None
    verdict = decision.get("verdict", {})

    # lower_court_correct
    lcc = verdict.get("lower_court_correct")

    # identified_errors analysis
    errors = verdict.get("identified_errors", [])
    severities = [e.get("severity", "unknown") for e in errors]
    error_types = [e.get("error_type", "unknown") for e in errors]
    has_decisive = "decisive" in severities

    # if_incorrect / if_correct population
    if_incorrect = verdict.get("if_incorrect")
    if_correct = verdict.get("if_correct")

    # Consistency check: decisive errors but lower_court_correct=True
    inconsistent = has_decisive and lcc is True

    # Outcome
    if lcc is True:
        outcome = "rejection"
    elif lcc is False:
        outcome = "annulment"
    else:
        outcome = "unknown"

    return {
        "lower_court_correct": lcc,
        "outcome": outcome,
        "n_errors": len(errors),
        "severities": severities,
        "error_types": error_types,
        "has_decisive": has_decisive,
        "if_incorrect_populated": if_incorrect is not None,
        "if_correct_populated": if_correct is not None,
        "inconsistent": inconsistent,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config_id: str,
    config: dict,
    case_data: dict,
    n_runs: int,
    output_dir: Path,
) -> dict:
    """Run all runs for a single experiment config. Returns summary."""
    print(f"\n{'='*60}")
    print(f"Experiment {config_id}: {config['label']}")
    print(f"  temp_judge={config['temp_judge']}, patches={config['schema_patches']}")
    print(f"  {n_runs} runs")
    print(f"{'='*60}\n")

    # Apply schema patches
    patched_schema = build_patched_schema(config["schema_patches"])
    _monkey_patch_agent_schemas(patched_schema)

    metrics_list = []
    raw_results = []

    for i in range(n_runs):
        run_id = f"cal_{config_id}_{i:03d}"
        t0 = time.time()
        print(f"\n--- Run {i+1}/{n_runs} ({run_id}) ---")

        try:
            result = run_single_experiment(
                case_data, config["temp_judge"], run_id,
            )
            elapsed = time.time() - t0
            error = result.get("error")
            if error:
                print(f"  FAIL ({elapsed:.1f}s): {error}")
                raw_results.append({"run_id": run_id, "error": error})
                continue

            m = extract_metrics(result)
            if m is None:
                print(f"  FAIL ({elapsed:.1f}s): no judge_decision")
                raw_results.append({"run_id": run_id, "error": "no_decision"})
                continue

            metrics_list.append(m)
            raw_results.append({
                "run_id": run_id,
                "metrics": m,
                "verdict": result["judge_decision"]["verdict"],
            })

            print(
                f"  OK ({elapsed:.1f}s): "
                f"lower_court_correct={m['lower_court_correct']}, "
                f"severities={m['severities']}, "
                f"inconsistent={m['inconsistent']}"
            )

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  EXCEPTION ({elapsed:.1f}s): {e}")
            raw_results.append({"run_id": run_id, "error": str(e)})

    # Restore schema
    _restore_agent_schemas()

    # Compute summary
    summary = _compute_summary(config_id, config, metrics_list, n_runs)

    # Save raw results
    exp_dir = output_dir / f"exp_{config_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "raw_results.json", "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def _compute_summary(
    config_id: str,
    config: dict,
    metrics_list: list[dict],
    n_runs: int,
) -> dict:
    """Compute aggregate summary from metrics."""
    n_ok = len(metrics_list)
    n_fail = n_runs - n_ok

    if n_ok == 0:
        return {
            "config_id": config_id,
            "label": config["label"],
            "n_runs": n_runs,
            "n_ok": 0,
            "n_fail": n_fail,
            "error": "all_failed",
        }

    # lower_court_correct distribution
    lcc_counts = Counter(m["lower_court_correct"] for m in metrics_list)
    p_correct = lcc_counts.get(True, 0) / n_ok
    p_incorrect = lcc_counts.get(False, 0) / n_ok

    # Outcome distribution
    outcome_counts = Counter(m["outcome"] for m in metrics_list)

    # Severity distribution (across all errors)
    all_severities = []
    for m in metrics_list:
        all_severities.extend(m["severities"])
    severity_counts = Counter(all_severities)

    # Error type distribution
    all_error_types = []
    for m in metrics_list:
        all_error_types.extend(m["error_types"])
    error_type_counts = Counter(all_error_types)

    # Consistency
    n_inconsistent = sum(1 for m in metrics_list if m["inconsistent"])

    # Branch completion
    n_if_incorrect_ok = sum(
        1 for m in metrics_list
        if not m["lower_court_correct"] and m["if_incorrect_populated"]
    )
    n_if_correct_ok = sum(
        1 for m in metrics_list
        if m["lower_court_correct"] and m["if_correct_populated"]
    )
    n_lcc_false = lcc_counts.get(False, 0)
    n_lcc_true = lcc_counts.get(True, 0)

    return {
        "config_id": config_id,
        "label": config["label"],
        "temp_judge": config["temp_judge"],
        "schema_patches": config["schema_patches"],
        "n_runs": n_runs,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "lower_court_correct": {
            "true": lcc_counts.get(True, 0),
            "false": lcc_counts.get(False, 0),
            "p_correct": round(p_correct, 3),
            "p_incorrect": round(p_incorrect, 3),
        },
        "outcome_distribution": dict(outcome_counts),
        "severity_distribution": dict(severity_counts),
        "error_type_distribution": dict(error_type_counts),
        "consistency": {
            "n_inconsistent": n_inconsistent,
            "inconsistency_rate": round(n_inconsistent / n_ok, 3),
        },
        "branch_completion": {
            "if_incorrect_populated": n_if_incorrect_ok,
            "if_incorrect_total": n_lcc_false,
            "if_correct_populated": n_if_correct_ok,
            "if_correct_total": n_lcc_true,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(summaries: list[dict]) -> None:
    """Print a comparison table across all experiments."""
    print(f"\n{'='*80}")
    print("CALIBRATION RESULTS")
    print(f"{'='*80}\n")

    # Header
    print(f"{'Exp':<4} {'Label':<50} {'LCC=T':>6} {'LCC=F':>6} {'P(T)':>6} {'Incon':>6}")
    print("-" * 80)

    for s in summaries:
        if "error" in s and s.get("n_ok", 0) == 0:
            print(f"{s['config_id']:<4} {s['label']:<50} {'FAILED':>6}")
            continue
        lcc = s["lower_court_correct"]
        con = s["consistency"]
        print(
            f"{s['config_id']:<4} {s['label']:<50} "
            f"{lcc['true']:>6} {lcc['false']:>6} "
            f"{lcc['p_correct']:>6.1%} {con['n_inconsistent']:>6}"
        )

    # Severity distribution
    print(f"\n{'Exp':<4} {'decisive':>10} {'significant':>12} {'minor':>8} {'none':>6} {'Total':>6}")
    print("-" * 50)
    for s in summaries:
        if "error" in s and s.get("n_ok", 0) == 0:
            continue
        sev = s.get("severity_distribution", {})
        total = sum(sev.values())
        print(
            f"{s['config_id']:<4} "
            f"{sev.get('decisive', 0):>10} "
            f"{sev.get('significant', 0):>12} "
            f"{sev.get('minor', 0):>8} "
            f"{sev.get('none', 0):>6} "
            f"{total:>6}"
        )

    # Branch completion
    print(f"\n{'Exp':<4} {'if_incorr OK':>14} {'if_incorr N':>12} {'if_corr OK':>12} {'if_corr N':>10}")
    print("-" * 56)
    for s in summaries:
        if "error" in s and s.get("n_ok", 0) == 0:
            continue
        bc = s.get("branch_completion", {})
        print(
            f"{s['config_id']:<4} "
            f"{bc.get('if_incorrect_populated', 0):>14} "
            f"{bc.get('if_incorrect_total', 0):>12} "
            f"{bc.get('if_correct_populated', 0):>12} "
            f"{bc.get('if_correct_total', 0):>10}"
        )

    print()


def print_statistical_analysis(summaries: list[dict]) -> None:
    """Print Fisher exact test p-values for key comparisons."""
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        print("[WARN] scipy not installed — skipping Fisher exact tests\n")
        return

    baseline = next((s for s in summaries if s["config_id"] == "A"), None)
    if baseline is None or "error" in baseline:
        return

    print("Statistical significance (Fisher exact, 2-sided):")
    print(f"{'Comparison':<20} {'p-value':>10} {'Significant':>12}")
    print("-" * 44)

    b_true = baseline["lower_court_correct"]["true"]
    b_false = baseline["lower_court_correct"]["false"]

    for s in summaries:
        if s["config_id"] == "A" or "error" in s:
            continue
        s_true = s["lower_court_correct"]["true"]
        s_false = s["lower_court_correct"]["false"]
        table = [[b_true, b_false], [s_true, s_false]]
        _, p = fisher_exact(table)
        sig = "YES" if p < 0.05 else "no"
        print(f"A vs {s['config_id']:<16} {p:>10.4f} {sig:>12}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CH judge bias calibration experiment")
    parser.add_argument("--case", required=True, help="Path to case YAML")
    parser.add_argument("--runs-per-config", type=int, default=5, help="Runs per experiment config")
    parser.add_argument("--configs", default="A,B,C,D,E,F", help="Comma-separated config IDs to run")
    parser.add_argument("--output", default="output/calibration", help="Output directory")
    args = parser.parse_args()

    config_ids = [c.strip() for c in args.configs.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load case
    print(f"Loading case: {args.case}")
    case_data = load_case(args.case)
    case_id = case_data.get("case_id", "unknown")
    print(f"Case ID: {case_id}, jurisdiction: {case_data.get('jurisdiction', {}).get('country', '?')}")

    # Load ground truth
    gt_path = Path("ground_truth") / f"{case_id}.json"
    if gt_path.exists():
        gt = json.loads(gt_path.read_text())
        print(f"Ground truth: {gt['outcome']}")
    else:
        print(f"No ground truth found at {gt_path}")

    # Run experiments
    summaries = []
    total_start = time.time()

    for config_id in config_ids:
        if config_id not in EXPERIMENT_CONFIGS:
            print(f"Unknown config: {config_id}, skipping")
            continue
        config = EXPERIMENT_CONFIGS[config_id]
        summary = run_experiment(config_id, config, case_data, args.runs_per_config, output_dir)
        summaries.append(summary)

    total_elapsed = time.time() - total_start
    total_runs = sum(s.get("n_ok", 0) + s.get("n_fail", 0) for s in summaries)
    print(f"\nTotal: {total_runs} runs in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Report
    print_comparison_table(summaries)
    print_statistical_analysis(summaries)

    # Save combined report
    report = {
        "case_id": case_id,
        "ground_truth": gt["outcome"] if gt_path.exists() else None,
        "runs_per_config": args.runs_per_config,
        "total_elapsed_s": round(total_elapsed, 1),
        "summaries": summaries,
    }
    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
