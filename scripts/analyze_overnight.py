#!/usr/bin/env python3
"""Analyze overnight two-step validation results.

Reads raw_results.json from each case directory and produces:
1. Accuracy table per case (with majority vote)
2. Step1→Step2 flow analysis
3. Severity distribution
4. Confusion matrix
5. Systematic error identification

Usage:
    python scripts/analyze_overnight.py \
        --results-dir output/overnight-twostep-YYYYMMDD \
        --ground-truth ground_truth
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from athena.validation.ground_truth import load_ground_truths


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def extract_run_outcome(result: dict) -> dict:
    """Extract outcome + step1/step2 details from a single run result."""
    decision = result.get("judge_decision", {})
    verdict = decision.get("verdict", {})

    # Two-step schema: lower_court_correct
    lcc = verdict.get("lower_court_correct")
    if lcc is True:
        outcome = "rejection"
    elif lcc is False:
        outcome = "annulment"
    else:
        # Legacy flat schema fallback
        ao = verdict.get("appeal_outcome", "")
        if ao == "dismissed":
            outcome = "rejection"
        elif ao in ("upheld", "partially_upheld", "remanded"):
            outcome = "annulment"
        else:
            outcome = "unknown"

    # Step 1: identified errors
    errors = verdict.get("identified_errors", [])
    severities = [e.get("severity", "unknown") for e in errors]
    error_types = [e.get("error_type", "unknown") for e in errors]

    # Step 2: if_incorrect / if_correct
    if_incorrect = verdict.get("if_incorrect")
    if_correct = verdict.get("if_correct")

    return {
        "outcome": outcome,
        "lower_court_correct": lcc,
        "n_errors": len(errors),
        "severities": severities,
        "error_types": error_types,
        "has_decisive": "decisive" in severities,
        "if_incorrect": if_incorrect,
        "if_correct": if_correct,
    }


def majority_vote(outcomes: list[str]) -> str:
    """Return majority outcome from a list."""
    counts = Counter(outcomes)
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(results_dir: Path, gt_dir: Path) -> dict:
    """Run full analysis. Returns structured report."""
    ground_truths = load_ground_truths(gt_dir)
    case_ids = sorted(ground_truths.keys())

    cases = []
    all_severities = Counter()
    tp = tn = fp = fn = 0
    systematic_errors = []

    for case_id in case_ids:
        gt = ground_truths[case_id]
        raw_path = results_dir / case_id / "raw_results.json"
        if not raw_path.exists():
            cases.append({
                "case_id": case_id,
                "gt": gt.outcome,
                "status": "missing",
            })
            continue

        raw_results = json.loads(raw_path.read_text())
        runs = []
        for r in raw_results:
            run_data = extract_run_outcome(r)
            run_data["run_id"] = r.get("run_id", "?")
            runs.append(run_data)

        outcomes = [r["outcome"] for r in runs]
        mv = majority_vote(outcomes)
        correct = mv == gt.outcome

        # Confusion matrix (rejection=positive, annulment=negative)
        for o in outcomes:
            if o == "rejection" and gt.outcome == "rejection":
                tp += 1
            elif o == "annulment" and gt.outcome == "annulment":
                tn += 1
            elif o == "rejection" and gt.outcome == "annulment":
                fp += 1
            elif o == "annulment" and gt.outcome == "rejection":
                fn += 1

        # Severity counts
        for r in runs:
            all_severities.update(r["severities"])

        # Systematic error: all runs wrong
        if all(o != gt.outcome for o in outcomes if o != "unknown"):
            systematic_errors.append(case_id)

        cases.append({
            "case_id": case_id,
            "gt": gt.outcome,
            "runs": outcomes,
            "majority_vote": mv,
            "correct": correct,
            "run_details": runs,
            "status": "ok",
        })

    # Compute aggregate metrics
    scored = [c for c in cases if c["status"] == "ok"]
    n_correct = sum(1 for c in scored if c["correct"])
    accuracy = n_correct / len(scored) if scored else 0

    return {
        "cases": cases,
        "accuracy": accuracy,
        "n_scored": len(scored),
        "n_correct": n_correct,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "severity_distribution": dict(all_severities),
        "systematic_errors": systematic_errors,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    """Print human-readable report to stdout."""
    cases = report["cases"]

    # 1. Accuracy table
    print("=" * 80)
    print("OVERNIGHT TWO-STEP VALIDATION — RESULTS")
    print("=" * 80)
    print()
    print(f"{'Case':<10} {'GT':<12} {'Run1':<12} {'Run2':<12} {'Run3':<12} {'MajVote':<12} {'OK?'}")
    print("-" * 80)

    for c in cases:
        if c["status"] == "missing":
            print(f"{c['case_id']:<10} {c['gt']:<12} {'— MISSING —'}")
            continue
        runs = c["runs"]
        run_strs = [f"{r:<12}" for r in (runs + [""] * 3)[:3]]
        mark = "Y" if c["correct"] else "N"
        print(f"{c['case_id']:<10} {c['gt']:<12} {''.join(run_strs)}{c['majority_vote']:<12} {mark}")

    print("-" * 80)
    print(f"Accuracy: {report['n_correct']}/{report['n_scored']} = {report['accuracy']:.0%}")
    print()

    # 2. Confusion matrix
    cm = report["confusion_matrix"]
    print("CONFUSION MATRIX (rejection=positive)")
    print(f"  TP={cm['tp']}  FP={cm['fp']}")
    print(f"  FN={cm['fn']}  TN={cm['tn']}")
    total = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
    if total:
        precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0
        recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0
        print(f"  Precision(rejection)={precision:.2f}  Recall(rejection)={recall:.2f}")
    print()

    # 3. Step1→Step2 flow
    print("STEP 1→STEP 2 FLOW")
    print(f"{'Case':<10} {'Run':<30} {'#Errors':>8} {'Decisive?':>10} {'LCC':>6} {'Outcome':<12}")
    print("-" * 80)
    for c in cases:
        if c["status"] != "ok":
            continue
        for rd in c["run_details"]:
            lcc_str = str(rd["lower_court_correct"]) if rd["lower_court_correct"] is not None else "N/A"
            dec = "Y" if rd["has_decisive"] else "N"
            print(
                f"{c['case_id']:<10} {rd['run_id']:<30} "
                f"{rd['n_errors']:>8} {dec:>10} {lcc_str:>6} {rd['outcome']:<12}"
            )
    print()

    # 4. Severity distribution
    sev = report["severity_distribution"]
    print("SEVERITY DISTRIBUTION (all runs)")
    for s in ["decisive", "significant", "minor", "none"]:
        print(f"  {s}: {sev.get(s, 0)}")
    print(f"  Total: {sum(sev.values())}")
    print()

    # 5. Systematic errors
    if report["systematic_errors"]:
        print(f"SYSTEMATIC ERRORS (all runs wrong): {', '.join(report['systematic_errors'])}")
    else:
        print("SYSTEMATIC ERRORS: none")
    print()


def save_json_report(report: dict, output_path: Path) -> None:
    """Save structured JSON report (without run_details for brevity)."""
    slim = {
        "accuracy": report["accuracy"],
        "n_scored": report["n_scored"],
        "n_correct": report["n_correct"],
        "confusion_matrix": report["confusion_matrix"],
        "severity_distribution": report["severity_distribution"],
        "systematic_errors": report["systematic_errors"],
        "cases": [],
    }
    for c in report["cases"]:
        entry = {
            "case_id": c["case_id"],
            "gt": c["gt"],
            "status": c["status"],
        }
        if c["status"] == "ok":
            entry["runs"] = c["runs"]
            entry["majority_vote"] = c["majority_vote"]
            entry["correct"] = c["correct"]
        slim["cases"].append(entry)

    output_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False))
    print(f"JSON report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze overnight two-step validation results")
    parser.add_argument("--results-dir", required=True, help="Directory with per-case results")
    parser.add_argument("--ground-truth", default="ground_truth", help="Ground truth directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    gt_dir = Path(args.ground_truth)

    report = analyze(results_dir, gt_dir)
    print_report(report)

    json_path = results_dir / "analysis.json"
    save_json_report(report, json_path)


if __name__ == "__main__":
    main()
