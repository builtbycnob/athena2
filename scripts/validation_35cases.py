#!/usr/bin/env python3
"""Full validation: 35 CH cases × 7 runs with 35B + RAG (~10h unattended).

Phase 1: Fetch 25 new cases (different seed from existing 10)
Phase 2: Run all 35 cases × 7 runs, report per-case as we go
Phase 3: Score + report after each case, final report at end

Crash-safe: each case writes results immediately.

Usage:
    OMLX_MODEL=qwen3.5-35b-a3b-text-hi uv run python3 scripts/validation_35cases.py
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMLX_MODEL", "qwen3.5-35b-a3b-text-hi")
os.environ["ATHENA_RAG_ENABLED"] = "1"
os.environ.setdefault("ATHENA_CONCURRENCY", "4")

OUTPUT_DIR = Path("output/validation-35cases")
SIM_CONFIG = "simulations/validation-ch-rag-7.yaml"
CASES_DIR = Path("cases/validation")
GT_DIR = Path("ground_truth")

# Existing 10 cases
EXISTING_CASES = [
    "ch-247", "ch-741", "ch-1124", "ch-1253", "ch-1272",
    "ch-2358", "ch-2434", "ch-2461", "ch-3295", "ch-3408",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    # Also append to run log file
    try:
        with open(OUTPUT_DIR / "run.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Phase 1: Fetch new cases ─────────────────────────────────────

def fetch_new_cases() -> list[str]:
    """Fetch 25 new CH civil law cases. Returns list of case IDs."""
    log("PHASE 1: Fetching 25 new cases from HuggingFace...")

    from athena.validation.dataset_fetcher import fetch_swiss_cases
    from athena.validation.case_extractor import extract_and_save

    # Use different seed (99) to avoid duplicating existing cases (seed=42)
    records = fetch_swiss_cases(
        legal_area="civil_law",
        min_year=2000,
        max_words=2000,
        n_rejection=13,
        n_approval=12,
        seed=99,
    )

    # Filter out IDs we already have
    existing_ids = {c.replace("ch-", "") for c in EXISTING_CASES}
    records = [r for r in records if str(r["id"]) not in existing_ids]

    # Take up to 25
    records = records[:25]
    log(f"Fetched {len(records)} new records (after dedup)")

    new_case_ids = []
    for i, record in enumerate(records, 1):
        case_id = f"ch-{record['id']}"
        label_str = "rejection" if record["label"] == 0 else "annulment"
        log(f"  [{i}/{len(records)}] Extracting {case_id} ({label_str})...")
        try:
            yaml_path, gt_path = extract_and_save(
                record, str(CASES_DIR), str(GT_DIR), use_llm=True,
            )
            new_case_ids.append(case_id)
            log(f"  [{i}/{len(records)}] {case_id} — OK → {yaml_path}")
        except Exception as e:
            log(f"  [{i}/{len(records)}] {case_id} — FAILED: {e}")

    log(f"Phase 1 done: {len(new_case_ids)} new cases extracted")
    return new_case_ids


# ── Phase 2: Run simulations ─────────────────────────────────────

def run_single_case(case_id: str) -> dict:
    """Run one case × 7 runs. Returns timing dict. Never raises."""
    import yaml as yaml_lib
    from athena.api.models import PipelineOptions, ProgressEvent
    from athena.api.pipeline import (
        prepare_case_data,
        prepare_sim_config,
        run_pipeline,
        write_pipeline_outputs,
    )

    case_path = CASES_DIR / f"{case_id}.yaml"
    case_output = OUTPUT_DIR / case_id
    timing = {
        "case_id": case_id,
        "status": "unknown",
        "llm_calls": [],
    }

    t0 = time.time()
    try:
        _patch_llm_logging(timing)

        with open(case_path) as f:
            case_raw = yaml_lib.safe_load(f)
        case_data = prepare_case_data(case_raw)

        with open(SIM_CONFIG) as f:
            sim_raw = yaml_lib.safe_load(f)
        sim_config_dict = prepare_sim_config(sim_raw)

        options = PipelineOptions(
            concurrency=None,
            kg_enabled=False,
            rag_enabled=True,
        )

        result = run_pipeline(case_data, sim_config_dict, options, lambda e: None)
        write_pipeline_outputs(result, str(case_output))

        elapsed = time.time() - t0
        timing["status"] = "ok"
        timing["elapsed_s"] = round(elapsed, 1)
        timing["n_results"] = len(result.results)

        # Extract verdicts
        verdicts = []
        for r in result.results:
            v = r.get("judge_decision", {}).get("verdict", {})
            lcc = v.get("lower_court_correct")
            verdicts.append(lcc)
        timing["verdicts"] = verdicts
        timing["n_reject"] = sum(1 for v in verdicts if v is True)
        timing["n_annul"] = sum(1 for v in verdicts if v is False)
        timing["n_none"] = sum(1 for v in verdicts if v is None)

    except Exception as e:
        elapsed = time.time() - t0
        timing["status"] = "error"
        timing["elapsed_s"] = round(elapsed, 1)
        timing["error"] = str(e)
        timing["traceback"] = traceback.format_exc()

    finally:
        _unpatch_llm_logging()

    # Write per-case timing immediately
    case_output.mkdir(parents=True, exist_ok=True)
    (case_output / "timing.json").write_text(json.dumps(timing, indent=2, default=str))
    return timing


# ── Phase 3: Scoring ─────────────────────────────────────────────

def score_and_report(all_timings: list[dict], label: str = "") -> None:
    """Score current results and write report."""
    try:
        from athena.validation.scorer import score_results
        report = score_results(str(OUTPUT_DIR), str(GT_DIR))

        if report.n == 0:
            log(f"Scoring: no results to score yet")
            return

        ci_low, ci_high = report.accuracy_ci
        log(f"SCORE{label}: {report.accuracy:.0%} [{ci_low:.0%}, {ci_high:.0%}] "
            f"({report.n} cases, log_loss={report.log_loss:.3f}, ECE={report.ece:.3f})")

        # Build detailed report
        lines = [
            f"# ATHENA v1.4b — 35-Case Validation Report\n",
            f"- **Date**: {time.strftime('%Y-%m-%d %H:%M')}",
            f"- **Model**: 35B (qwen3.5-35b-a3b-text-hi)",
            f"- **RAG**: enabled (747K Swiss law chunks)",
            f"- **Runs per case**: 7",
            f"- **Cases scored**: {report.n}",
            "",
        ]

        lines.append(report.to_markdown())

        # Per-case detail table
        lines.append("\n## Per-Case Results\n")
        lines.append("| Case | Status | Reject | Annul | None | Time | GT | Pred | Match |")
        lines.append("|------|--------|--------|-------|------|------|----|------|-------|")

        for t in sorted(all_timings, key=lambda x: x["case_id"]):
            cid = t["case_id"]
            gt_file = GT_DIR / f"{cid}.json"
            gt_outcome = "?"
            if gt_file.exists():
                gt_outcome = json.loads(gt_file.read_text()).get("outcome", "?")

            nr = t.get("n_reject", 0)
            na = t.get("n_annul", 0)
            nn = t.get("n_none", 0)
            pred = "rejection" if nr > na else "annulment" if na > 0 else "?"
            match = "✓" if pred == gt_outcome else "✗" if t["status"] == "ok" else "—"
            elapsed = t.get("elapsed_s", 0)

            lines.append(
                f"| {cid} | {t['status']} | {nr} | {na} | {nn} "
                f"| {elapsed:.0f}s | {gt_outcome} | {pred} | {match} |"
            )

        # Throughput stats
        all_calls = []
        for t in all_timings:
            all_calls.extend(t.get("llm_calls", []))
        if all_calls:
            judge_calls = [c for c in all_calls if c.get("prompt_tokens", 0) > 10000]
            party_calls = [c for c in all_calls if c.get("prompt_tokens", 0) <= 10000]
            lines.append("\n## Throughput\n")
            if judge_calls:
                avg = sum(c["tok_s"] for c in judge_calls) / len(judge_calls)
                lines.append(f"- **Judge**: {len(judge_calls)} calls, avg {avg:.1f} tok/s")
            if party_calls:
                avg = sum(c["tok_s"] for c in party_calls) / len(party_calls)
                lines.append(f"- **Party**: {len(party_calls)} calls, avg {avg:.1f} tok/s")
            lines.append(f"- **Total LLM calls**: {len(all_calls)}")

        report_path = OUTPUT_DIR / "validation_report.md"
        report_path.write_text("\n".join(lines))

        # Also write errors
        errors = report.error_analysis()
        if errors:
            log(f"  Errors: {', '.join(e['case_id'] for e in errors)}")

    except Exception as e:
        log(f"Scoring failed: {e}")
        traceback.print_exc()


# ── LLM instrumentation ──────────────────────────────────────────

_original_call_model = None
_current_timing = None


def _patch_llm_logging(timing: dict) -> None:
    global _original_call_model, _current_timing
    from athena.agents import llm

    if _original_call_model is None:
        _original_call_model = llm._call_model

    _current_timing = timing

    def _instrumented(*args, **kwargs):
        t0 = time.time()
        result = _original_call_model(*args, **kwargs)
        elapsed = time.time() - t0
        raw, finish, ptok, otok = result
        tok_s = otok / elapsed if elapsed > 0 else 0
        if _current_timing is not None:
            _current_timing["llm_calls"].append({
                "prompt_tokens": ptok,
                "output_tokens": otok,
                "elapsed_s": round(elapsed, 1),
                "tok_s": round(tok_s, 1),
            })
        return result

    llm._call_model = _instrumented


def _unpatch_llm_logging() -> None:
    global _current_timing
    _current_timing = None


# ── Main ──────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    log("=" * 70)
    log("ATHENA v1.4b — 35-CASE VALIDATION (35B + RAG, 7 runs/case)")
    log(f"Estimated: ~10 hours")
    log("=" * 70)

    # Phase 1: Fetch new cases
    try:
        new_case_ids = fetch_new_cases()
    except Exception as e:
        log(f"PHASE 1 FAILED: {e}")
        traceback.print_exc()
        new_case_ids = []

    all_cases = EXISTING_CASES + new_case_ids
    log(f"\nTotal cases: {len(all_cases)} ({len(EXISTING_CASES)} existing + {len(new_case_ids)} new)")

    # Save case list for reference
    (OUTPUT_DIR / "cases.json").write_text(json.dumps({
        "existing": EXISTING_CASES,
        "new": new_case_ids,
        "all": all_cases,
    }, indent=2))

    # Phase 2: Run simulations
    log(f"\nPHASE 2: Running {len(all_cases)} cases × 7 runs = {len(all_cases) * 7} simulations")

    all_timings = []
    for i, case_id in enumerate(all_cases, 1):
        log(f"\n[{i}/{len(all_cases)}] {case_id}...")

        timing = run_single_case(case_id)

        status = timing["status"]
        elapsed = timing.get("elapsed_s", 0)
        if status == "ok":
            nr = timing.get("n_reject", "?")
            na = timing.get("n_annul", "?")
            log(f"[{i}/{len(all_cases)}] {case_id} — OK ({elapsed:.0f}s) reject={nr} annul={na}")
        else:
            err = timing.get("error", "unknown")[:120]
            log(f"[{i}/{len(all_cases)}] {case_id} — FAILED ({elapsed:.0f}s): {err}")

        all_timings.append(timing)

        # Score after every 5 cases (intermediate reports)
        if i % 5 == 0 or i == len(all_cases):
            score_and_report(all_timings, f" (after {i}/{len(all_cases)} cases)")

    # Phase 3: Final report
    log("\nPHASE 3: Final scoring")
    score_and_report(all_timings, " [FINAL]")

    # Write aggregated timing
    total_elapsed = time.time() - total_start
    (OUTPUT_DIR / "timing.json").write_text(json.dumps({
        "total_elapsed_s": round(total_elapsed, 1),
        "total_elapsed_h": round(total_elapsed / 3600, 2),
        "n_cases": len(all_cases),
        "n_runs_per_case": 7,
        "cases": all_timings,
    }, indent=2, default=str))

    hours = total_elapsed / 3600
    log(f"\n{'=' * 70}")
    log(f"VALIDATION COMPLETE in {hours:.1f} hours")
    log(f"Results: {OUTPUT_DIR}/")
    log(f"Report: {OUTPUT_DIR}/validation_report.md")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
