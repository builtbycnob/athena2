#!/usr/bin/env python3
"""Ablation study: 4 configs × 10 cases × 6 runs (~10h unattended).

Each config writes its own report immediately after completion.
Each case writes results as it completes (crash-safe).

Output structure:
  output/ablation-v14b/
    A-35b-rag/ch-247/... + validation_report.md + timing.json
    B-35b-norag/ch-247/... + validation_report.md + timing.json
    C-122b-rag/ch-247/... + validation_report.md + timing.json
    D-122b-norag/ch-247/... + validation_report.md + timing.json
    comparison_report.md  (written at the end, non-critical)

Usage:
    OMLX_MODEL=qwen3.5-35b-a3b-text-hi uv run python3 scripts/ablation_v14b.py
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── Ensure ATHENA is importable ──────────────────────────────────
os.environ.setdefault("OMLX_MODEL", "qwen3.5-35b-a3b-text-hi")
os.environ.setdefault("ATHENA_CONCURRENCY", "4")

BASE_OUTPUT = Path("output/ablation-v14b")
SIM_CONFIG = "simulations/validation-ch-rag-6.yaml"
CASES_DIR = Path("cases/validation")
GT_DIR = "ground_truth"

CASES = [
    "ch-247", "ch-741", "ch-1124", "ch-1253", "ch-1272",
    "ch-2358", "ch-2434", "ch-2461", "ch-3295", "ch-3408",
]

CONFIGS = [
    {"id": "A-35b-rag",    "model": None,                       "rag": True},
    {"id": "B-35b-norag",  "model": None,                       "rag": False},
    {"id": "C-122b-rag",   "model": "qwen3.5-122b-a10b-4bit",  "rag": True},
    {"id": "D-122b-norag", "model": "qwen3.5-122b-a10b-4bit",  "rag": False},
]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_single_case(
    case_id: str,
    output_dir: Path,
    model_override: str | None,
    rag_enabled: bool,
) -> dict:
    """Run one case, return timing dict. Never raises."""
    from athena.api.models import PipelineOptions, ProgressEvent
    from athena.api.pipeline import (
        prepare_case_data,
        prepare_sim_config,
        run_pipeline,
        write_pipeline_outputs,
    )
    import yaml

    case_path = CASES_DIR / f"{case_id}.yaml"
    case_output = output_dir / case_id
    timing = {
        "case_id": case_id,
        "model": model_override or os.environ.get("OMLX_MODEL", "default"),
        "rag": rag_enabled,
        "status": "unknown",
        "llm_calls": [],
    }

    t0 = time.time()
    try:
        # Intercept LLM calls for timing data
        _patch_llm_logging(timing)

        # Set env for this run
        if rag_enabled:
            os.environ["ATHENA_RAG_ENABLED"] = "1"
        else:
            os.environ.pop("ATHENA_RAG_ENABLED", None)

        # Load case
        with open(case_path) as f:
            case_raw = yaml.safe_load(f)
        case_data = prepare_case_data(case_raw)

        # Load sim config
        with open(SIM_CONFIG) as f:
            sim_raw = yaml.safe_load(f)
        sim_config_dict = prepare_sim_config(sim_raw)

        # Inject model override via sim config
        if model_override:
            sim_config_dict["models"] = {"judge": model_override}

        options = PipelineOptions(
            concurrency=None,
            kg_enabled=False,
            rag_enabled=rag_enabled,
        )

        result = run_pipeline(case_data, sim_config_dict, options, lambda e: None)
        write_pipeline_outputs(result, str(case_output))

        elapsed = time.time() - t0
        timing["status"] = "ok"
        timing["elapsed_s"] = round(elapsed, 1)
        timing["n_results"] = len(result.results)

        # Extract verdicts for quick check
        verdicts = []
        for r in result.results:
            v = r.get("judge_decision", {}).get("verdict", {})
            lcc = v.get("lower_court_correct")
            verdicts.append(lcc)
        timing["verdicts"] = verdicts
        timing["n_reject"] = sum(1 for v in verdicts if v is True)
        timing["n_annul"] = sum(1 for v in verdicts if v is False)

    except Exception as e:
        elapsed = time.time() - t0
        timing["status"] = "error"
        timing["elapsed_s"] = round(elapsed, 1)
        timing["error"] = str(e)
        timing["traceback"] = traceback.format_exc()

    finally:
        _unpatch_llm_logging()

    # Write timing for this case immediately (crash-safe)
    case_output.mkdir(parents=True, exist_ok=True)
    (case_output / "timing.json").write_text(json.dumps(timing, indent=2))
    return timing


def run_config(config: dict) -> None:
    """Run all 10 cases for one config, write report immediately."""
    config_id = config["id"]
    model = config["model"]
    rag = config["rag"]
    output_dir = BASE_OUTPUT / config_id

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"{'='*60}")
    log(f"CONFIG {config_id}: model={model or 'default(35B)'}, rag={rag}")
    log(f"Output: {output_dir}")
    log(f"{'='*60}")

    all_timings = []
    config_start = time.time()

    for i, case_id in enumerate(CASES, 1):
        log(f"[{config_id}] [{i}/{len(CASES)}] {case_id}...")

        timing = run_single_case(case_id, output_dir, model, rag)

        status = timing["status"]
        elapsed = timing.get("elapsed_s", 0)
        if status == "ok":
            n_r = timing.get("n_reject", "?")
            n_a = timing.get("n_annul", "?")
            log(f"[{config_id}] [{i}/{len(CASES)}] {case_id} — OK ({elapsed:.0f}s) reject={n_r} annul={n_a}")
        else:
            err = timing.get("error", "unknown")[:100]
            log(f"[{config_id}] [{i}/{len(CASES)}] {case_id} — FAILED ({elapsed:.0f}s): {err}")

        all_timings.append(timing)

    config_elapsed = time.time() - config_start

    # Write aggregated timing for this config
    (output_dir / "timing.json").write_text(json.dumps({
        "config": config,
        "total_elapsed_s": round(config_elapsed, 1),
        "cases": all_timings,
    }, indent=2))

    # Score against ground truth
    log(f"[{config_id}] Scoring against ground truth...")
    try:
        from athena.validation.scorer import score_results
        report = score_results(str(output_dir), GT_DIR)
        md = report.to_markdown()

        # Add config header
        header = (
            f"# Ablation: {config_id}\n\n"
            f"- **Model**: {model or 'default (35B)'}\n"
            f"- **RAG**: {'enabled' if rag else 'disabled'}\n"
            f"- **Total time**: {config_elapsed/60:.0f} min\n"
            f"- **Cases**: {len(CASES)} × 6 runs\n\n"
        )
        md = header + md

        # Add per-case details
        md += "\n## Per-Case Results\n\n"
        md += "| Case | Status | Reject | Annul | Time | GT |\n"
        md += "|------|--------|--------|-------|------|----|\n"
        for t in all_timings:
            gt_file = Path(GT_DIR) / f"{t['case_id']}.json"
            gt_outcome = "?"
            if gt_file.exists():
                gt_outcome = json.loads(gt_file.read_text()).get("outcome", "?")
            md += (
                f"| {t['case_id']} | {t['status']} "
                f"| {t.get('n_reject', '-')} | {t.get('n_annul', '-')} "
                f"| {t.get('elapsed_s', 0):.0f}s | {gt_outcome} |\n"
            )

        # Add LLM throughput summary
        all_calls = []
        for t in all_timings:
            all_calls.extend(t.get("llm_calls", []))
        if all_calls:
            judge_calls = [c for c in all_calls if c.get("prompt_tokens", 0) > 10000]
            party_calls = [c for c in all_calls if c.get("prompt_tokens", 0) <= 10000]
            md += "\n## Throughput\n\n"
            if judge_calls:
                avg_toks = sum(c["tok_s"] for c in judge_calls) / len(judge_calls)
                md += f"- **Judge**: {len(judge_calls)} calls, avg {avg_toks:.1f} tok/s\n"
            if party_calls:
                avg_toks = sum(c["tok_s"] for c in party_calls) / len(party_calls)
                md += f"- **Party**: {len(party_calls)} calls, avg {avg_toks:.1f} tok/s\n"

        report_path = output_dir / "validation_report.md"
        report_path.write_text(md)
        log(f"[{config_id}] Report saved: {report_path}")

        ci_low, ci_high = report.accuracy_ci
        log(f"[{config_id}] Accuracy: {report.accuracy:.0%} [{ci_low:.0%}, {ci_high:.0%}]")

    except Exception as e:
        log(f"[{config_id}] Scoring failed: {e}")
        (output_dir / "scoring_error.txt").write_text(traceback.format_exc())

    log(f"[{config_id}] Done in {config_elapsed/60:.0f} min")


# ── LLM call timing instrumentation ──────────────────────────────

_original_call_model = None
_current_timing = None


def _patch_llm_logging(timing: dict) -> None:
    """Monkey-patch _call_model to capture per-call timing."""
    global _original_call_model, _current_timing
    from athena.agents import llm

    if _original_call_model is None:
        _original_call_model = llm._call_model

    _current_timing = timing

    def _instrumented_call_model(*args, **kwargs):
        t0 = time.time()
        result = _original_call_model(*args, **kwargs)
        elapsed = time.time() - t0
        raw, finish, prompt_tok, output_tok = result
        tok_s = output_tok / elapsed if elapsed > 0 else 0
        if _current_timing is not None:
            _current_timing["llm_calls"].append({
                "prompt_tokens": prompt_tok,
                "output_tokens": output_tok,
                "elapsed_s": round(elapsed, 1),
                "tok_s": round(tok_s, 1),
            })
        return result

    llm._call_model = _instrumented_call_model


def _unpatch_llm_logging() -> None:
    """Restore original _call_model."""
    global _current_timing
    _current_timing = None
    # Keep patch active — it's safe across cases


# ── Comparison report ─────────────────────────────────────────────

def write_comparison_report() -> None:
    """Write final comparison across all configs (non-critical)."""
    log("Writing comparison report...")
    try:
        from athena.validation.scorer import score_results

        lines = ["# Ablation Study — v1.4b Comparison\n"]
        lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        lines.append("| Config | Model | RAG | Accuracy | CI 95% | Log Loss | ECE | Time |")
        lines.append("|--------|-------|-----|----------|--------|----------|-----|------|")

        for config in CONFIGS:
            config_dir = BASE_OUTPUT / config["id"]
            timing_path = config_dir / "timing.json"
            if not timing_path.exists():
                lines.append(f"| {config['id']} | — | — | MISSING | — | — | — | — |")
                continue

            timing_data = json.loads(timing_path.read_text())
            total_min = timing_data.get("total_elapsed_s", 0) / 60

            report = score_results(str(config_dir), GT_DIR)
            ci_low, ci_high = report.accuracy_ci
            model_name = config["model"] or "35B"
            rag_str = "on" if config["rag"] else "off"
            lines.append(
                f"| {config['id']} | {model_name} | {rag_str} "
                f"| {report.accuracy:.0%} | [{ci_low:.0%}, {ci_high:.0%}] "
                f"| {report.log_loss:.3f} | {report.ece:.3f} | {total_min:.0f}m |"
            )

        # Per-case comparison matrix
        lines.append("\n## Per-Case Comparison\n")
        lines.append("| Case | GT | " + " | ".join(c["id"] for c in CONFIGS) + " |")
        lines.append("|------|----| " + " | ".join("---" for _ in CONFIGS) + " |")

        for case_id in CASES:
            gt_file = Path(GT_DIR) / f"{case_id}.json"
            gt_outcome = "?"
            if gt_file.exists():
                gt_outcome = json.loads(gt_file.read_text()).get("outcome", "?")

            row = [case_id, gt_outcome]
            for config in CONFIGS:
                config_dir = BASE_OUTPUT / config["id"]
                case_timing = config_dir / case_id / "timing.json"
                if case_timing.exists():
                    t = json.loads(case_timing.read_text())
                    nr = t.get("n_reject", 0)
                    na = t.get("n_annul", 0)
                    predicted = "rej" if nr > na else "ann"
                    match = "✓" if (
                        (predicted == "rej" and gt_outcome == "rejection") or
                        (predicted == "ann" and gt_outcome == "annulment")
                    ) else "✗"
                    row.append(f"{nr}/{nr+na} rej {match}")
                else:
                    row.append("—")

            lines.append("| " + " | ".join(row) + " |")

        report_path = BASE_OUTPUT / "comparison_report.md"
        report_path.write_text("\n".join(lines))
        log(f"Comparison report: {report_path}")

    except Exception as e:
        log(f"Comparison report failed (non-critical): {e}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    total_start = time.time()
    log("=" * 60)
    log("ATHENA v1.4b ABLATION STUDY")
    log(f"Configs: {len(CONFIGS)}, Cases: {len(CASES)}, Runs/case: 6")
    log(f"Estimated: ~10 hours")
    log("=" * 60)

    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        try:
            run_config(config)
        except Exception as e:
            log(f"CONFIG {config['id']} FAILED: {e}")
            traceback.print_exc()
            # Continue to next config

    write_comparison_report()

    total_hours = (time.time() - total_start) / 3600
    log(f"ABLATION COMPLETE in {total_hours:.1f} hours")


if __name__ == "__main__":
    main()
