#!/usr/bin/env python3
"""Run full 122B validation: 10 CH cases × 6 runs with RAG."""
import os
import sys
import time

os.environ.setdefault("OMLX_MODEL", "qwen3.5-35b-a3b-text-hi")
os.environ["ATHENA_RAG_ENABLED"] = "1"
os.environ.setdefault("ATHENA_CONCURRENCY", "4")

OUTPUT_DIR = "output/validation-v14b-122b"
SIM_CONFIG = "simulations/validation-ch-rag-6.yaml"
CASES_DIR = "cases/validation"

CASES = [
    "ch-247", "ch-741", "ch-1124", "ch-1253", "ch-1272",
    "ch-2358", "ch-2434", "ch-2461", "ch-3295", "ch-3408",
]


def main():
    from athena.cli import main as cli_main

    print(f"=== ATHENA v1.4b Validation — 122B Judge ===")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    total_start = time.time()

    ok = 0
    fail = 0

    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] Running {case}...")
        case_start = time.time()
        case_output = f"{OUTPUT_DIR}/{case}"
        try:
            cli_main([
                "run",
                "--case", f"{CASES_DIR}/{case}.yaml",
                "--simulation", SIM_CONFIG,
                "--output", case_output,
                "--rag",
            ])
            elapsed = time.time() - case_start
            print(f"[{i}/{len(CASES)}] {case} — OK ({elapsed:.0f}s)")
            ok += 1
        except Exception as e:
            elapsed = time.time() - case_start
            print(f"[{i}/{len(CASES)}] {case} — FAILED ({elapsed:.0f}s): {e}")
            fail += 1

    total_time = time.time() - total_start
    print(f"\n=== Runs complete: {ok}/{len(CASES)} OK, {fail} failed ({total_time:.0f}s) ===\n")

    # Score
    print("=== Scoring against ground truth ===")
    cli_main(["validate", "--results-dir", OUTPUT_DIR, "--ground-truth", "ground_truth"])

    print(f"\n=== Done: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()
