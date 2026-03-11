# src/athena/simulation/orchestrator.py
"""Monte Carlo orchestrator for running simulations across all parameter combinations.

Iterates over all (judge_profile × appellant_profile × N runs) combinations,
invokes run_single (the LangGraph graph) for each, and collects results.
Supports concurrent execution via ThreadPoolExecutor.
"""

import concurrent.futures
import itertools
import os
import sys
import time
from typing import Any

from langfuse import observe

from athena.simulation.graph import build_graph


_DEFAULT_CONCURRENCY = 4


def _get_concurrency() -> int:
    """Read concurrency from ATHENA_CONCURRENCY env var, default 4."""
    raw = os.environ.get("ATHENA_CONCURRENCY", "")
    if raw.strip().isdigit() and int(raw) > 0:
        return int(raw)
    return _DEFAULT_CONCURRENCY


def _log(msg: str) -> None:
    """Print and flush immediately for real-time progress visibility."""
    print(msg, flush=True)


def _run_one(
    graph,
    case_data: dict,
    judge_profile: dict,
    appellant_profile: dict,
    run_n: int,
    temperature: dict,
    language: str,
    run_index: int,
    total: int,
) -> dict:
    """Execute a single simulation run. Returns a status dict; never raises."""
    run_id = f"{judge_profile['id']}__{appellant_profile['id']}__{run_n:03d}"
    run_start = time.time()

    _log(f"[MC] Run {run_index}/{total}: {run_id}")

    initial_state = {
        "case": case_data,
        "params": {
            "run_id": run_id,
            "judge_profile": judge_profile,
            "appellant_profile": appellant_profile,
            "temperature": temperature,
            "language": language,
        },
        "appellant_brief": None,
        "appellant_validation": None,
        "respondent_brief": None,
        "respondent_validation": None,
        "judge_decision": None,
        "judge_validation": None,
        "retry_count": 0,
        "error": None,
    }

    try:
        final_state = graph.invoke(initial_state)
        elapsed = time.time() - run_start

        if final_state.get("error"):
            _log(f"[MC]   FAIL ({elapsed:.1f}s): {final_state['error']}")
            return {"status": "fail", "run_id": run_id, "error": final_state["error"]}

        result = {
            "run_id": run_id,
            "judge_profile": judge_profile["id"],
            "appellant_profile": appellant_profile["id"],
            "appellant_brief": final_state.get("appellant_brief"),
            "respondent_brief": final_state.get("respondent_brief"),
            "judge_decision": final_state["judge_decision"],
            "validation_warnings": {
                "appellant": (final_state.get("appellant_validation") or {}).get("warnings", []),
                "respondent": (final_state.get("respondent_validation") or {}).get("warnings", []),
                "judge": (final_state.get("judge_validation") or {}).get("warnings", []),
            },
        }
        _log(f"[MC]   OK ({elapsed:.1f}s) — {run_id}")
        return {"status": "ok", "run_id": run_id, "elapsed": elapsed, "result": result}

    except Exception as e:
        elapsed = time.time() - run_start
        _log(f"[MC]   EXCEPTION ({elapsed:.1f}s): {e}")
        return {"status": "exception", "run_id": run_id, "error": str(e)}


@observe(name="monte_carlo")
def run_monte_carlo(case_data: dict, simulation_config: dict) -> list[dict]:
    """Execute all simulation combinations and collect results."""
    combinations = list(itertools.product(
        simulation_config["judge_profiles"],
        simulation_config["appellant_profiles"],
        range(simulation_config["runs_per_combination"]),
    ))

    total = len(combinations)
    results: list[dict] = []
    failed: list[dict] = []
    graph = build_graph()
    sim_start = time.time()
    concurrency = _get_concurrency()
    temperature = simulation_config["temperature"]
    language = simulation_config.get("language", "it")

    _log(f"[MC] Starting {total} runs (concurrency={concurrency})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_one, graph, case_data,
                jp, ap, rn, temperature, language, i, total,
            ): i
            for i, (jp, ap, rn) in enumerate(combinations, 1)
        }
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            if outcome["status"] == "ok":
                results.append(outcome["result"])
            else:
                failed.append({"run_id": outcome["run_id"], "error": outcome.get("error", "unknown")})

    total_time = time.time() - sim_start
    _log(f"[MC] Done in {total_time:.0f}s — {len(results)}/{total} succeeded, {len(failed)} failed")
    if failed:
        _log(f"[MC] Failed runs:")
        for f in failed:
            _log(f"[MC]   {f['run_id']}: {f['error'][:100]}")

    return results
