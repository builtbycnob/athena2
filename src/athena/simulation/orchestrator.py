# src/athena/simulation/orchestrator.py
"""Monte Carlo orchestrator for running simulations across all parameter combinations.

Iterates over all (judge_profile × appellant_profile × N runs) combinations,
invokes run_single (the LangGraph graph) for each, and collects results.
"""

import itertools
import sys
import time
from typing import Any

from langfuse import observe

from athena.simulation.graph import build_graph


def _log(msg: str) -> None:
    """Print and flush immediately for real-time progress visibility."""
    print(msg, flush=True)


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

    _log(f"[MC] Starting {total} runs")

    for i, (judge_profile, appellant_profile, run_n) in enumerate(combinations, 1):
        run_id = f"{judge_profile['id']}__{appellant_profile['id']}__{run_n:03d}"
        run_start = time.time()

        _log(f"[MC] Run {i}/{total}: {run_id}")

        initial_state = {
            "case": case_data,
            "params": {
                "run_id": run_id,
                "judge_profile": judge_profile,
                "appellant_profile": appellant_profile,
                "temperature": simulation_config["temperature"],
                "language": simulation_config.get("language", "it"),
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
                failed.append({"run_id": run_id, "error": final_state["error"]})
                continue

            results.append({
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
            })
            _log(f"[MC]   OK ({elapsed:.1f}s) — {len(results)}/{i} succeeded so far")

        except Exception as e:
            elapsed = time.time() - run_start
            _log(f"[MC]   EXCEPTION ({elapsed:.1f}s): {e}")
            failed.append({"run_id": run_id, "error": str(e)})
            continue

    total_time = time.time() - sim_start
    _log(f"[MC] Done in {total_time:.0f}s — {len(results)}/{total} succeeded, {len(failed)} failed")
    if failed:
        _log(f"[MC] Failed runs:")
        for f in failed:
            _log(f"[MC]   {f['run_id']}: {f['error'][:100]}")

    return results
