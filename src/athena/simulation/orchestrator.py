# src/athena/simulation/orchestrator.py
"""Monte Carlo orchestrator for running simulations across all parameter combinations.

Iterates over all (judge_profile × appellant_profile × N runs) combinations,
invokes run_single (the LangGraph graph) for each, and collects results.
"""

import itertools
from typing import Any

from athena.simulation.graph import build_graph


def run_monte_carlo(case_data: dict, simulation_config: dict) -> list[dict]:
    """Execute all simulation combinations and collect results.

    Args:
        case_data: Parsed case file data.
        simulation_config: Simulation configuration containing:
            - judge_profiles: list of judge profile dicts (each with 'id' key)
            - appellant_profiles: list of appellant profile dicts (each with 'id' key)
            - runs_per_combination: int, number of runs per (judge, appellant) pair
            - temperature: dict with per-role temperature settings
            - language: str, output language (default 'it')

    Returns:
        List of result dicts, one per successful run, each containing:
            run_id, judge_profile, appellant_profile, and run output fields.
    """
    combinations = list(itertools.product(
        simulation_config["judge_profiles"],
        simulation_config["appellant_profiles"],
        range(simulation_config["runs_per_combination"]),
    ))

    results: list[dict] = []
    graph = build_graph()

    for judge_profile, appellant_profile, run_n in combinations:
        run_id = f"{judge_profile['id']}__{appellant_profile['id']}__{run_n:03d}"

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

            if final_state.get("error"):
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
        except Exception:
            # Run failed — skip and count as failed
            continue

    return results
