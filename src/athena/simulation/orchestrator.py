# src/athena/simulation/orchestrator.py
"""Monte Carlo orchestrator for running simulations across all parameter combinations.

Supports N-party profile combinations via itertools.product.
Backward compatible with bilateral (appellant × judge) configurations.
Supports concurrent execution via ThreadPoolExecutor.
"""

import concurrent.futures
import itertools
import os
import sys
import time
from typing import Any

from langfuse import observe

from athena.simulation.graph import build_graph_from_phases, build_bilateral_phases


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


def _generate_combinations(sim_config: dict) -> list[dict]:
    """Generate all run parameter combinations for N-party configs.

    Returns list of run_params dicts.
    """
    judge_profiles = sim_config["judge_profiles"]
    party_profile_map = sim_config.get("party_profiles", {})
    temperatures = sim_config.get("temperatures", sim_config.get("temperature", {}))
    models = sim_config.get("models", {})
    language = sim_config.get("language", "it")
    runs_per = sim_config["runs_per_combination"]

    party_ids = sorted(party_profile_map.keys())
    profile_lists = [party_profile_map[pid] for pid in party_ids]

    combinations = []
    for combo in itertools.product(judge_profiles, *profile_lists, range(runs_per)):
        judge_prof, *mid, run_n = combo
        party_profs = {pid: mid[i] for i, pid in enumerate(party_ids)}

        # Build run_id from all profile IDs
        parts = [judge_prof["id"]]
        for pid in party_ids:
            parts.append(party_profs[pid]["id"])
        parts.append(f"{run_n:03d}")
        run_id = "__".join(parts)

        combinations.append({
            "run_id": run_id,
            "judge_profile": {**judge_prof.get("parameters", judge_prof), "id": judge_prof["id"]},
            "party_profiles": {
                pid: prof for pid, prof in party_profs.items()
            },
            "temperatures": temperatures,
            "models": models,
            "language": language,
            # Legacy fields for backward compat with graph nodes
            "temperature": temperatures,
        })

        # Legacy: set appellant_profile for backward compat
        for pid, prof in party_profs.items():
            if prof.get("role_type") == "advocate":
                combinations[-1]["appellant_profile"] = prof.get("parameters", prof)
                break

    return combinations


def _run_one(
    graph,
    case_data: dict,
    run_params: dict,
    run_index: int,
    total: int,
) -> dict:
    """Execute a single simulation run. Returns a status dict; never raises."""
    run_id = run_params["run_id"]
    run_start = time.time()

    _log(f"[MC] Run {run_index}/{total}: {run_id}")

    initial_state = {
        "case": case_data,
        "params": run_params,
        "briefs": {},
        "validations": {},
        "decision": None,
        "decision_validation": None,
        "error": None,
    }

    try:
        final_state = graph.invoke(initial_state)
        elapsed = time.time() - run_start

        if final_state.get("error"):
            _log(f"[MC]   FAIL ({elapsed:.1f}s): {final_state['error']}")
            return {"status": "fail", "run_id": run_id, "error": final_state["error"]}

        # Extract profile IDs for aggregation
        judge_profile_id = run_params.get("judge_profile", {}).get("id", "unknown")
        # For backward compat, extract appellant profile ID
        appellant_profile_id = "unknown"
        for pid, prof in run_params.get("party_profiles", {}).items():
            if prof.get("role_type") == "advocate":
                appellant_profile_id = prof.get("id", pid)
                break
        # Fallback for legacy run_params
        if appellant_profile_id == "unknown":
            appellant_profile_id = run_params.get("appellant_profile", {}).get("id", "unknown")

        # Map GraphState to downstream-compatible result dict
        briefs = final_state.get("briefs", {})
        validations = final_state.get("validations", {})

        # Find party IDs by role from case_data
        appellant_id = next(
            (p["id"] for p in case_data.get("parties", []) if p["role"] == "appellant"),
            "opponente",
        )
        respondent_id = next(
            (p["id"] for p in case_data.get("parties", []) if p["role"] == "respondent"),
            "comune_milano",
        )

        result = {
            "run_id": run_id,
            "judge_profile": judge_profile_id,
            "appellant_profile": appellant_profile_id,
            "party_profiles": {
                pid: prof.get("id", pid)
                for pid, prof in run_params.get("party_profiles", {}).items()
            },
            "appellant_brief": briefs.get(appellant_id),
            "respondent_brief": briefs.get(respondent_id),
            "judge_decision": final_state.get("decision"),
            "validation_warnings": {
                "appellant": (validations.get(appellant_id) or {}).get("warnings", []),
                "respondent": (validations.get(respondent_id) or {}).get("warnings", []),
                "judge": (final_state.get("decision_validation") or {}).get("warnings", []),
            },
        }

        # KG: store run result (thread-safe, fail-safe)
        if os.environ.get("ATHENA_KG_ENABLED") == "1":
            try:
                from athena.knowledge import store_run_result
                case_id = case_data.get("case_id", "unknown")
                store_run_result(case_id, result)
            except Exception as e:
                _log(f"[KG] Warning: result ingestion failed: {e}")

        _log(f"[MC]   OK ({elapsed:.1f}s) — {run_id}")
        return {"status": "ok", "run_id": run_id, "elapsed": elapsed, "result": result}

    except Exception as e:
        elapsed = time.time() - run_start
        _log(f"[MC]   EXCEPTION ({elapsed:.1f}s): {e}")
        return {"status": "exception", "run_id": run_id, "error": str(e)}


@observe(name="monte_carlo")
def run_monte_carlo(case_data: dict, simulation_config: dict) -> list[dict]:
    """Execute all simulation combinations and collect results."""
    # Generate combinations using the new N-party system
    if "party_profiles" in simulation_config:
        all_combos = _generate_combinations(simulation_config)
    else:
        # Legacy format: generate from old-style config
        all_combos = []
        for jp, ap, rn in itertools.product(
            simulation_config["judge_profiles"],
            simulation_config.get("appellant_profiles", [{"id": "default"}]),
            range(simulation_config["runs_per_combination"]),
        ):
            run_id = f"{jp['id']}__{ap['id']}__{rn:03d}"
            temperature = simulation_config.get("temperatures",
                            simulation_config.get("temperature", {}))
            all_combos.append({
                "run_id": run_id,
                "judge_profile": jp,
                "appellant_profile": ap,
                "party_profiles": {},
                "temperature": temperature,
                "temperatures": temperature,
                "models": simulation_config.get("models", {}),
                "language": simulation_config.get("language", "it"),
            })

    total = len(all_combos)
    results: list[dict] = []
    failed: list[dict] = []
    # Build generic graph from phases — graph topology is the same for all
    # bilateral runs (temperatures/profiles come from state["params"])
    template_params = all_combos[0] if all_combos else {}
    phases = build_bilateral_phases(case_data, template_params)
    graph = build_graph_from_phases(phases)
    sim_start = time.time()
    concurrency = _get_concurrency()

    # Pre-load RAG embedder on main thread to avoid multiprocessing
    # deadlocks when sentence-transformers loads inside ThreadPoolExecutor
    if os.environ.get("ATHENA_RAG_ENABLED") == "1":
        try:
            from athena.rag.embedder import is_embedder_available
            if is_embedder_available():
                _log("[MC] RAG embedder pre-loaded on main thread")
        except Exception as e:
            _log(f"[MC] RAG embedder pre-load failed (non-fatal): {e}")

    _log(f"[MC] Starting {total} runs (concurrency={concurrency})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_one, graph, case_data, combo, i, total,
            ): i
            for i, combo in enumerate(all_combos, 1)
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
