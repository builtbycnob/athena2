# src/athena/api/pipeline.py
"""Core pipeline extracted from cli.py — pure logic, no file I/O.

Usage:
    case_data = prepare_case_data(raw_yaml)
    sim_config_dict = prepare_sim_config(raw_yaml)
    result = run_pipeline(case_data, sim_config_dict, options, progress_callback)
    write_pipeline_outputs(result, output_dir)
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from athena.api.models import PipelineOptions, PipelineResult, ProgressEvent
from athena.schemas.simulation import SimulationConfig, migrate_simulation_v1


def _emit(
    callback: Callable[[ProgressEvent], None] | None,
    stage: str,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Helper to emit a progress event if callback is set."""
    if callback is not None:
        callback(ProgressEvent(stage=stage, message=message, detail=detail))


def prepare_case_data(raw: dict) -> dict:
    """Unwrap, migrate, and normalise a raw case YAML dict.

    Handles:
    - Top-level 'case' key unwrapping
    - id → case_id rename
    - jurisdiction.key_precedents promotion
    - v1 → v2 N-party migration
    """
    from athena.cli import migrate_case_v1

    case_data = raw.get("case", raw)

    if "id" in case_data and "case_id" not in case_data:
        case_data["case_id"] = case_data.pop("id")

    if "key_precedents" not in case_data:
        jur = case_data.get("jurisdiction")
        if isinstance(jur, dict):
            case_data["key_precedents"] = jur.get("key_precedents", [])

    case_data = migrate_case_v1(case_data)
    return case_data


def prepare_sim_config(raw: dict) -> dict:
    """Unwrap and validate a raw simulation YAML dict.

    Returns a validated dict (via SimulationConfig round-trip).
    """
    sim_data = raw.get("simulation", raw)
    sim_data = migrate_simulation_v1(sim_data)
    sim_config = SimulationConfig(**sim_data)
    return sim_config.model_dump()


def run_pipeline(
    case_data: dict,
    sim_config: dict,
    options: PipelineOptions | None = None,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
) -> PipelineResult:
    """Run the full ATHENA pipeline: simulate → aggregate → game theory → meta-agents → outputs.

    This is a pure-logic function: it takes parsed dicts, returns a PipelineResult,
    and never writes files. File I/O is handled by write_pipeline_outputs().

    Args:
        case_data: Prepared case dict (from prepare_case_data).
        sim_config: Validated simulation config dict (from prepare_sim_config).
        options: Pipeline options (concurrency, KG, meta-agents).
        progress_callback: Optional callback for progress events.

    Returns:
        PipelineResult with all outputs.
    """
    from athena.agents.llm import get_stats
    from athena.output.decision_tree import generate_decision_tree
    from athena.output.memo import generate_strategic_memo
    from athena.output.table import format_probability_table
    from athena.simulation.aggregator import aggregate_results
    from athena.simulation.orchestrator import run_monte_carlo

    opts = options or PipelineOptions()
    started_at = datetime.utcnow()

    # --- Concurrency ---
    if opts.concurrency is not None:
        os.environ["ATHENA_CONCURRENCY"] = str(opts.concurrency)

    # --- KG init ---
    kg_active = False
    if opts.kg_enabled:
        os.environ["ATHENA_KG_ENABLED"] = "1"
        try:
            from athena.knowledge.config import get_driver
            get_driver()
            kg_active = True
            _emit(progress_callback, "kg", "Knowledge graph connected")
        except Exception as e:
            os.environ["ATHENA_KG_ENABLED"] = "0"
            _emit(progress_callback, "kg", f"KG unavailable: {e}")

    # --- KG case ingestion ---
    if kg_active:
        try:
            from athena.knowledge import ingest_case
            counts = ingest_case(case_data)
            _emit(progress_callback, "kg", f"Case ingested: {counts['nodes']} nodes, {counts['edges']} edges")
        except Exception as e:
            _emit(progress_callback, "kg", f"Case ingestion failed: {e}")

    # --- Simulate ---
    sim = SimulationConfig(**sim_config)
    _emit(progress_callback, "simulation", f"Starting Monte Carlo ({sim.total_runs} runs)")

    results = run_monte_carlo(case_data, sim_config)
    _emit(progress_callback, "simulation", f"Completed: {len(results)}/{sim.total_runs} runs")

    # --- Aggregate ---
    _emit(progress_callback, "aggregation", "Aggregating results")
    aggregated = aggregate_results(results, sim.total_runs)

    # --- Game theory ---
    game_analysis = None
    if not opts.skip_game_theory and "stakes" in case_data:
        _emit(progress_callback, "game_theory", "Running game theory analysis")
        from athena.game_theory import analyze as gt_analyze
        game_analysis = gt_analyze(aggregated, case_data, results)

    # --- KG: store aggregation + game theory ---
    if kg_active:
        try:
            from athena.knowledge import store_aggregation, store_game_theory
            store_aggregation(case_data["case_id"], aggregated)
            if game_analysis:
                store_game_theory(case_data["case_id"], game_analysis)
        except Exception:
            pass

    # --- KG: post-analysis ---
    kg_post = None
    if kg_active:
        try:
            from athena.knowledge import get_post_analysis
            kg_post = get_post_analysis(case_data["case_id"])
        except Exception:
            pass

    # --- Meta-agents ---
    red_team_output = None
    game_theorist_output = None
    irac_output = None

    if not opts.skip_meta_agents:
        _emit(progress_callback, "meta_agents", "Running red team analysis")
        try:
            from athena.agents.meta_agents import run_red_team
            red_team_output = run_red_team(
                aggregated, case_data,
                game_analysis=game_analysis, kg_insights=kg_post,
            )
        except Exception as e:
            _emit(progress_callback, "meta_agents", f"Red team failed: {e}")

        if game_analysis is not None:
            _emit(progress_callback, "meta_agents", "Running game theorist analysis")
            try:
                from athena.agents.meta_agents import run_game_theorist
                game_theorist_output = run_game_theorist(
                    aggregated, case_data, game_analysis,
                )
            except Exception as e:
                _emit(progress_callback, "meta_agents", f"Game theorist failed: {e}")

        _emit(progress_callback, "meta_agents", "Running IRAC extraction")
        try:
            from athena.agents.meta_agents import run_irac_extraction
            irac_output = run_irac_extraction(results, case_data)
        except Exception as e:
            _emit(progress_callback, "meta_agents", f"IRAC extraction failed: {e}")

    # --- KG: store IRAC ---
    if kg_active and irac_output and irac_output.get("irac_analyses"):
        try:
            from athena.knowledge import store_irac
            store_irac(case_data["case_id"], irac_output)
        except Exception:
            pass

    # --- Generate text outputs ---
    _emit(progress_callback, "outputs", "Generating probability table")
    table_md = format_probability_table(aggregated)

    _emit(progress_callback, "outputs", "Generating decision tree")
    tree_txt = generate_decision_tree(aggregated)

    gt_summary_md = None
    if game_analysis is not None:
        from athena.output.game_theory_summary import format_game_theory_summary
        gt_summary_md = format_game_theory_summary(game_analysis)

    _emit(progress_callback, "outputs", "Generating strategic memo")
    try:
        memo_md = generate_strategic_memo(
            aggregated, case_data, game_analysis=game_analysis, kg_insights=kg_post,
            red_team_output=red_team_output, game_theorist_output=game_theorist_output,
            irac_output=irac_output,
        )
    except Exception as e:
        memo_md = f"# Strategic Memo\n\nMemo generation failed: {e}\n"

    # --- Stats ---
    stats = get_stats()
    completed_at = datetime.utcnow()

    # Serialise game_analysis if it's a Pydantic model
    game_analysis_dict = None
    if game_analysis is not None:
        game_analysis_dict = (
            game_analysis.model_dump()
            if hasattr(game_analysis, "model_dump")
            else game_analysis
        )

    _emit(progress_callback, "done", "Pipeline complete")

    return PipelineResult(
        case_id=case_data.get("case_id", "unknown"),
        results=results,
        aggregated=aggregated,
        game_analysis=game_analysis_dict,
        red_team=red_team_output,
        game_theorist=game_theorist_output,
        irac=irac_output,
        memo=memo_md,
        table_md=table_md,
        tree_txt=tree_txt,
        gt_summary_md=gt_summary_md,
        stats=stats,
        started_at=started_at,
        completed_at=completed_at,
    )


def write_pipeline_outputs(result: PipelineResult, output_dir: str) -> list[str]:
    """Write all pipeline outputs to disk. Returns list of written file paths."""
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []

    def _write(filename: str, content: str) -> None:
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        written.append(path)

    def _write_json(filename: str, data: Any) -> None:
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        written.append(path)

    # Always written
    _write("probability_table.md", result.table_md)
    _write("decision_tree.txt", result.tree_txt)
    _write("strategic_memo.md", result.memo)
    _write_json("raw_results.json", result.results)

    # Conditional
    if result.game_analysis is not None:
        _write_json("game_theory.json", result.game_analysis)
    if result.gt_summary_md is not None:
        _write("game_theory_summary.md", result.gt_summary_md)
    if result.red_team:
        _write_json("red_team.json", result.red_team)
    if result.game_theorist:
        _write_json("game_theorist_agent.json", result.game_theorist)
    if result.irac and result.irac.get("irac_analyses"):
        _write_json("irac_analysis.json", result.irac)

    return written
