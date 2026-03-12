# src/athena/agents/meta_agents.py
"""Meta-agent entry points: Red Team and Game Theorist.

Post-processing agents that run AFTER aggregation + game theory,
producing structured output via invoke_llm (JSON repair + retry + Langfuse).
"""

import json

from athena.agents.llm import invoke_llm
from athena.agents.meta_prompts import RED_TEAM_SYSTEM_PROMPT, GAME_THEORIST_SYSTEM_PROMPT
from athena.schemas.meta_output import RED_TEAM_SCHEMA, GAME_THEORIST_SCHEMA


def _format_probability_summary(aggregated: dict) -> str:
    """Format probability table as readable text for prompts."""
    lines = []
    for (judge, style), entry in aggregated.get("probability_table", {}).items():
        lines.append(
            f"- {judge} / {style}: "
            f"annullamento={entry['p_annulment']:.0%}, "
            f"riqualificazione={entry['p_reclassification']:.0%}, "
            f"rigetto={entry['p_rejection']:.0%} (n={entry['n_runs']})"
        )
    return "\n".join(lines) if lines else "Nessun dato disponibile."


def _format_argument_effectiveness(aggregated: dict) -> str:
    """Format argument effectiveness as readable text for prompts."""
    lines = []
    for arg_id, data in aggregated.get("argument_effectiveness", {}).items():
        lines.append(
            f"- {arg_id}: persuasività media={data['mean_persuasiveness']:.2f}, "
            f"std={data['std_persuasiveness']:.2f}, "
            f"tasso determinativo={data['determinative_rate']:.0%}, "
            f"n={data['n_evaluations']}"
        )
    return "\n".join(lines) if lines else "Nessun dato disponibile."


def _format_dominated_strategies(aggregated: dict) -> str:
    """Format dominated strategies."""
    dominated = aggregated.get("dominated_strategies", [])
    if dominated:
        return f"Strategie dominate: {', '.join(dominated)}"
    return "Nessuna strategia dominata."


def _build_red_team_user_prompt(
    aggregated: dict, case_data: dict,
    game_analysis=None, kg_insights=None,
) -> str:
    """Build the user prompt for the red team agent."""
    sections = []

    # Case info
    case_info = case_data.get("case", case_data)
    sections.append("## Dati del caso")
    sections.append(f"```json\n{json.dumps(case_info, indent=2, ensure_ascii=False)}\n```")

    # Argument effectiveness (primary data)
    sections.append("\n## Efficacia argomenti")
    sections.append(_format_argument_effectiveness(aggregated))

    # Probability table
    sections.append("\n## Tabella probabilistica")
    sections.append(_format_probability_summary(aggregated))

    # Dominated strategies
    sections.append(f"\n## {_format_dominated_strategies(aggregated)}")

    # Seed arguments
    seed_args = case_data.get("seed_arguments", {}).get("by_party", {})
    if seed_args:
        sections.append("\n## Seed arguments")
        sections.append(f"```json\n{json.dumps(seed_args, indent=2, ensure_ascii=False)}\n```")

    # Game theory highlights
    if game_analysis is not None:
        sections.append("\n## Highlights teoria dei giochi")
        batna = game_analysis.batna if hasattr(game_analysis, "batna") else game_analysis.get("batna", {})
        if isinstance(batna, dict):
            for pid, b in batna.items():
                ev = b.expected_value if hasattr(b, "expected_value") else b.get("expected_value", 0)
                sections.append(f"- BATNA {pid}: {ev:.2f} EUR")
        ev_by_strat = (
            game_analysis.expected_value_by_strategy
            if hasattr(game_analysis, "expected_value_by_strategy")
            else game_analysis.get("expected_value_by_strategy", {})
        )
        if ev_by_strat:
            sections.append("- EV per strategia:")
            for strat, ev in sorted(ev_by_strat.items(), key=lambda x: x[1], reverse=True):
                sections.append(f"  - {strat}: {ev:.2f} EUR")

    # KG determinative arguments
    if kg_insights:
        det_args = kg_insights.get("determinative_arguments", [])
        if det_args:
            sections.append("\n## Argomenti determinativi (Knowledge Graph)")
            for da in det_args[:10]:
                sections.append(
                    f"- {da.get('argument_id', '?')}: {da.get('claim', '')!r} "
                    f"(determinativo {da.get('times_determinative', 0)}/"
                    f"{da.get('total_evaluations', 0)} volte)"
                )

    return "\n".join(sections)


def _build_game_theorist_user_prompt(
    aggregated: dict, case_data: dict, game_analysis,
) -> str:
    """Build the user prompt for the game theorist agent."""
    sections = []

    # Case info with stakes
    case_info = case_data.get("case", case_data)
    sections.append("## Dati del caso")
    sections.append(f"```json\n{json.dumps(case_info, indent=2, ensure_ascii=False)}\n```")

    # Full game analysis
    sections.append("\n## Analisi di teoria dei giochi (completa)")
    if hasattr(game_analysis, "model_dump"):
        ga_dict = game_analysis.model_dump()
    else:
        ga_dict = game_analysis
    sections.append(f"```json\n{json.dumps(ga_dict, indent=2, ensure_ascii=False)}\n```")

    # Probability summary
    sections.append("\n## Tabella probabilistica")
    sections.append(_format_probability_summary(aggregated))

    # Dominated strategies
    sections.append(f"\n## {_format_dominated_strategies(aggregated)}")

    # Run stats
    sections.append(f"\n## Statistiche simulazione")
    sections.append(f"Run totali: {aggregated.get('total_runs', 0)}")
    sections.append(f"Run falliti: {aggregated.get('failed_runs', 0)}")

    return "\n".join(sections)


def run_red_team(
    aggregated: dict, case_data: dict,
    game_analysis=None, kg_insights=None,
) -> dict:
    """Run red team adversarial analysis. Returns structured output dict."""
    system = RED_TEAM_SYSTEM_PROMPT.format(n_runs=aggregated.get("total_runs", 0))
    user = _build_red_team_user_prompt(aggregated, case_data, game_analysis, kg_insights)
    return invoke_llm(system, user, temperature=0.6, max_tokens=4096, json_schema=RED_TEAM_SCHEMA)


def run_game_theorist(
    aggregated: dict, case_data: dict, game_analysis,
) -> dict:
    """Interpret game theory for the lawyer. Returns structured output dict."""
    system = GAME_THEORIST_SYSTEM_PROMPT.format(n_runs=aggregated.get("total_runs", 0))
    user = _build_game_theorist_user_prompt(aggregated, case_data, game_analysis)
    return invoke_llm(system, user, temperature=0.3, max_tokens=4096, json_schema=GAME_THEORIST_SCHEMA)
