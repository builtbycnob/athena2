# src/athena/game_theory/__init__.py
"""Game theory analysis: BATNA, Nash bargaining, sensitivity analysis.

Public API: analyze(aggregated, case_data, results) -> GameTheoryAnalysis
"""

from athena.game_theory.schemas import GameTheoryAnalysis, PartyValuations
from athena.game_theory.valuation import compute_outcome_values, compute_status_quo
from athena.game_theory.equilibrium import (
    compute_batna,
    compute_ev_by_strategy,
    compute_settlement_range,
)
from athena.game_theory.sensitivity import run_all_sensitivity


def analyze(
    aggregated: dict,
    case_data: dict,
    results: list[dict] | None = None,
) -> GameTheoryAnalysis:
    """Run full game theory analysis on aggregated Monte Carlo results.

    Args:
        aggregated: Dict from aggregate_results().
        case_data: Parsed case YAML dict.
        results: Raw simulation results (unused for now, reserved for future).

    Returns:
        GameTheoryAnalysis with valuations, BATNA, settlement, sensitivity.
    """
    prob_table = aggregated["probability_table"]
    stakes = case_data["stakes"]

    # 1. Outcome valuations for each perspective
    party_valuations = {}
    for perspective in ("appellant", "respondent"):
        outcomes = compute_outcome_values(stakes, perspective)
        sq = compute_status_quo(stakes, perspective)
        party_valuations[perspective] = PartyValuations(
            party_id=perspective,
            outcomes=outcomes,
            litigation_cost=stakes["litigation_cost_estimate"],
            status_quo=sq,
        )

    # 2. EV by strategy (appellant perspective — the decision-maker)
    ev_by_strategy = compute_ev_by_strategy(prob_table, stakes, "appellant")

    # 3. BATNA for each side
    batna_appellant = compute_batna(prob_table, stakes, "appellant")
    batna_respondent = compute_batna(prob_table, stakes, "respondent")
    batna = {
        "appellant": batna_appellant,
        "respondent": batna_respondent,
    }

    # 4. Settlement range
    sq_app = compute_status_quo(stakes, "appellant")
    sq_resp = compute_status_quo(stakes, "respondent")
    settlement = compute_settlement_range(
        batna_appellant, batna_respondent, sq_app, sq_resp,
    )

    # 5. Sensitivity analysis
    sensitivity = run_all_sensitivity(prob_table, stakes)

    # 6. Recommended strategy
    recommended = None
    dominated = set(aggregated.get("dominated_strategies", []))
    if ev_by_strategy:
        viable = {s: ev for s, ev in ev_by_strategy.items() if s not in dominated}
        if viable:
            recommended = max(viable, key=viable.get)

    return GameTheoryAnalysis(
        party_valuations=party_valuations,
        batna=batna,
        settlement=settlement,
        sensitivity=sensitivity,
        expected_value_by_strategy=ev_by_strategy,
        recommended_strategy=recommended,
        analysis_metadata={
            "total_runs": aggregated.get("total_runs", 0),
            "failed_runs": aggregated.get("failed_runs", 0),
            "n_strategies": len(ev_by_strategy),
            "n_judge_profiles": len({j for j, _ in prob_table}),
        },
    )
