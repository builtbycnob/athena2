# src/athena/game_theory/sensitivity.py
"""Parameter sensitivity analysis with tornado ranking."""

from athena.game_theory.schemas import SensitivityResult
from athena.game_theory.equilibrium import (
    compute_weighted_probabilities,
    _ev_from_probs,
    compute_ev_by_strategy,
)


def find_threshold(
    sweep_values: list[float], ev_values: list[float], reference_ev: float,
) -> float | None:
    """Find parameter value where EV crosses reference_ev via linear interpolation.

    Returns None if no crossing found.
    """
    for i in range(len(ev_values) - 1):
        a, b = ev_values[i], ev_values[i + 1]
        if (a - reference_ev) * (b - reference_ev) <= 0 and a != b:
            # Linear interpolation
            t = (reference_ev - a) / (b - a)
            return sweep_values[i] + t * (sweep_values[i + 1] - sweep_values[i])
    return None


def sensitivity_litigation_cost(
    prob_table: dict,
    stakes: dict,
    cost_range: tuple[float, float] = (0, 5000),
    n_steps: int = 20,
) -> SensitivityResult:
    """Sweep litigation cost and compute appellant EV at each point."""
    base_cost = stakes["litigation_cost_estimate"]
    probs = compute_weighted_probabilities(prob_table)

    sweep = [cost_range[0] + i * (cost_range[1] - cost_range[0]) / n_steps for i in range(n_steps + 1)]
    evs = [_ev_from_probs(probs, stakes, "appellant", litigation_cost=c) for c in sweep]

    # Threshold: where EV crosses status quo
    from athena.game_theory.valuation import compute_status_quo
    sq = compute_status_quo(stakes, "appellant")
    threshold = find_threshold(sweep, evs, sq)

    return SensitivityResult(
        parameter="litigation_cost",
        base_value=base_cost,
        sweep_values=sweep,
        ev_at_each=evs,
        threshold=threshold,
        impact=max(evs) - min(evs) if evs else 0.0,
    )


def sensitivity_rejection_probability(
    prob_table: dict,
    stakes: dict,
    strategy: str,
    p_range: tuple[float, float] = (0, 1),
    n_steps: int = 20,
) -> SensitivityResult:
    """Sweep p_rejection (adjusting others proportionally) for a given strategy."""
    from athena.game_theory.valuation import compute_outcome_values, compute_status_quo

    base_probs = compute_weighted_probabilities(prob_table, strategy=strategy)
    base_p_rej = base_probs["p_rejection"]
    values = compute_outcome_values(stakes, "appellant")

    # Base split of non-rejection probability
    p_non_rej = 1 - base_p_rej
    if p_non_rej > 0:
        ann_share = base_probs["p_annulment"] / p_non_rej
        recl_share = base_probs["p_reclassification"] / p_non_rej
    else:
        ann_share = 0.5
        recl_share = 0.5

    sweep = [p_range[0] + i * (p_range[1] - p_range[0]) / n_steps for i in range(n_steps + 1)]
    evs = []
    for p_rej in sweep:
        remainder = 1 - p_rej
        probs = {
            "p_rejection": p_rej,
            "p_annulment": remainder * ann_share,
            "p_reclassification": remainder * recl_share,
        }
        ev = sum(probs[f"p_{o}"] * values[o].net_value for o in values)
        evs.append(ev)

    sq = compute_status_quo(stakes, "appellant")
    threshold = find_threshold(sweep, evs, sq)

    return SensitivityResult(
        parameter=f"p_rejection ({strategy})",
        base_value=base_p_rej,
        sweep_values=sweep,
        ev_at_each=evs,
        threshold=threshold,
        impact=max(evs) - min(evs) if evs else 0.0,
    )


def sensitivity_fine_amount(
    prob_table: dict,
    stakes: dict,
    fine_range: tuple[float, float] = (100, 1000),
    n_steps: int = 20,
) -> SensitivityResult:
    """Sweep current fine midpoint and recompute appellant EV."""
    from athena.game_theory.valuation import compute_status_quo

    probs = compute_weighted_probabilities(prob_table)
    base_fine = (stakes["current_sanction"]["fine_range"][0] + stakes["current_sanction"]["fine_range"][1]) / 2
    cost = stakes["litigation_cost_estimate"]

    # Alternative fine stays fixed
    alt_fine = (stakes["alternative_sanction"]["fine_range"][0] + stakes["alternative_sanction"]["fine_range"][1]) / 2

    sweep = [fine_range[0] + i * (fine_range[1] - fine_range[0]) / n_steps for i in range(n_steps + 1)]
    evs = []
    for fine_mid in sweep:
        # Recompute appellant net values inline
        v_rejection = -(fine_mid + cost)
        v_annulment = -cost
        v_reclassification = -(alt_fine + cost)
        ev = (
            probs["p_rejection"] * v_rejection
            + probs["p_annulment"] * v_annulment
            + probs["p_reclassification"] * v_reclassification
        )
        evs.append(ev)

    # Status quo also shifts with fine
    # Use base status quo as reference
    sq = compute_status_quo(stakes, "appellant")
    threshold = find_threshold(sweep, evs, sq)

    return SensitivityResult(
        parameter="current_fine",
        base_value=base_fine,
        sweep_values=sweep,
        ev_at_each=evs,
        threshold=threshold,
        impact=max(evs) - min(evs) if evs else 0.0,
    )


def sensitivity_judge_weight(
    prob_table: dict,
    stakes: dict,
    judge_profiles: list[str],
) -> SensitivityResult:
    """For each judge, compute EV when that judge is given weight=1.

    Impact = range of EVs across judges.
    """
    from athena.game_theory.valuation import compute_status_quo

    evs = []
    for jp in judge_profiles:
        weights = {j: (1.0 if j == jp else 0.0) for j in judge_profiles}
        probs = compute_weighted_probabilities(prob_table, judge_weights=weights)
        ev = _ev_from_probs(probs, stakes, "appellant")
        evs.append(ev)

    base_ev = _ev_from_probs(
        compute_weighted_probabilities(prob_table), stakes, "appellant"
    )

    sq = compute_status_quo(stakes, "appellant")
    threshold = find_threshold(
        list(range(len(judge_profiles))), evs, sq
    ) if evs else None

    return SensitivityResult(
        parameter="judge_weight",
        base_value=0.0,  # No single base; uniform is the base
        sweep_values=list(range(len(judge_profiles))),
        ev_at_each=evs,
        threshold=threshold,
        impact=(max(evs) - min(evs)) if evs else 0.0,
    )


def run_all_sensitivity(
    prob_table: dict,
    stakes: dict,
) -> list[SensitivityResult]:
    """Run all sensitivity analyses, return sorted by impact (tornado ranking)."""
    results = []

    results.append(sensitivity_litigation_cost(prob_table, stakes))
    results.append(sensitivity_fine_amount(prob_table, stakes))

    # Per-strategy rejection probability
    strategies = sorted({style for _, style in prob_table})
    for strat in strategies:
        results.append(sensitivity_rejection_probability(prob_table, stakes, strat))

    # Judge weight sensitivity
    judge_profiles = sorted({judge for judge, _ in prob_table})
    if len(judge_profiles) > 1:
        results.append(sensitivity_judge_weight(prob_table, stakes, judge_profiles))

    # Sort by impact descending (tornado ranking)
    results.sort(key=lambda r: r.impact, reverse=True)
    return results
