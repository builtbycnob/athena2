# src/athena/game_theory/valuation.py
"""Map verdict outcomes to monetary values for each party."""

from athena.game_theory.schemas import OutcomeValuation


def _midpoint(range_pair: list | tuple) -> float:
    return (range_pair[0] + range_pair[1]) / 2


def compute_outcome_values(
    stakes: dict,
    perspective: str,
    litigation_cost: float | None = None,
    outcome_space: list[str] | None = None,
) -> dict[str, OutcomeValuation]:
    """Map verdict outcomes to monetary values for a party.

    Args:
        stakes: Case stakes dict with current_sanction, alternative_sanction.
        perspective: "appellant" or "respondent".
        litigation_cost: Override litigation cost (for sensitivity sweeps).
        outcome_space: List of outcome names to include. Defaults to
            ["rejection", "annulment", "reclassification"].

    Returns:
        Dict mapping outcome name to OutcomeValuation.
    """
    if outcome_space is None:
        outcome_space = ["rejection", "annulment", "reclassification"]

    current = stakes["current_sanction"]
    alt = stakes.get("alternative_sanction", current)
    cost = litigation_cost if litigation_cost is not None else stakes["litigation_cost_estimate"]

    fine_mid = _midpoint(current["fine_range"])
    alt_fine_mid = _midpoint(alt["fine_range"])
    current_points = current.get("points_deducted", 0)
    alt_points = alt.get("points_deducted", 0)

    all_outcomes: dict[str, dict] = {}

    if perspective == "appellant":
        all_outcomes = {
            "rejection": OutcomeValuation(
                outcome="rejection",
                description="Appeal rejected — original sanction confirmed",
                fine=fine_mid,
                fine_range=tuple(current["fine_range"]),
                points=current_points,
                net_value=-(fine_mid + cost),
            ),
            "annulment": OutcomeValuation(
                outcome="annulment",
                description="Sanction annulled — no fine",
                fine=0.0,
                fine_range=(0.0, 0.0),
                points=0,
                net_value=-cost,
            ),
            "reclassification": OutcomeValuation(
                outcome="reclassification",
                description="Reclassified to lesser offence",
                fine=alt_fine_mid,
                fine_range=tuple(alt["fine_range"]),
                points=alt_points,
                net_value=-(alt_fine_mid + cost),
            ),
        }
    else:  # respondent
        all_outcomes = {
            "rejection": OutcomeValuation(
                outcome="rejection",
                description="Appeal rejected — sanction upheld, fine collected",
                fine=fine_mid,
                fine_range=tuple(current["fine_range"]),
                points=current_points,
                net_value=fine_mid - cost,
            ),
            "annulment": OutcomeValuation(
                outcome="annulment",
                description="Sanction annulled — fine lost",
                fine=0.0,
                fine_range=(0.0, 0.0),
                points=0,
                net_value=-cost,
            ),
            "reclassification": OutcomeValuation(
                outcome="reclassification",
                description="Reclassified — reduced fine collected",
                fine=alt_fine_mid,
                fine_range=tuple(alt["fine_range"]),
                points=alt_points,
                net_value=alt_fine_mid - cost,
            ),
        }

    return {k: v for k, v in all_outcomes.items() if k in outcome_space}


def compute_status_quo(stakes: dict, perspective: str) -> float:
    """Value of not litigating (no trial costs).

    Appellant: -midpoint(current_fine_range)  (pays fine, no legal fees)
    Respondent: +midpoint(current_fine_range)  (collects fine, no legal fees)
    """
    fine_mid = _midpoint(stakes["current_sanction"]["fine_range"])
    if perspective == "appellant":
        return -fine_mid
    else:
        return fine_mid
