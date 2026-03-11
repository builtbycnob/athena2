# src/athena/game_theory/equilibrium.py
"""BATNA computation and Nash bargaining for bilateral settlement."""

from athena.game_theory.schemas import BATNA, SettlementRange
from athena.game_theory.valuation import compute_outcome_values, compute_status_quo


def compute_weighted_probabilities(
    probability_table: dict,
    strategy: str | None = None,
    judge_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Outcome probabilities weighted across judge profiles.

    Args:
        probability_table: Dict with (judge_id, style) tuple keys.
        strategy: If set, filter to only this appellant style.
        judge_weights: Weights per judge_id. Uniform if None.

    Returns:
        Dict with p_rejection, p_annulment, p_reclassification.
    """
    # Filter entries
    entries = []
    for (judge, style), entry in probability_table.items():
        if strategy is not None and style != strategy:
            continue
        entries.append((judge, entry))

    if not entries:
        return {"p_rejection": 0.0, "p_annulment": 0.0, "p_reclassification": 0.0}

    # Build weights
    if judge_weights is None:
        # Uniform across unique judge profiles in filtered entries
        judges = list({j for j, _ in entries})
        judge_weights = {j: 1.0 / len(judges) for j in judges}

    # Aggregate per judge first (average across strategies within same judge)
    from collections import defaultdict
    by_judge: dict[str, list[dict]] = defaultdict(list)
    for judge, entry in entries:
        by_judge[judge].append(entry)

    total_weight = sum(judge_weights.get(j, 0) for j in by_judge)
    if total_weight == 0:
        return {"p_rejection": 0.0, "p_annulment": 0.0, "p_reclassification": 0.0}

    result = {"p_rejection": 0.0, "p_annulment": 0.0, "p_reclassification": 0.0}
    for judge, judge_entries in by_judge.items():
        w = judge_weights.get(judge, 0) / total_weight
        n = len(judge_entries)
        for outcome_key in result:
            avg = sum(e[outcome_key] for e in judge_entries) / n
            result[outcome_key] += w * avg

    return result


def _ev_from_probs(
    probs: dict[str, float], stakes: dict, perspective: str,
    litigation_cost: float | None = None,
) -> float:
    """Compute expected value given outcome probabilities."""
    values = compute_outcome_values(stakes, perspective, litigation_cost)
    return sum(
        probs.get(f"p_{outcome}", 0.0) * val.net_value
        for outcome, val in values.items()
    )


def _ev_range_from_cis(
    probability_table: dict, stakes: dict, perspective: str,
    strategy: str | None = None,
) -> tuple[float, float]:
    """Compute EV range using CI bounds (best/worst case)."""
    values = compute_outcome_values(stakes, perspective)

    # Collect all CI bounds for filtered entries
    entries = []
    for (judge, style), entry in probability_table.items():
        if strategy is not None and style != strategy:
            continue
        entries.append(entry)

    if not entries:
        return (0.0, 0.0)

    # For each entry, compute EV at CI low/high for each outcome
    evs = []
    for entry in entries:
        for outcome in ["rejection", "annulment", "reclassification"]:
            ci = entry.get(f"ci_{outcome}", (0, 0))
            for p_bound in ci:
                # Rough: use this single probability with others at point estimate
                probs = {
                    f"p_{outcome}": p_bound,
                }
                # Fill others with point estimates
                for other in ["rejection", "annulment", "reclassification"]:
                    if other != outcome:
                        probs[f"p_{other}"] = entry[f"p_{other}"]
                ev = sum(probs[f"p_{o}"] * values[o].net_value for o in values)
                evs.append(ev)

    return (min(evs), max(evs)) if evs else (0.0, 0.0)


def compute_batna(
    probability_table: dict,
    stakes: dict,
    perspective: str,
    strategy: str | None = None,
) -> BATNA:
    """Compute Best Alternative To Negotiated Agreement.

    BATNA = expected value of litigating = sum(P(outcome_i) * V(outcome_i)).
    """
    probs = compute_weighted_probabilities(probability_table, strategy=strategy)
    ev = _ev_from_probs(probs, stakes, perspective)
    ev_range = _ev_range_from_cis(probability_table, stakes, perspective, strategy)

    # Find best strategy
    best_strategy = None
    if strategy is None:
        ev_by_strat = compute_ev_by_strategy(probability_table, stakes, perspective)
        if ev_by_strat:
            best_strategy = max(ev_by_strat, key=ev_by_strat.get)
    else:
        best_strategy = strategy

    # Strip p_ prefix for outcome_probabilities
    outcome_probs = {k.replace("p_", ""): v for k, v in probs.items()}

    party_id = "appellant" if perspective == "appellant" else "respondent"
    return BATNA(
        party_id=party_id,
        expected_value=ev,
        expected_value_range=ev_range,
        best_strategy=best_strategy,
        outcome_probabilities=outcome_probs,
    )


def compute_ev_by_strategy(
    probability_table: dict,
    stakes: dict,
    perspective: str,
) -> dict[str, float]:
    """EV for each appellant strategy."""
    strategies = {style for _, style in probability_table}
    result = {}
    for strat in sorted(strategies):
        probs = compute_weighted_probabilities(probability_table, strategy=strat)
        result[strat] = _ev_from_probs(probs, stakes, perspective)
    return result


def compute_settlement_range(
    batna_appellant: BATNA,
    batna_respondent: BATNA,
    sq_appellant: float,
    sq_respondent: float,
) -> SettlementRange:
    """Nash bargaining for bilateral settlement.

    The settlement is a transfer T from respondent to appellant (or negative = appellant pays).
    Appellant accepts if: T >= batna_appellant.expected_value (litigation EV)
    Respondent accepts if: -T >= batna_respondent.expected_value (their litigation EV)
    i.e., T <= -batna_respondent.expected_value

    But we compare against status quo (disagreement point = not litigating at all).
    Appellant's reservation: max(BATNA, status_quo)
    Respondent's reservation: min(-BATNA, -status_quo) as transfer cap

    ZOPA exists when appellant's minimum acceptable <= respondent's maximum acceptable.
    Nash solution (symmetric bargaining power): midpoint of ZOPA.
    """
    # Appellant's minimum acceptable settlement (from their perspective, net value)
    # They'll accept settlement if it's at least as good as their best outside option
    appellant_reservation = max(batna_appellant.expected_value, sq_appellant)

    # Respondent's maximum they'd pay in settlement
    # They prefer settling if the payment is less than what litigation would cost them
    # Respondent BATNA is their EV from litigation (could be positive or negative)
    respondent_max_payment = -min(batna_respondent.expected_value, sq_respondent)

    # ZOPA: [appellant_reservation, respondent_max_payment]
    if appellant_reservation <= respondent_max_payment:
        zopa = (appellant_reservation, respondent_max_payment)
        nash = (appellant_reservation + respondent_max_payment) / 2
        surplus = respondent_max_payment - appellant_reservation
        return SettlementRange(
            zopa=zopa,
            nash_solution=nash,
            surplus=surplus,
            settlement_exists=True,
        )
    else:
        return SettlementRange(
            zopa=None,
            nash_solution=None,
            surplus=appellant_reservation - respondent_max_payment,
            settlement_exists=False,
        )
