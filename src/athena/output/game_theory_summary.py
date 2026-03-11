# src/athena/output/game_theory_summary.py
"""Markdown summary of game theory analysis."""

from athena.game_theory.schemas import GameTheoryAnalysis


def format_game_theory_summary(game_analysis: GameTheoryAnalysis) -> str:
    """Markdown summary: EV table, BATNA, settlement, tornado ranking."""
    lines = ["# Analisi di Teoria dei Giochi", ""]

    # EV by strategy
    lines.append("## Expected Value per Strategia (prospettiva opponente)")
    lines.append("")
    lines.append("| Strategia | EV (EUR) |")
    lines.append("|-----------|----------|")
    for strat, ev in sorted(
        game_analysis.expected_value_by_strategy.items(), key=lambda x: x[1], reverse=True,
    ):
        marker = " **" if strat == game_analysis.recommended_strategy else ""
        lines.append(f"| {strat}{marker} | {ev:,.2f} |")
    lines.append("")

    if game_analysis.recommended_strategy:
        lines.append(f"**Strategia raccomandata:** {game_analysis.recommended_strategy}")
        lines.append("")

    # BATNA
    lines.append("## BATNA (Best Alternative To Negotiated Agreement)")
    lines.append("")
    for party_id, batna in game_analysis.batna.items():
        lines.append(f"### {party_id.title()}")
        lines.append(f"- **EV litigio:** {batna.expected_value:,.2f} EUR")
        lines.append(f"- **Range:** [{batna.expected_value_range[0]:,.2f}, {batna.expected_value_range[1]:,.2f}] EUR")
        if batna.best_strategy:
            lines.append(f"- **Miglior strategia:** {batna.best_strategy}")
        lines.append(f"- **Probabilità esiti:** " + ", ".join(
            f"{k}={v:.1%}" for k, v in batna.outcome_probabilities.items()
        ))
        lines.append("")

    # Status quo
    lines.append("## Status Quo (non litigare)")
    lines.append("")
    for party_id, pv in game_analysis.party_valuations.items():
        lines.append(f"- **{party_id.title()}:** {pv.status_quo:,.2f} EUR")
    lines.append("")

    # Settlement
    lines.append("## Analisi Transattiva")
    lines.append("")
    s = game_analysis.settlement
    if s.settlement_exists:
        lines.append(f"- **ZOPA:** [{s.zopa[0]:,.2f}, {s.zopa[1]:,.2f}] EUR")
        lines.append(f"- **Soluzione di Nash:** {s.nash_solution:,.2f} EUR")
        lines.append(f"- **Surplus:** {s.surplus:,.2f} EUR")
    else:
        lines.append("- **ZOPA:** Non esiste — nessun accordo mutuamente vantaggioso")
        lines.append(f"- **Gap:** {s.surplus:,.2f} EUR")
    lines.append("")

    # Sensitivity (tornado)
    lines.append("## Analisi di Sensibilità (ranking tornado)")
    lines.append("")
    lines.append("| Parametro | Impatto EV (EUR) | Soglia |")
    lines.append("|-----------|-----------------|--------|")
    for sr in game_analysis.sensitivity:
        threshold_str = f"{sr.threshold:,.2f}" if sr.threshold is not None else "—"
        lines.append(f"| {sr.parameter} | {sr.impact:,.2f} | {threshold_str} |")
    lines.append("")

    # Metadata
    lines.append("---")
    lines.append(f"*Basato su {game_analysis.analysis_metadata.get('total_runs', 0)} simulazioni, "
                 f"{game_analysis.analysis_metadata.get('n_strategies', 0)} strategie, "
                 f"{game_analysis.analysis_metadata.get('n_judge_profiles', 0)} profili giudice.*")

    return "\n".join(lines)
