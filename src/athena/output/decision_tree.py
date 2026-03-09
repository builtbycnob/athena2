# src/athena/output/decision_tree.py
"""Generates a text-based decision tree from aggregated simulation results.

For each judge profile, recommends the best appellant style (lowest p_rejection,
i.e., highest chance of success for the appellant). Marks dominated strategies
and shows argument effectiveness rankings.
"""

from collections import defaultdict


def generate_decision_tree(aggregated: dict) -> str:
    """Generate a text-based decision tree from aggregated results.

    Args:
        aggregated: Dict from aggregate_results() with probability_table,
            argument_effectiveness, and dominated_strategies.

    Returns:
        Formatted text string with the decision tree.
    """
    prob_table = aggregated["probability_table"]
    arg_effectiveness = aggregated.get("argument_effectiveness", {})
    dominated = aggregated.get("dominated_strategies", [])

    # Group by judge profile
    by_judge: dict[str, dict[str, dict]] = defaultdict(dict)
    for (judge, style), entry in prob_table.items():
        by_judge[judge][style] = entry

    lines: list[str] = []
    lines.append("DECISION TREE — Profilo probabile del giudice?")
    lines.append("=" * 55)
    lines.append("")

    for judge in sorted(by_judge.keys()):
        styles = by_judge[judge]
        lines.append(f"┌─ {judge}")

        # Find best style (lowest p_rejection = best for appellant)
        best_style = min(styles, key=lambda s: styles[s]["p_rejection"])
        best_entry = styles[best_style]

        for style in sorted(styles.keys()):
            entry = styles[style]
            p_success = 1 - entry["p_rejection"]
            ci_succ_low = 1 - entry["ci_rejection"][1]
            ci_succ_high = 1 - entry["ci_rejection"][0]

            marker = " ★ RACCOMANDATO" if style == best_style else ""
            dom_marker = " ✗ DOMINATO" if style in dominated else ""

            lines.append(
                f"│  ├─ {style}: "
                f"successo {p_success * 100:.0f}% "
                f"[{ci_succ_low * 100:.0f}%-{ci_succ_high * 100:.0f}%]"
                f"{marker}{dom_marker}"
            )
            lines.append(
                f"│  │     A:{entry['p_annulment'] * 100:.0f}% "
                f"R:{entry['p_reclassification'] * 100:.0f}% "
                f"X:{entry['p_rejection'] * 100:.0f}%"
            )

        lines.append("│")

    # --- Dominated strategies section ---
    lines.append("")
    if dominated:
        lines.append("STRATEGIE DOMINATE")
        lines.append("-" * 30)
        for style in dominated:
            lines.append(
                f"  ✗ {style} — dominato: per ogni profilo giudice esiste "
                f"una strategia con probabilità di successo superiore."
            )
    else:
        lines.append("Nessuna strategia dominata individuata.")

    # --- Argument effectiveness ranking ---
    lines.append("")
    lines.append("RANKING EFFICACIA ARGOMENTI")
    lines.append("-" * 30)

    sorted_args = sorted(
        arg_effectiveness.items(),
        key=lambda x: x[1]["mean_persuasiveness"],
        reverse=True,
    )
    for rank, (arg_id, stats) in enumerate(sorted_args, 1):
        lines.append(
            f"  {rank}. {arg_id}: "
            f"persuasività media {stats['mean_persuasiveness']:.2f} "
            f"(±{stats['std_persuasiveness']:.2f}), "
            f"determinativo nel {stats['determinative_rate'] * 100:.0f}% dei casi"
        )
        # Show per-profile breakdown
        if stats.get("by_judge_profile"):
            profile_parts = []
            for jp in sorted(stats["by_judge_profile"]):
                profile_parts.append(
                    f"{jp}: {stats['by_judge_profile'][jp]:.2f}"
                )
            lines.append(f"     per profilo: {', '.join(profile_parts)}")

    lines.append("")
    return "\n".join(lines)
