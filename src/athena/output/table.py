# src/athena/output/table.py
"""Formats aggregated simulation results as a markdown probability table.

Output format matches the design spec (section 3.7, Output 1):
each cell shows A:xx% [ci] R:xx% [ci] X:xx% [ci] for annulment,
reclassification, and rejection respectively.
"""


def format_probability_table(aggregated: dict) -> str:
    """Format the probability table as a markdown table.

    Args:
        aggregated: Dict from aggregate_results() with probability_table key.

    Returns:
        Markdown-formatted string with the probability table.
    """
    prob_table = aggregated["probability_table"]

    if not prob_table:
        return (
            "| Profilo Giudice | (nessun dato) |\n"
            "|-----------------|---------------|\n"
            "\nNessun risultato disponibile.\n"
        )

    # Extract unique judge profiles and styles, preserving order of appearance
    judge_profiles: list[str] = []
    styles: list[str] = []
    seen_jp: set[str] = set()
    seen_st: set[str] = set()
    for judge, style in sorted(prob_table.keys()):
        if judge not in seen_jp:
            judge_profiles.append(judge)
            seen_jp.add(judge)
        if style not in seen_st:
            styles.append(style)
            seen_st.add(style)

    # Build header
    header_cells = ["Profilo Giudice"] + [s.capitalize() for s in styles]
    header = "| " + " | ".join(header_cells) + " |"
    separator = "|" + "|".join("-" * (len(c) + 2) for c in header_cells) + "|"

    rows = []
    for jp in judge_profiles:
        cells = [jp]
        for style in styles:
            key = (jp, style)
            if key not in prob_table:
                cells.append("—")
                continue
            entry = prob_table[key]
            cell = _format_cell(entry)
            cells.append(cell)
        rows.append("| " + " | ".join(cells) + " |")

    # Legend
    n_per_cell = None
    for entry in prob_table.values():
        n_per_cell = entry.get("n_runs")
        break

    legend_lines = [
        "",
        "A = annullamento, R = riqualificazione, X = rigetto, [CI 95%] = Wilson score interval",
    ]
    if n_per_cell is not None:
        legend_lines.append(
            f"N={n_per_cell} per cella — intervalli ampi, interpretare con cautela"
        )

    return "\n".join([header, separator] + rows + legend_lines) + "\n"


def _format_cell(entry: dict) -> str:
    """Format a single cell with A/R/X percentages and CIs."""
    p_a = entry["p_annulment"]
    ci_a = entry["ci_annulment"]
    p_r = entry["p_reclassification"]
    ci_r = entry["ci_reclassification"]
    p_x = entry["p_rejection"]
    ci_x = entry["ci_rejection"]

    return (
        f"A:{_pct(p_a)} [{_pct(ci_a[0])}-{_pct(ci_a[1])}] "
        f"R:{_pct(p_r)} [{_pct(ci_r[0])}-{_pct(ci_r[1])}] "
        f"X:{_pct(p_x)} [{_pct(ci_x[0])}-{_pct(ci_x[1])}]"
    )


def _pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.0f}%"
