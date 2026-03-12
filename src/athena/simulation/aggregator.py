# src/athena/simulation/aggregator.py
"""Statistical aggregator for Monte Carlo simulation results.

Compiles results across all simulation runs into probability tables,
argument effectiveness stats, precedent analysis, and dominated strategy
detection — all with Wilson score confidence intervals.
"""

from collections import defaultdict
import math
import statistics


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportions with small samples."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return (max(0, centre - margin), min(1, centre + margin))


def aggregate_results(results: list[dict], total_expected: int = 60) -> dict:
    """Aggregate Monte Carlo run results with confidence intervals.

    Args:
        results: List of result dicts from successful runs, each containing
            run_id, judge_profile, appellant_profile, and judge_decision.
        total_expected: Total number of runs expected (used to compute failed_runs).

    Returns:
        Dict with keys: probability_table, argument_effectiveness,
        precedent_analysis, total_runs, failed_runs, dominated_strategies.
    """
    by_combination: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_style: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        # Use appellant_profile for backward compat; for N-party, build composite key
        appellant_profile = r.get("appellant_profile", "unknown")
        judge_profile = r.get("judge_profile", "unknown")
        key = (judge_profile, appellant_profile)
        by_combination[key].append(r)
        by_style[appellant_profile].append(r)

    # --- 1. Probability table with CI ---
    probability_table: dict[tuple[str, str], dict] = {}
    for (judge, style), runs in by_combination.items():
        decisions = [r["judge_decision"]["verdict"] for r in runs]
        n = len(decisions)

        n_rejection = sum(
            1 for d in decisions if d["qualification_correct"]
        )
        n_annulment = sum(
            1 for d in decisions
            if not d["qualification_correct"]
            and d.get("if_incorrect", {}).get("consequence") == "annulment"
        )
        n_reclassification = sum(
            1 for d in decisions
            if not d["qualification_correct"]
            and d.get("if_incorrect", {}).get("consequence") == "reclassification"
        )

        ci_low, ci_high = wilson_ci(n_rejection, n)

        probability_table[(judge, style)] = {
            "n_runs": n,
            "p_rejection": n_rejection / n,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_annulment": n_annulment / n,
            "ci_annulment": wilson_ci(n_annulment, n),
            "p_reclassification": n_reclassification / n,
            "ci_reclassification": wilson_ci(n_reclassification, n),
            "ci_rejection": (ci_low, ci_high),
        }

    # --- 2. Argument effectiveness ---
    argument_scores: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        for eval_item in r["judge_decision"].get("argument_evaluation", []):
            argument_scores[eval_item["argument_id"]].append({
                "persuasiveness": eval_item["persuasiveness"],
                "determinative": eval_item["determinative"],
                "judge_profile": r["judge_profile"],
                "appellant_style": r["appellant_profile"],
            })

    argument_effectiveness: dict[str, dict] = {}
    for arg_id, scores in argument_scores.items():
        n_evaluations = len(scores)
        argument_effectiveness[arg_id] = {
            "mean_persuasiveness": statistics.mean(
                s["persuasiveness"] for s in scores
            ),
            "std_persuasiveness": (
                statistics.stdev(s["persuasiveness"] for s in scores)
                if n_evaluations > 1
                else 0.0
            ),
            "determinative_rate": sum(
                1 for s in scores if s["determinative"]
            ) / n_evaluations,
            "n_evaluations": n_evaluations,
            "by_judge_profile": {
                jp: statistics.mean(
                    s["persuasiveness"]
                    for s in scores
                    if s["judge_profile"] == jp
                )
                for jp in set(s["judge_profile"] for s in scores)
            },
        }

    # --- 3. Precedent analysis ---
    precedent_followed: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        for prec_id, analysis in r["judge_decision"].get("precedent_analysis", {}).items():
            precedent_followed[prec_id].append({
                "followed": analysis["followed"],
                "distinguished": analysis["distinguished"],
                "judge_profile": r["judge_profile"],
            })

    precedent_analysis: dict[str, dict] = {}
    for prec_id, analyses in precedent_followed.items():
        n_prec = len(analyses)
        precedent_analysis[prec_id] = {
            "followed_rate": sum(
                1 for a in analyses if a["followed"]
            ) / n_prec,
            "distinguished_rate": sum(
                1 for a in analyses if a["distinguished"]
            ) / n_prec,
            "by_judge_profile": {
                jp: {
                    "followed_rate": sum(
                        1 for a in analyses
                        if a["judge_profile"] == jp and a["followed"]
                    ) / max(
                        sum(1 for a in analyses if a["judge_profile"] == jp), 1
                    )
                }
                for jp in set(a["judge_profile"] for a in analyses)
            },
        }

    # --- 4. Dominated strategy detection ---
    # A style X is dominated if for every judge profile, there exists another
    # style with a strictly higher p_rejection (i.e., X is never the best).
    all_judge_profiles = set()
    all_styles = set()
    for judge, style in probability_table:
        all_judge_profiles.add(judge)
        all_styles.add(style)

    dominated_strategies: list[str] = []
    for style_a in all_styles:
        is_dominated = True
        for jp in all_judge_profiles:
            # Get style_a's rejection rate for this judge profile
            a_key = (jp, style_a)
            if a_key not in probability_table:
                is_dominated = False
                break
            a_rate = probability_table[a_key]["p_rejection"]
            # Check if any other style has a strictly higher rate
            beaten_by_someone = False
            for style_b in all_styles:
                if style_b == style_a:
                    continue
                b_key = (jp, style_b)
                if b_key not in probability_table:
                    continue
                if probability_table[b_key]["p_rejection"] > a_rate:
                    beaten_by_someone = True
                    break
            if not beaten_by_someone:
                is_dominated = False
                break
        if is_dominated and len(all_styles) > 1:
            dominated_strategies.append(style_a)

    return {
        "probability_table": probability_table,
        "argument_effectiveness": argument_effectiveness,
        "precedent_analysis": precedent_analysis,
        "total_runs": len(results),
        "failed_runs": total_expected - len(results),
        "dominated_strategies": sorted(dominated_strategies),
    }
