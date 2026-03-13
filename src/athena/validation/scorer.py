# src/athena/validation/scorer.py
"""Scoring module: accuracy, ECE, log loss, stratified analysis.

Compares ATHENA simulation results against ground truth.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from athena.validation.ground_truth import GroundTruth, load_ground_truths


class CaseScore:
    """Score for a single case."""

    def __init__(
        self,
        case_id: str,
        ground_truth: GroundTruth,
        predicted_outcome: str,
        p_rejection: float,
        p_annulment: float,
    ):
        self.case_id = case_id
        self.ground_truth = ground_truth
        self.predicted_outcome = predicted_outcome
        self.p_rejection = p_rejection
        self.p_annulment = p_annulment
        self.correct = predicted_outcome == ground_truth.outcome

    @property
    def predicted_probability(self) -> float:
        """Probability assigned to the actual outcome."""
        if self.ground_truth.outcome == "rejection":
            return self.p_rejection
        return self.p_annulment


class ValidationReport:
    """Aggregate validation report."""

    def __init__(self, scores: list[CaseScore]):
        self.scores = scores
        self.n = len(scores)

    @property
    def accuracy(self) -> float:
        if self.n == 0:
            return 0.0
        return sum(1 for s in self.scores if s.correct) / self.n

    @property
    def accuracy_ci(self) -> tuple[float, float]:
        """Wilson score 95% CI for accuracy."""
        if self.n == 0:
            return (0.0, 0.0)
        from athena.simulation.aggregator import wilson_ci
        return wilson_ci(sum(1 for s in self.scores if s.correct), self.n)

    @property
    def log_loss(self) -> float:
        """Mean log loss (binary cross-entropy)."""
        if self.n == 0:
            return float("inf")
        total = 0.0
        for s in self.scores:
            p = max(min(s.predicted_probability, 1 - 1e-15), 1e-15)
            total -= math.log(p)
        return total / self.n

    @property
    def ece(self) -> float:
        """Expected Calibration Error (10 bins)."""
        if self.n == 0:
            return 0.0
        n_bins = 10
        bins: list[list[CaseScore]] = [[] for _ in range(n_bins)]
        for s in self.scores:
            # Use predicted probability for the majority class (rejection)
            confidence = max(s.p_rejection, s.p_annulment)
            bin_idx = min(int(confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(s)

        ece = 0.0
        for b in bins:
            if not b:
                continue
            avg_confidence = sum(max(s.p_rejection, s.p_annulment) for s in b) / len(b)
            avg_accuracy = sum(1 for s in b if s.correct) / len(b)
            ece += abs(avg_confidence - avg_accuracy) * len(b) / self.n
        return ece

    def stratify_by(self, key: str) -> dict[str, "ValidationReport"]:
        """Stratify scores by a ground truth attribute (legal_area, year, canton)."""
        groups: dict[str, list[CaseScore]] = defaultdict(list)
        for s in self.scores:
            val = getattr(s.ground_truth, key, None)
            groups[str(val)].append(s)
        return {k: ValidationReport(v) for k, v in groups.items()}

    def error_analysis(self) -> list[dict[str, Any]]:
        """Return details for incorrect predictions."""
        errors = []
        for s in self.scores:
            if not s.correct:
                errors.append({
                    "case_id": s.case_id,
                    "expected": s.ground_truth.outcome,
                    "predicted": s.predicted_outcome,
                    "p_rejection": s.p_rejection,
                    "p_annulment": s.p_annulment,
                    "legal_area": s.ground_truth.legal_area,
                    "year": s.ground_truth.year,
                })
        return errors

    def to_markdown(self) -> str:
        """Generate a markdown validation report."""
        lines = ["# ATHENA Validation Report\n"]

        # Overall metrics
        ci_low, ci_high = self.accuracy_ci
        lines.append("## Overall Metrics\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Cases | {self.n} |")
        lines.append(f"| Accuracy | {self.accuracy:.1%} [{ci_low:.1%}, {ci_high:.1%}] |")
        lines.append(f"| Log Loss | {self.log_loss:.3f} |")
        lines.append(f"| ECE | {self.ece:.3f} |")
        lines.append("")

        # Stratification by legal area
        by_area = self.stratify_by("legal_area")
        if len(by_area) > 1:
            lines.append("## By Legal Area\n")
            lines.append("| Area | N | Accuracy |")
            lines.append("|------|---|----------|")
            for area, report in sorted(by_area.items()):
                lines.append(f"| {area} | {report.n} | {report.accuracy:.1%} |")
            lines.append("")

        # Error analysis
        errors = self.error_analysis()
        if errors:
            lines.append("## Error Analysis\n")
            lines.append(f"| Case | Expected | Predicted | P(rejection) | P(annulment) |")
            lines.append(f"|------|----------|-----------|--------------|--------------|")
            for e in errors:
                lines.append(
                    f"| {e['case_id']} | {e['expected']} | {e['predicted']} "
                    f"| {e['p_rejection']:.2f} | {e['p_annulment']:.2f} |"
                )
            lines.append("")

        return "\n".join(lines)


def score_results(
    results_dir: str | Path,
    ground_truth_dir: str | Path,
) -> ValidationReport:
    """Score ATHENA results against ground truth.

    Expects results_dir to contain subdirectories per case (ch-{id}/),
    each with raw_results.json from an ATHENA run.
    """
    results_dir = Path(results_dir)
    ground_truths = load_ground_truths(ground_truth_dir)

    scores: list[CaseScore] = []
    for gt in ground_truths.values():
        case_dir = results_dir / gt.case_id
        raw_path = case_dir / "raw_results.json"
        if not raw_path.exists():
            continue

        results = json.loads(raw_path.read_text())
        if not results:
            continue

        # Compute aggregate outcome from raw results
        p_rejection, p_annulment = _compute_outcome_probabilities(results)
        predicted = "rejection" if p_rejection >= p_annulment else "annulment"

        scores.append(CaseScore(
            case_id=gt.case_id,
            ground_truth=gt,
            predicted_outcome=predicted,
            p_rejection=p_rejection,
            p_annulment=p_annulment,
        ))

    return ValidationReport(scores)


def _compute_outcome_probabilities(results: list[dict]) -> tuple[float, float]:
    """Compute P(rejection) and P(annulment) from raw ATHENA results.

    Auto-detects verdict schema:
    - Italian: qualification_correct=True → rejection, else → annulment
    - Swiss: appeal_outcome=dismissed → rejection, else → annulment
    """
    n = len(results)
    if n == 0:
        return (0.5, 0.5)

    # Detect schema from first verdict
    first_verdict = results[0].get("judge_decision", {}).get("verdict", {})
    is_swiss = "appeal_outcome" in first_verdict or "lower_court_correct" in first_verdict

    n_rejection = 0
    n_annulment = 0
    for r in results:
        verdict = r.get("judge_decision", {}).get("verdict", {})
        if is_swiss:
            # New two-step schema
            if "lower_court_correct" in verdict:
                if verdict.get("lower_court_correct"):
                    n_rejection += 1
                else:
                    n_annulment += 1
            # Legacy flat schema
            elif verdict.get("appeal_outcome", "dismissed") == "dismissed":
                n_rejection += 1
            else:
                n_annulment += 1
        else:
            if verdict.get("qualification_correct"):
                n_rejection += 1
            else:
                n_annulment += 1

    return (n_rejection / n, n_annulment / n)
