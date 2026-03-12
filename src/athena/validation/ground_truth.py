# src/athena/validation/ground_truth.py
"""Ground truth schema and persistence for validation cases."""

import json
from pathlib import Path

from pydantic import BaseModel


class GroundTruth(BaseModel):
    case_id: str
    source: str  # "swiss_judgment_prediction" | "ecthr" | "manual"
    outcome: str  # "rejection" | "annulment"
    outcome_raw: int  # Original label (0=dismissal, 1=approval)
    extraction_confidence: str  # "high" | "medium" | "low"
    legal_area: str | None = None
    year: int | None = None
    canton: str | None = None
    region: str | None = None


# Swiss Judgment Prediction: label=0 → dismissal → rejection, label=1 → approval → annulment
SWISS_LABEL_MAP = {0: "rejection", 1: "annulment"}


def save_ground_truth(gt: GroundTruth, ground_truth_dir: str | Path) -> Path:
    """Save ground truth to JSON file."""
    gt_dir = Path(ground_truth_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)
    path = gt_dir / f"{gt.case_id}.json"
    path.write_text(json.dumps(gt.model_dump(), indent=2, ensure_ascii=False))
    return path


def load_ground_truths(ground_truth_dir: str | Path) -> dict[str, GroundTruth]:
    """Load all ground truths from a directory. Returns {case_id: GroundTruth}."""
    gt_dir = Path(ground_truth_dir)
    if not gt_dir.exists():
        return {}
    result = {}
    for path in sorted(gt_dir.glob("*.json")):
        data = json.loads(path.read_text())
        gt = GroundTruth(**data)
        result[gt.case_id] = gt
    return result
