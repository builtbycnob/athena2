# src/athena/validation/dataset_fetcher.py
"""Download and filter cases from the Swiss Judgment Prediction dataset (HuggingFace).

Requires: pip install athena[validation]  (installs `datasets` library)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def fetch_swiss_cases(
    *,
    split: str = "train",
    legal_area: str | None = "civil_law",
    min_year: int = 2000,
    max_words: int = 2000,
    n_rejection: int = 5,
    n_approval: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Fetch and filter Swiss Judgment Prediction cases (Italian subset).

    Args:
        split: Dataset split ("train", "test", "validation").
        legal_area: Filter by legal_area (None = all areas).
        min_year: Minimum decision year.
        max_words: Maximum word count in text field.
        n_rejection: Number of label=0 (dismissal) cases to select.
        n_approval: Number of label=1 (approval) cases to select.
        seed: Random seed for sampling.

    Returns:
        List of dicts with keys: id, text, label, legal_area, year, region, canton, language.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for fetching validation cases. "
            "Install with: pip install athena[validation]"
        )

    ds = load_dataset("rcds/swiss_judgment_prediction", "it", split=split,
                       trust_remote_code=True)

    # Dataset uses "legal area" (with space) as column name
    LEGAL_AREA_COL = "legal area"

    # Normalize legal_area filter: accept both "civil_law" and "civil law"
    legal_area_match = legal_area.replace("_", " ") if legal_area else None

    # Filter
    def _filter(example: dict) -> bool:
        if legal_area_match and example.get(LEGAL_AREA_COL) != legal_area_match:
            return False
        if example.get("year", 0) < min_year:
            return False
        text = example.get("text", "")
        if len(text.split()) > max_words:
            return False
        return True

    filtered = ds.filter(_filter)

    # Split by label
    rejections = [r for r in filtered if r["label"] == 0]
    approvals = [r for r in filtered if r["label"] == 1]

    # Deterministic sample
    import random
    rng = random.Random(seed)
    selected_rejections = rng.sample(rejections, min(n_rejection, len(rejections)))
    selected_approvals = rng.sample(approvals, min(n_approval, len(approvals)))

    records = []
    for r in selected_rejections + selected_approvals:
        records.append({
            "id": str(r["id"]),
            "text": r["text"],
            "label": r["label"],
            "legal_area": r.get(LEGAL_AREA_COL, "unknown"),
            "year": r.get("year"),
            "region": r.get("region"),
            "canton": r.get("canton"),
            "language": r.get("language", "it"),
        })

    return records


def fetch_swiss_stratified(
    *,
    split: str = "train",
    min_year: int = 2000,
    max_words: int = 2000,
    per_area: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Fetch stratified Swiss cases across legal areas (Phase 2).

    Args:
        per_area: Dict of {legal_area: n_cases}. Default: 15 civil, 15 penal, 10 public, 10 social.
    """
    if per_area is None:
        per_area = {
            "civil law": 15,
            "penal law": 15,
            "public law": 10,
            "social law": 10,
        }

    all_records: list[dict[str, Any]] = []
    for area, n_total in per_area.items():
        n_half = n_total // 2
        records = fetch_swiss_cases(
            split=split,
            legal_area=area,
            min_year=min_year,
            max_words=max_words,
            n_rejection=n_half,
            n_approval=n_total - n_half,
            seed=seed,
        )
        all_records.extend(records)

    return all_records


def save_fetched_records(records: list[dict], output_dir: str | Path) -> Path:
    """Save fetched records to a JSON file for offline use."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fetched_records.json"
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    return path


def load_fetched_records(path: str | Path) -> list[dict]:
    """Load previously fetched records from JSON."""
    return json.loads(Path(path).read_text())
