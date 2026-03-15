"""Dataset ingestion pipeline for ATHENA2.

Downloads, validates, and preprocesses Swiss Federal Supreme Court datasets
from HuggingFace. Produces clean Parquet files ready for feature extraction
and model training.

Usage:
    python -m athena2.data.ingestion                    # Download + process all
    python -m athena2.data.ingestion --dataset primary  # Just SJP-XL
    python -m athena2.data.ingestion --stats-only       # Just print statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parents[3] / "configs" / "data.yaml"


def load_config() -> dict[str, Any]:
    """Load data configuration from YAML."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Data Classes ───────────────────────────────────────────────────

@dataclass
class DatasetStats:
    """Statistics for a single dataset."""
    name: str
    total_rows: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    languages: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)
    law_areas: Counter = field(default_factory=Counter)
    years: Counter = field(default_factory=Counter)
    facts_lengths: list[int] = field(default_factory=list)
    considerations_lengths: list[int] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"Dataset: {self.name}",
            f"{'='*60}",
            f"Total rows: {self.total_rows:,}",
        ]

        if self.splits:
            lines.append(f"Splits: {dict(self.splits)}")

        if self.languages:
            lines.append(f"Languages: {dict(self.languages.most_common())}")

        if self.labels:
            lines.append(f"Labels: {dict(self.labels.most_common())}")
            total = sum(self.labels.values())
            for label, count in self.labels.most_common():
                lines.append(f"  {label}: {count:,} ({count/total:.1%})")

        if self.law_areas:
            lines.append(f"Law areas: {dict(self.law_areas.most_common())}")

        if self.years:
            years_sorted = sorted(self.years.keys())
            lines.append(f"Year range: {years_sorted[0]}–{years_sorted[-1]}")

        if self.facts_lengths:
            avg = sum(self.facts_lengths) / len(self.facts_lengths)
            median = sorted(self.facts_lengths)[len(self.facts_lengths) // 2]
            p95 = sorted(self.facts_lengths)[int(len(self.facts_lengths) * 0.95)]
            lines.append(f"Facts length (chars): avg={avg:.0f}, median={median}, p95={p95}")

        if self.considerations_lengths:
            avg = sum(self.considerations_lengths) / len(self.considerations_lengths)
            median = sorted(self.considerations_lengths)[len(self.considerations_lengths) // 2]
            p95 = sorted(self.considerations_lengths)[int(len(self.considerations_lengths) * 0.95)]
            lines.append(f"Considerations length (chars): avg={avg:.0f}, median={median}, p95={p95}")

        return "\n".join(lines)


# ── Text Cleaning ──────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_legal_text(text: str | None) -> str:
    """Clean legal text: remove HTML, normalize whitespace, strip artifacts."""
    if not text:
        return ""

    # Remove HTML tags (some HF records have residual HTML)
    text = _HTML_TAG_RE.sub(" ", text)

    # Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


# ── Ingestion Functions ────────────────────────────────────────────

def ingest_primary(config: dict, output_dir: Path, stats_only: bool = False) -> DatasetStats:
    """Download and process the primary SJP-XL dataset (329K cases).

    Args:
        config: Data configuration dict.
        output_dir: Directory for processed Parquet files.
        stats_only: If True, compute statistics without saving.

    Returns:
        DatasetStats for the dataset.
    """
    import pandas as pd
    from datasets import load_dataset

    ds_config = config["datasets"]["primary"]
    name = ds_config["name"]
    label_map = ds_config["label_map"]

    logger.info(f"Loading {name}...")
    t0 = time.time()
    dataset = load_dataset(name, trust_remote_code=True)
    logger.info(f"Loaded in {time.time()-t0:.1f}s")

    stats = DatasetStats(name=name)

    all_rows = []
    for split_name, split_data in dataset.items():
        stats.splits[split_name] = len(split_data)
        logger.info(f"  Split '{split_name}': {len(split_data):,} rows")

        for row in split_data:
            stats.total_rows += 1

            # Language distribution
            lang = row.get("language", "unknown")
            stats.languages[lang] += 1

            # Label distribution
            label_raw = row.get("label")
            label_str = label_map.get(label_raw, f"unknown_{label_raw}")
            stats.labels[label_str] += 1

            # Law area
            law_area = row.get("law_area", "unknown")
            stats.law_areas[law_area] += 1

            # Year
            year = row.get("year")
            if year:
                stats.years[year] += 1

            # Text lengths
            facts = row.get("facts", "") or ""
            considerations = row.get("considerations", "") or ""
            stats.facts_lengths.append(len(facts))
            if considerations:
                stats.considerations_lengths.append(len(considerations))

            if not stats_only:
                # Clean and store
                cleaned_facts = clean_legal_text(facts)
                cleaned_considerations = clean_legal_text(considerations)

                proc = config.get("processing", {})
                min_facts = proc.get("min_facts_tokens", 50) * 4  # rough char estimate
                if len(cleaned_facts) < min_facts:
                    continue

                all_rows.append({
                    "decision_id": row.get("decision_id", ""),
                    "facts": cleaned_facts,
                    "considerations": cleaned_considerations,
                    "label": label_raw,
                    "label_str": label_str,
                    "law_area": law_area,
                    "law_sub_area": row.get("law_sub_area", ""),
                    "language": lang,
                    "year": year,
                    "court": row.get("court", ""),
                    "chamber": row.get("chamber", ""),
                    "canton": row.get("canton", ""),
                    "region": row.get("region", ""),
                    "split": split_name,
                    "facts_len": len(cleaned_facts),
                    "considerations_len": len(cleaned_considerations),
                })

    if not stats_only and all_rows:
        df = pd.DataFrame(all_rows)

        # Apply temporal split strategy (official SJP-XL: train≤2015, val 2016-17, test≥2018)
        split_config = config.get("splits", {})
        train_max = split_config.get("train_max_year", 2015)
        val_years = set(split_config.get("validation_years", [2016, 2017]))
        test_min = split_config.get("test_min_year", 2018)

        def assign_split(year):
            if year <= train_max:
                return "train"
            elif year in val_years:
                return "validation"
            elif year >= test_min:
                return "test"
            else:
                return "validation"  # gap years default to validation

        df["athena2_split"] = df["year"].apply(assign_split)

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "sjp_xl.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df):,} rows to {out_path}")

        # Split statistics
        for split_val, count in df["athena2_split"].value_counts().items():
            logger.info(f"  athena2_split '{split_val}': {count:,}")

    return stats


def ingest_secondary(config: dict, output_dir: Path, stats_only: bool = False) -> DatasetStats:
    """Download and process the secondary SJP dataset (85K cases, benchmark comparison)."""
    import pandas as pd
    from datasets import load_dataset

    ds_config = config["datasets"]["secondary"]
    name = ds_config["name"]
    label_map = ds_config["label_map"]

    logger.info(f"Loading {name}...")
    t0 = time.time()
    dataset = load_dataset(name, "all+mt", trust_remote_code=True)
    logger.info(f"Loaded in {time.time()-t0:.1f}s")

    stats = DatasetStats(name=name)

    all_rows = []
    for split_name, split_data in dataset.items():
        stats.splits[split_name] = len(split_data)

        for row in split_data:
            stats.total_rows += 1
            lang = row.get("language", "unknown")
            stats.languages[lang] += 1

            label_raw = row.get("label")
            label_str = label_map.get(label_raw, f"unknown_{label_raw}")
            stats.labels[label_str] += 1

            legal_area = row.get("legal_area", "unknown")
            stats.law_areas[legal_area] += 1

            year = row.get("year")
            if year:
                stats.years[year] += 1

            text = row.get("text", "") or ""
            stats.facts_lengths.append(len(text))

            if not stats_only:
                all_rows.append({
                    "id": row.get("id"),
                    "facts": clean_legal_text(text),
                    "label": label_raw,
                    "label_str": label_str,
                    "legal_area": legal_area,
                    "language": lang,
                    "year": year,
                    "region": row.get("region", ""),
                    "canton": row.get("canton", ""),
                    "split": split_name,
                    "facts_len": len(clean_legal_text(text)),
                })

    if not stats_only and all_rows:
        df = pd.DataFrame(all_rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "sjp_benchmark.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df):,} rows to {out_path}")

    return stats


def ingest_criticality(config: dict, output_dir: Path, stats_only: bool = False) -> DatasetStats:
    """Download Swiss criticality prediction dataset (139K cases)."""
    import pandas as pd
    from datasets import load_dataset

    name = config["datasets"]["auxiliary"]["criticality"]["name"]

    logger.info(f"Loading {name}...")
    t0 = time.time()
    dataset = load_dataset(name, trust_remote_code=True)
    logger.info(f"Loaded in {time.time()-t0:.1f}s")

    stats = DatasetStats(name=name)

    all_rows = []
    for split_name, split_data in dataset.items():
        stats.splits[split_name] = len(split_data)

        for row in split_data:
            stats.total_rows += 1
            stats.languages[row.get("language", "unknown")] += 1
            stats.labels[f"bge={row.get('bge_label', '?')}"] += 1

            year = row.get("year")
            if year:
                stats.years[year] += 1

            if not stats_only:
                all_rows.append({
                    "decision_id": row.get("decision_id", ""),
                    "bge_label": row.get("bge_label"),
                    "citation_label": row.get("citation_label"),
                    "law_area": row.get("law_area", ""),
                    "language": row.get("language", ""),
                    "year": year,
                    "chamber": row.get("chamber", ""),
                    "split": split_name,
                })

    if not stats_only and all_rows:
        df = pd.DataFrame(all_rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "criticality.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df):,} rows to {out_path}")

    return stats


def ingest_citation_extraction(config: dict, output_dir: Path, stats_only: bool = False) -> DatasetStats:
    """Download Swiss citation extraction dataset (127K cases, NER tags)."""
    from datasets import load_dataset

    name = config["datasets"]["auxiliary"]["citation"]["name"]

    logger.info(f"Loading {name}...")
    t0 = time.time()
    dataset = load_dataset(name, trust_remote_code=True)
    logger.info(f"Loaded in {time.time()-t0:.1f}s")

    stats = DatasetStats(name=name)

    for split_name, split_data in dataset.items():
        stats.splits[split_name] = len(split_data)
        for row in split_data:
            stats.total_rows += 1
            stats.languages[row.get("language", "unknown")] += 1
            year = row.get("year")
            if year:
                stats.years[year] += 1

    # NER data is token-level — save as-is for citation graph construction
    if not stats_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save each split as Arrow/Parquet
        for split_name, split_data in dataset.items():
            out_path = output_dir / f"citation_extraction_{split_name}.parquet"
            split_data.to_parquet(str(out_path))
            logger.info(f"Saved {split_name} to {out_path}")

    return stats


# ── EDA Report ─────────────────────────────────────────────────────

def generate_eda_report(stats_list: list[DatasetStats], output_path: Path) -> None:
    """Generate comprehensive EDA report as Markdown."""
    lines = [
        "# ATHENA2 — Exploratory Data Analysis Report",
        f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for stats in stats_list:
        lines.append(stats.summary())
        lines.append("")

        # Distribution tables
        if stats.languages:
            lines.append(f"\n### Language Distribution — {stats.name}")
            lines.append("| Language | Count | Percentage |")
            lines.append("|----------|-------|------------|")
            total = sum(stats.languages.values())
            for lang, count in stats.languages.most_common():
                lines.append(f"| {lang} | {count:,} | {count/total:.1%} |")

        if stats.law_areas:
            lines.append(f"\n### Law Area Distribution — {stats.name}")
            lines.append("| Law Area | Count | Percentage |")
            lines.append("|----------|-------|------------|")
            total = sum(stats.law_areas.values())
            for area, count in stats.law_areas.most_common():
                lines.append(f"| {area} | {count:,} | {count/total:.1%} |")

        if stats.labels:
            lines.append(f"\n### Label Distribution — {stats.name}")
            lines.append("| Label | Count | Percentage |")
            lines.append("|-------|-------|------------|")
            total = sum(stats.labels.values())
            for label, count in stats.labels.most_common():
                lines.append(f"| {label} | {count:,} | {count/total:.1%} |")

        if stats.facts_lengths:
            lines.append(f"\n### Text Length Distribution — {stats.name}")
            sorted_lens = sorted(stats.facts_lengths)
            n = len(sorted_lens)
            lines.append(f"- Facts: min={sorted_lens[0]:,}, p25={sorted_lens[n//4]:,}, "
                         f"median={sorted_lens[n//2]:,}, p75={sorted_lens[3*n//4]:,}, "
                         f"p95={sorted_lens[int(n*0.95)]:,}, max={sorted_lens[-1]:,}")

        if stats.considerations_lengths:
            sorted_lens = sorted(stats.considerations_lengths)
            n = len(sorted_lens)
            lines.append(f"- Considerations: min={sorted_lens[0]:,}, p25={sorted_lens[n//4]:,}, "
                         f"median={sorted_lens[n//2]:,}, p75={sorted_lens[3*n//4]:,}, "
                         f"p95={sorted_lens[int(n*0.95)]:,}, max={sorted_lens[-1]:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"EDA report saved to {output_path}")


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ATHENA2 Data Ingestion Pipeline")
    parser.add_argument("--dataset", choices=["primary", "secondary", "criticality", "citation", "all"],
                        default="all", help="Which dataset to process")
    parser.add_argument("--stats-only", action="store_true",
                        help="Compute statistics without saving processed data")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Path to data config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config() if args.config == CONFIG_PATH else yaml.safe_load(args.config.read_text())
    output_dir = Path(config["paths"]["processed"])

    all_stats = []

    if args.dataset in ("primary", "all"):
        stats = ingest_primary(config, output_dir, args.stats_only)
        all_stats.append(stats)
        print(stats.summary())

    if args.dataset in ("secondary", "all"):
        stats = ingest_secondary(config, output_dir, args.stats_only)
        all_stats.append(stats)
        print(stats.summary())

    if args.dataset in ("criticality", "all"):
        stats = ingest_criticality(config, output_dir, args.stats_only)
        all_stats.append(stats)
        print(stats.summary())

    if args.dataset in ("citation", "all"):
        stats = ingest_citation_extraction(config, output_dir, args.stats_only)
        all_stats.append(stats)
        print(stats.summary())

    # Generate EDA report
    if all_stats:
        report_path = Path(config["paths"]["reports"]) / "eda_report.md"
        generate_eda_report(all_stats, report_path)


if __name__ == "__main__":
    main()
