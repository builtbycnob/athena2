"""LLM-assisted feature extraction from Swiss legal text.

Extracts structured features from the `considerations` field that require
understanding legal reasoning — error types, severity, decisive arguments,
reasoning patterns. These features become the world model's training signal.

IMPORTANT: This module prepares extraction scripts for LOCAL execution.
It does NOT call external APIs. Run with a local model via oMLX:

    OMLX_MODEL=qwen3.5-35b-a3b-text-hi python -m athena2.features.llm_features \
        --input data/processed/sjp_xl.parquet \
        --output data/features/llm_features.parquet \
        --batch-size 50

Extraction rate: ~50 cases/hour with 35B model (each case = 1 LLM call).
Full dataset (329K): would take ~6500 hours. Use sampling strategy:
- Phase 1: 10K random sample (200 hours, ~8 days)
- Phase 2: Additional 40K if Phase 1 shows value (~33 days)
- Alternative: Use faster model (8B) for bulk, 35B for validation subset

Or: Extract features from considerations via regex/heuristics for the majority,
use LLM only for the hard cases where regex fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Extraction Schema ──────────────────────────────────────────────

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "errors_identified": {
            "type": "array",
            "description": "Legal errors identified by the court in the lower court decision",
            "items": {
                "type": "object",
                "properties": {
                    "error_type": {
                        "type": "string",
                        "enum": [
                            "fact_finding",           # Sachverhaltsfeststellung
                            "law_application",        # Rechtsanwendung
                            "procedural",             # Verfahrensrecht
                            "constitutional",         # Verfassungsrecht
                            "discretion_abuse",       # Ermessensmissbrauch
                            "insufficient_reasoning", # Begründungsmangel
                            "none",                   # No error found
                        ],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["decisive", "significant", "minor", "none"],
                    },
                    "description": {
                        "type": "string",
                        "maxLength": 200,
                    },
                    "legal_basis": {
                        "type": "string",
                        "description": "Law article referenced (e.g., 'Art. 9 BV')",
                        "maxLength": 50,
                    },
                },
                "required": ["error_type", "severity"],
            },
        },
        "reasoning_pattern": {
            "type": "string",
            "enum": [
                "de_novo_review",         # Full re-examination of law
                "arbitrariness_review",   # Willkürprüfung — only manifest errors
                "proportionality_test",   # Verhältnismässigkeit
                "balancing_test",         # Interessenabwägung
                "subsumption",            # Subsumtion — fact → legal category
                "teleological",           # Purpose-driven interpretation
                "systematic",             # Systematic interpretation
                "historical",             # Historical interpretation
                "mixed",                  # Multiple patterns
            ],
            "description": "Primary reasoning pattern used by the court",
        },
        "decisive_factor": {
            "type": "string",
            "description": "The single most important factor in the court's decision",
            "maxLength": 300,
        },
        "standard_of_review": {
            "type": "string",
            "enum": [
                "free_review",            # Freie Überprüfung
                "limited_review",         # Eingeschränkte Kognition
                "arbitrariness_only",     # Nur Willkür
            ],
            "description": "How thoroughly the court reviews the lower decision",
        },
        "outcome_granular": {
            "type": "string",
            "enum": [
                "full_dismissal",         # Beschwerde abgewiesen
                "full_approval",          # Beschwerde gutgeheissen
                "partial_approval",       # Teilweise gutgeheissen
                "remand",                 # Rückweisung
                "inadmissible",           # Nicht eintreten
                "withdrawn",              # Gegenstandslos
                "other",
            ],
            "description": "Granular outcome beyond binary approval/dismissal",
        },
    },
    "required": ["errors_identified", "reasoning_pattern", "outcome_granular"],
}


# ── Extraction Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Swiss legal analyst. You analyze considerations (Erwägungen)
from Swiss Federal Supreme Court decisions and extract structured features.

You must respond with a JSON object following the provided schema exactly.
Be precise and conservative — only mark errors as "decisive" if the court
explicitly states they affected the outcome. Use the court's own language
and reasoning, not your interpretation.

Important:
- error_type should reflect the Swiss Federal Supreme Court's categorization
- severity should match the court's emphasis (decisive = the court says this error changes the result)
- reasoning_pattern should reflect the PRIMARY method of legal analysis used
- decisive_factor should be a concise statement of what determined the outcome
- outcome_granular should capture nuances beyond simple approval/dismissal"""

USER_PROMPT_TEMPLATE = """Analyze the following considerations (Erwägungen) from Swiss Federal
Supreme Court decision {decision_id}.

Language: {language}

CONSIDERATIONS:
{considerations}

Extract the structured features as specified in the JSON schema."""


# ── Extraction Functions ───────────────────────────────────────────

@dataclass
class LLMFeatures:
    """LLM-extracted features for a single case."""
    decision_id: str
    errors_identified: list[dict[str, str]] = field(default_factory=list)
    reasoning_pattern: str = ""
    decisive_factor: str = ""
    standard_of_review: str = ""
    outcome_granular: str = ""
    extraction_success: bool = True
    extraction_error: str = ""
    extraction_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "n_errors": len(self.errors_identified),
            "has_decisive_error": any(
                e.get("severity") == "decisive" for e in self.errors_identified
            ),
            "error_types": [e.get("error_type", "") for e in self.errors_identified],
            "max_severity": max(
                (e.get("severity", "none") for e in self.errors_identified),
                key=lambda s: {"decisive": 3, "significant": 2, "minor": 1, "none": 0}.get(s, -1),
                default="none",
            ),
            "reasoning_pattern": self.reasoning_pattern,
            "decisive_factor": self.decisive_factor,
            "standard_of_review": self.standard_of_review,
            "outcome_granular": self.outcome_granular,
            "errors_json": json.dumps(self.errors_identified),
            "extraction_success": self.extraction_success,
            "extraction_error": self.extraction_error,
            "extraction_time_s": self.extraction_time_s,
        }


def extract_single(
    decision_id: str,
    considerations: str,
    language: str,
    omlx_base_url: str = "http://localhost:8000",
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> LLMFeatures:
    """Extract LLM features from a single case's considerations.

    Args:
        decision_id: Case identifier.
        considerations: Full considerations text.
        language: Language code (de/fr/it).
        omlx_base_url: oMLX server URL.
        model: Model name (defaults to OMLX_MODEL env var).
        max_tokens: Max output tokens.
        temperature: LLM temperature.

    Returns:
        LLMFeatures with extracted structured data.
    """
    import httpx

    if not model:
        model = os.environ.get("OMLX_MODEL", "qwen3.5-35b-a3b-text-hi")

    features = LLMFeatures(decision_id=decision_id)

    # Truncate considerations if too long (keep first 12K chars ≈ 3K tokens)
    if len(considerations) > 12000:
        considerations = considerations[:12000] + "\n\n[TRUNCATED]"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        decision_id=decision_id,
        language=language,
        considerations=considerations,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "features", "schema": EXTRACTION_SCHEMA}},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.time()
    try:
        client = httpx.Client(base_url=omlx_base_url, timeout=300.0)
        resp = client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        features.errors_identified = parsed.get("errors_identified", [])
        features.reasoning_pattern = parsed.get("reasoning_pattern", "")
        features.decisive_factor = parsed.get("decisive_factor", "")
        features.standard_of_review = parsed.get("standard_of_review", "")
        features.outcome_granular = parsed.get("outcome_granular", "")
        features.extraction_success = True

    except Exception as e:
        features.extraction_success = False
        features.extraction_error = str(e)
        logger.warning(f"Extraction failed for {decision_id}: {e}")

    features.extraction_time_s = round(time.time() - t0, 1)
    return features


def extract_batch_sequential(
    rows: list[dict[str, Any]],
    omlx_base_url: str = "http://localhost:8000",
    model: str | None = None,
    progress_every: int = 10,
) -> list[dict[str, Any]]:
    """Extract LLM features for a batch of rows (sequential, crash-safe).

    Each result is yielded immediately for crash-safe persistence.
    """
    results = []
    total = len(rows)

    for i, row in enumerate(rows, 1):
        decision_id = row.get("decision_id", str(i))
        considerations = row.get("considerations", "")

        if not considerations or len(considerations) < 100:
            features = LLMFeatures(
                decision_id=decision_id,
                extraction_success=False,
                extraction_error="considerations too short",
            )
        else:
            features = extract_single(
                decision_id=decision_id,
                considerations=considerations,
                language=row.get("language", "de"),
                omlx_base_url=omlx_base_url,
                model=model,
            )

        results.append(features.to_dict())

        if i % progress_every == 0:
            logger.info(f"[{i}/{total}] Last: {decision_id} ({features.extraction_time_s:.1f}s)")

    return results


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Feature Extraction for ATHENA2")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input Parquet file (from ingestion pipeline)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output Parquet file for extracted features")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample size (default: all)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start offset (for resuming)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Save checkpoint every N cases")
    parser.add_argument("--model", type=str, default=None,
                        help="Override oMLX model name")
    parser.add_argument("--omlx-url", type=str, default="http://localhost:8000",
                        help="oMLX server URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import pandas as pd

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} rows from {args.input}")

    # Filter to cases with considerations
    df = df[df["considerations"].str.len() > 100].copy()
    logger.info(f"After filtering: {len(df):,} cases with considerations")

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        logger.info(f"Sampled {len(df):,} cases")

    if args.offset > 0:
        df = df.iloc[args.offset:]
        logger.info(f"Starting from offset {args.offset}: {len(df):,} remaining")

    rows = df.to_dict(orient="records")

    # Process in batches with checkpointing
    all_results = []
    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start:batch_start + args.batch_size]
        logger.info(f"Batch {batch_start//args.batch_size + 1}: {len(batch)} cases")

        batch_results = extract_batch_sequential(
            batch,
            omlx_base_url=args.omlx_url,
            model=args.model,
        )
        all_results.extend(batch_results)

        # Checkpoint save
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_parquet(args.output, index=False)
        logger.info(f"Checkpoint saved: {len(all_results)} total features → {args.output}")

    logger.info(f"Complete: {len(all_results)} features extracted")


if __name__ == "__main__":
    main()
