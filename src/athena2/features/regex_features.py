"""Regex-based feature extraction from Swiss legal text.

Fast extraction (~100K cases/min) of structured legal references:
- BGE/DTF citations (Swiss Federal Supreme Court precedents)
- SR law references (Systematische Rechtssammlung numbers)
- Article references (Art. X patterns)
- Procedural markers (Beschwerde, recours, ricorso types)

These run on all 329K cases without LLM. LLM-based extraction (slower)
is in llm_features.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Citation Patterns ──────────────────────────────────────────────

# BGE citations: "BGE 144 III 120" or "DTF 144 III 120" (German/French/Italian variants)
BGE_PATTERN = re.compile(
    r"\b(BGE|DTF|ATF)\s+"           # BGE (de) / DTF (fr) / ATF (it)
    r"(\d{1,3})\s+"                  # Volume number
    r"(I{1,3}|IV|V)\s+"             # Part (Roman numeral)
    r"(\d+)"                         # Page number
    r"(?:\s+E\.\s*([\d.]+))?"       # Optional Erwägung (consideration) number
    ,
    re.IGNORECASE,
)

# SR (Systematische Rechtssammlung) law references: "SR 210" or "SR 220"
SR_PATTERN = re.compile(
    r"\bSR\s+(\d{3}(?:\.\d+)*)",     # SR number (e.g., 210, 220, 311.0)
    re.IGNORECASE,
)

# Article references: "Art. 8 ZGB", "art. 41 OR", "Art. 29 BV"
ARTICLE_PATTERN = re.compile(
    r"\b[Aa]rt\.?\s+"               # Art. or art
    r"(\d+)"                         # Article number
    r"(?:\s*(?:Abs|al|cpv)\.?\s*"   # Optional paragraph (de/fr/it)
    r"(\d+))?"
    r"(?:\s*(?:lit|let)\.?\s*"      # Optional letter
    r"([a-z]))?"
    r"(?:\s+([A-Z]{2,10}))?"       # Optional law abbreviation (ZGB, OR, BV, StGB...)
)

# Procedural type markers
BESCHWERDE_PATTERN = re.compile(
    r"\b(Beschwerde\s+in\s+(?:Zivilsachen|Strafsachen|öffentlich-rechtlichen\s+Angelegenheiten)|"
    r"subsidiäre\s+Verfassungsbeschwerde|"
    r"recours\s+en\s+matière\s+(?:civile|pénale|de\s+droit\s+public)|"
    r"recours\s+constitutionnel\s+subsidiaire|"
    r"ricorso\s+in\s+materia\s+(?:civile|penale|di\s+diritto\s+pubblico)|"
    r"ricorso\s+sussidiario\s+in\s+materia\s+costituzionale)\b",
    re.IGNORECASE,
)

# Outcome indicators in considerations
OUTCOME_INDICATORS_DE = re.compile(
    r"\b(Die\s+Beschwerde\s+(?:ist|wird)\s+(?:abgewiesen|gutgeheissen|teilweise\s+gutgeheissen)|"
    r"ist\s+(?:abzuweisen|gutzuheissen))\b",
    re.IGNORECASE,
)
OUTCOME_INDICATORS_FR = re.compile(
    r"\b(Le\s+recours\s+(?:est|sera)\s+(?:rejeté|admis|partiellement\s+admis)|"
    r"il\s+y\s+a\s+lieu\s+de\s+(?:rejeter|admettre))\b",
    re.IGNORECASE,
)
OUTCOME_INDICATORS_IT = re.compile(
    r"\b(Il\s+ricorso\s+(?:è|viene|va)\s+(?:respinto|accolto|parzialmente\s+accolto)|"
    r"dev['']essere\s+(?:respinto|accolto))\b",
    re.IGNORECASE,
)


# ── Feature Extraction ─────────────────────────────────────────────

@dataclass
class RegexFeatures:
    """Extracted regex-based features for a single case."""
    decision_id: str

    # Citations
    bge_citations: list[dict[str, str]] = field(default_factory=list)
    sr_references: list[str] = field(default_factory=list)
    article_references: list[dict[str, str | None]] = field(default_factory=list)

    # Counts
    n_bge_citations: int = 0
    n_sr_references: int = 0
    n_article_references: int = 0
    n_unique_laws: int = 0

    # Procedural
    procedure_type: str | None = None

    # Outcome indicators (from considerations)
    has_outcome_indicator: bool = False
    outcome_indicator_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for Parquet storage."""
        return {
            "decision_id": self.decision_id,
            "n_bge_citations": self.n_bge_citations,
            "n_sr_references": self.n_sr_references,
            "n_article_references": self.n_article_references,
            "n_unique_laws": self.n_unique_laws,
            "bge_citations_json": str(self.bge_citations),
            "sr_references_json": str(self.sr_references),
            "article_references_json": str(self.article_references),
            "procedure_type": self.procedure_type,
            "has_outcome_indicator": self.has_outcome_indicator,
        }


def extract_regex_features(
    decision_id: str,
    facts: str,
    considerations: str = "",
) -> RegexFeatures:
    """Extract regex-based features from a case's text fields.

    Args:
        decision_id: Unique case identifier.
        facts: Facts section text.
        considerations: Considerations section text (optional).

    Returns:
        RegexFeatures with all extracted references.
    """
    full_text = f"{facts}\n{considerations}"
    features = RegexFeatures(decision_id=decision_id)

    # BGE citations
    for m in BGE_PATTERN.finditer(full_text):
        features.bge_citations.append({
            "type": m.group(1).upper(),
            "volume": m.group(2),
            "part": m.group(3),
            "page": m.group(4),
            "consideration": m.group(5),
        })
    features.n_bge_citations = len(features.bge_citations)

    # SR references
    for m in SR_PATTERN.finditer(full_text):
        features.sr_references.append(m.group(1))
    features.sr_references = list(set(features.sr_references))
    features.n_sr_references = len(features.sr_references)

    # Article references
    unique_laws = set()
    for m in ARTICLE_PATTERN.finditer(full_text):
        law = m.group(4)
        features.article_references.append({
            "article": m.group(1),
            "paragraph": m.group(2),
            "letter": m.group(3),
            "law": law,
        })
        if law:
            unique_laws.add(law)
    features.n_article_references = len(features.article_references)
    features.n_unique_laws = len(unique_laws)

    # Procedure type
    proc_match = BESCHWERDE_PATTERN.search(facts)
    if proc_match:
        features.procedure_type = proc_match.group(1)

    # Outcome indicators (from considerations only)
    if considerations:
        for pattern in [OUTCOME_INDICATORS_DE, OUTCOME_INDICATORS_FR, OUTCOME_INDICATORS_IT]:
            outcome_match = pattern.search(considerations)
            if outcome_match:
                features.has_outcome_indicator = True
                features.outcome_indicator_text = outcome_match.group(1)
                break

    return features


def extract_batch(
    rows: list[dict[str, Any]],
    facts_key: str = "facts",
    considerations_key: str = "considerations",
    id_key: str = "decision_id",
) -> list[dict[str, Any]]:
    """Extract regex features for a batch of rows.

    Args:
        rows: List of dicts with at least facts and decision_id.
        facts_key: Key for facts field.
        considerations_key: Key for considerations field.
        id_key: Key for decision ID.

    Returns:
        List of feature dicts ready for DataFrame construction.
    """
    results = []
    for row in rows:
        features = extract_regex_features(
            decision_id=row.get(id_key, ""),
            facts=row.get(facts_key, ""),
            considerations=row.get(considerations_key, ""),
        )
        results.append(features.to_dict())
    return results
