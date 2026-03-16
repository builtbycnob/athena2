#!/usr/bin/env python3
"""ATHENA2 — Label Cleaning Pipeline.

Verifies and corrects labels in SJP-XL using the considerations text.
Three stages:
    1. Regex extraction from conclusion phrases in considerations
    2. LLM extraction for ambiguous/no-match cases (optional)
    3. Reconciliation: original vs extracted → verified/corrected/ambiguous

Usage:
    # Full pipeline (regex + reconciliation)
    uv run python scripts/clean_labels.py

    # Regex only, then review before LLM
    uv run python scripts/clean_labels.py --regex-only

    # With LLM for ambiguous cases
    uv run python scripts/clean_labels.py --with-llm

    # Just report on existing cleaned data
    uv run python scripts/clean_labels.py --report-only
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("athena2.clean_labels")

# ── Regex Patterns ────────────────────────────────────────────────
# Each pattern: (compiled_regex, predicted_label, priority)
# Priority: lower = stronger signal. Used to resolve conflicts.
# All patterns are applied to the LAST 1500 chars of considerations.

WINDOW_SIZE = 1500  # chars from end of considerations to search

def build_patterns():
    """Build compiled regex patterns for all three languages.

    Returns dict[language] → list[(regex, label, priority, name)]
    Patterns ordered by priority (strongest first).
    """
    patterns = {"de": [], "fr": [], "it": []}

    # ── GERMAN ────────────────────────────────────────────────
    de = patterns["de"]

    # Tier 1: Unambiguous conclusion phrases (98%+ precision)
    de.append((re.compile(r"Beschwerde\s+(?:ist\s+)?abzuweisen", re.I), "dismissal", 1, "de_abzuweisen"))
    de.append((re.compile(r"Beschwerde\s+(?:wird\s+)?abgewiesen", re.I), "dismissal", 1, "de_abgewiesen"))
    de.append((re.compile(r"Beschwerde\s+(?:ist\s+)?gutzuheissen", re.I), "approval", 1, "de_gutzuheissen"))
    de.append((re.compile(r"Beschwerde\s+(?:wird\s+)?gutgeheissen", re.I), "approval", 1, "de_gutgeheissen"))

    # Tier 2: Annulment / reversal (strong approval signal)
    de.append((re.compile(r"(?:Urteil|Entscheid|Verfügung|Beschluss)\s+(?:ist\s+)?aufzuheben", re.I), "approval", 2, "de_aufzuheben"))
    de.append((re.compile(r"(?:Urteil|Entscheid|Verfügung|Beschluss)\s+(?:wird\s+)?aufgehoben", re.I), "approval", 2, "de_aufgehoben"))
    # IMPORTANT: "in Gutheissung" must refer to the Beschwerde, NOT to "Gesuch um unentgeltliche Rechtspflege"
    # "in Gutheissung des Gesuchs" = approving legal aid (appears in both approval AND dismissal cases)
    de.append((re.compile(r"[Ii]n\s+Gutheissung\s+(?:der\s+Beschwerde|des\s+Rekurses)", re.I), "approval", 2, "de_in_gutheissung"))

    # Tier 2: Inadmissibility → mapped to dismissal in SJP-XL
    # IMPORTANT: require "auf die Beschwerde" context to avoid matching
    # "Schaden nicht eingetreten" (damage didn't occur) or descriptions of what lower court did
    de.append((re.compile(r"(?:ist|wird)\s+auf\s+die\s+Beschwerde\s+nicht\s+einzutreten", re.I), "dismissal", 2, "de_nicht_einzutreten"))
    de.append((re.compile(r"auf\s+die\s+Beschwerde\s+(?:wird\s+|ist\s+)?nicht\s+eingetreten", re.I), "dismissal", 2, "de_nicht_eingetreten"))
    # Note: Nichteintreten cases are EXCLUDED from SJP-XL per Niklaus et al.
    # But some may have slipped through. We map to dismissal as that's how SJP treats partial-inadmissibility.

    # Tier 2: Partial approval → mapped to approval in SJP-XL
    de.append((re.compile(r"teilweise\s+gutzuheissen", re.I), "approval", 2, "de_teilweise_gutzuheissen"))
    de.append((re.compile(r"teilweise\s+gutgeheissen", re.I), "approval", 2, "de_teilweise_gutgeheissen"))

    # Tier 3: Weaker signals
    de.append((re.compile(r"Beschwerde\s+(?:ist|erweist\s+sich\s+als)\s+unbegründet", re.I), "dismissal", 3, "de_unbegrundet"))
    de.append((re.compile(r"Beschwerde\s+(?:ist|erweist\s+sich\s+als)\s+begründet", re.I), "approval", 3, "de_begrundet"))
    # "Beschwerde als gegenstandslos abzuschreiben" = case written off → dismissal
    # IMPORTANT: bare "gegenstandslos" is NOT reliable — it usually refers to
    # "Gesuch um unentgeltliche Rechtspflege ist gegenstandslos" (legal aid moot because party WON)
    de.append((re.compile(r"Beschwerde\s+(?:als\s+)?(?:gegenstandslos\s+)?(?:abzuschreiben|abgeschrieben|vom\s+Geschäftsverzeichnis\s+abzuschreiben)", re.I), "dismissal", 3, "de_write_off"))
    de.append((re.compile(r"(?:als\s+)?gegenstandslos\s+(?:abzuschreiben|abgeschrieben|vom\s+Geschäftsverzeichnis)", re.I), "dismissal", 3, "de_gegenstandslos_abschreiben"))

    # Tier 4: Cost allocation proxy (weaker but helps coverage)
    # "Gerichtskosten dem Beschwerdeführer" = appellant pays = dismissal
    de.append((re.compile(r"Gerichtskosten\s+(?:werden\s+)?(?:dem|der)\s+Beschwerde(?:führer|führerin)", re.I), "dismissal", 4, "de_costs_appellant"))
    # "Gerichtskosten dem Beschwerdegegner/Kanton" = respondent pays = approval
    de.append((re.compile(r"Gerichtskosten\s+(?:werden\s+)?(?:dem|der)\s+(?:Beschwerdegegner|Kanton|Gemeinde)", re.I), "approval", 4, "de_costs_respondent"))
    # "keine Gerichtskosten" = no costs (often approval)
    de.append((re.compile(r"(?:keine|keinen)\s+Gerichtskosten", re.I), "approval", 4, "de_no_costs"))
    # "Parteientschädigung" = party compensation → approval
    de.append((re.compile(r"Parteientschädigung\s+(?:von|in)", re.I), "approval", 4, "de_party_compensation"))

    # ── FRENCH ────────────────────────────────────────────────
    fr = patterns["fr"]

    # Tier 1: Unambiguous (97%+ precision)
    fr.append((re.compile(r"recours\s+(?:est|doit\s+être|sera?)\s+rejeté", re.I), "dismissal", 1, "fr_rejete"))
    fr.append((re.compile(r"recours\s+(?:est|doit\s+être|sera?)\s+admis", re.I), "approval", 1, "fr_admis"))
    fr.append((re.compile(r"rejeter\s+le\s+recours", re.I), "dismissal", 1, "fr_rejeter"))
    fr.append((re.compile(r"admettre\s+le\s+recours", re.I), "approval", 1, "fr_admettre"))

    # Tier 2: Annulment
    fr.append((re.compile(r"(?:jugement|arrêt|décision|ordonnance)\s+(?:est|doit\s+être|sera?)\s+annulé", re.I), "approval", 2, "fr_annule"))
    fr.append((re.compile(r"(?:jugement|arrêt|décision)\s+(?:est|doit\s+être)\s+réformé", re.I), "approval", 2, "fr_reforme"))

    # Tier 2: Inadmissibility
    fr.append((re.compile(r"recours\s+(?:est|doit\s+être)\s+irrecevable", re.I), "dismissal", 2, "fr_irrecevable"))
    fr.append((re.compile(r"n'est\s+pas\s+entré\s+en\s+matière", re.I), "dismissal", 2, "fr_pas_entre"))

    # Tier 2: Partial
    fr.append((re.compile(r"partiellement\s+admis", re.I), "approval", 2, "fr_partiellement_admis"))

    # Tier 3: Weaker
    fr.append((re.compile(r"recours\s+(?:est|apparaît)\s+mal\s+fondé", re.I), "dismissal", 3, "fr_mal_fonde"))
    fr.append((re.compile(r"recours\s+(?:est|apparaît)\s+(?:bien\s+)?fondé", re.I), "approval", 3, "fr_bien_fonde"))
    fr.append((re.compile(r"recours\s+(?:est|apparaît)\s+infondé", re.I), "dismissal", 3, "fr_infonde"))
    # IMPORTANT: bare "sans objet" mostly refers to "demande d'assistance judiciaire sans objet"
    # (legal aid moot because party won). Only match when recours is the subject.
    fr.append((re.compile(r"recours\s+(?:est\s+)?(?:devenu\s+)?sans\s+objet", re.I), "dismissal", 3, "fr_write_off"))
    fr.append((re.compile(r"(?:cause|procédure)\s+(?:est\s+)?radiée", re.I), "dismissal", 3, "fr_radie"))

    # Tier 4: Cost proxy
    fr.append((re.compile(r"frais\s+(?:judiciaires\s+)?(?:sont\s+)?(?:mis\s+)?à\s+(?:la\s+)?charge\s+d[ue]\s+recourant", re.I), "dismissal", 4, "fr_costs_appellant"))
    fr.append((re.compile(r"frais\s+(?:judiciaires\s+)?(?:sont\s+)?(?:mis\s+)?à\s+(?:la\s+)?charge\s+d[ue]\s+(?:l')?intimé", re.I), "approval", 4, "fr_costs_respondent"))
    fr.append((re.compile(r"pas\s+(?:perçu|prélevé)\s+de\s+frais", re.I), "approval", 4, "fr_no_costs"))
    fr.append((re.compile(r"(?:canton|intimé)\s+versera.*dépens", re.I), "approval", 4, "fr_depens"))

    # ── ITALIAN ───────────────────────────────────────────────
    it = patterns["it"]

    # Tier 1: Unambiguous (98%+ precision)
    it.append((re.compile(r"ricorso\s+(?:è|dev'essere|viene|va)\s+respinto", re.I), "dismissal", 1, "it_respinto"))
    it.append((re.compile(r"ricorso\s+(?:è|dev'essere|viene|va)\s+accolto", re.I), "approval", 1, "it_accolto"))
    it.append((re.compile(r"respingere\s+il\s+ricorso", re.I), "dismissal", 1, "it_respingere"))
    it.append((re.compile(r"accogliere\s+il\s+ricorso", re.I), "approval", 1, "it_accogliere"))

    # Tier 2: Annulment
    it.append((re.compile(r"(?:sentenza|decisione|giudizio)\s+(?:è|viene|va|dev'essere)\s+annullat[ao]", re.I), "approval", 2, "it_annullata"))
    it.append((re.compile(r"(?:sentenza|decisione)\s+(?:è|viene)\s+riformat[ao]", re.I), "approval", 2, "it_riformata"))

    # Tier 2: Inadmissibility
    it.append((re.compile(r"ricorso\s+(?:è|viene)\s+(?:dichiarato\s+)?inammissibile", re.I), "dismissal", 2, "it_inammissibile"))

    # Tier 2: Partial
    it.append((re.compile(r"parzialmente\s+accolto", re.I), "approval", 2, "it_parzialmente_accolto"))

    # Tier 3: Weaker
    it.append((re.compile(r"ricorso\s+(?:è|risulta|appare)\s+infondato", re.I), "dismissal", 3, "it_infondato"))
    it.append((re.compile(r"ricorso\s+(?:è|risulta)\s+fondato", re.I), "approval", 3, "it_fondato"))
    # IMPORTANT: bare "privo/a d'oggetto" mostly refers to "istanza di assistenza giudiziaria priva d'oggetto"
    # Only match when ricorso is the subject.
    it.append((re.compile(r"ricorso\s+(?:è\s+)?(?:diventato\s+|divenuto\s+)?(?:privo|priva)\s+d'oggetto", re.I), "dismissal", 3, "it_write_off"))
    it.append((re.compile(r"ricorso\s+(?:va\s+|è\s+)?stralciat", re.I), "dismissal", 3, "it_stralciato"))
    # "decisione/sentenza impugnata confermata" = lower court confirmed = dismissal
    # Bare "confermata" is too ambiguous (can refer to facts, not the ruling)
    it.append((re.compile(r"(?:decisione|sentenza)\s+(?:impugnata\s+)?(?:è\s+)?confermat[oa]", re.I), "dismissal", 3, "it_confermata"))

    # Tier 4: Cost proxy
    it.append((re.compile(r"spese\s+(?:giudiziarie\s+)?(?:sono\s+)?(?:poste\s+)?a\s+carico\s+del\s+ricorrente", re.I), "dismissal", 4, "it_costs_appellant"))
    it.append((re.compile(r"spese\s+(?:giudiziarie\s+)?(?:sono\s+)?(?:poste\s+)?a\s+carico\s+del(?:l'opponente|lo\s+Stato|l'intimat)", re.I), "approval", 4, "it_costs_respondent"))
    it.append((re.compile(r"non\s+si\s+(?:prelevano|riscuotono)\s+(?:spese|tasse)", re.I), "approval", 4, "it_no_costs"))

    return patterns


# ── Stage 1: Regex Extraction ────────────────────────────────────

def extract_label_regex(text: str, language: str, patterns: dict) -> dict:
    """Extract label from considerations text using regex.

    Returns dict with:
        label: str or None
        confidence: 'high' | 'medium' | 'low' | None
        matches: list of (name, label, priority)
        conflict: bool
    """
    if not text or len(text) < 50:
        return {"label": None, "confidence": None, "matches": [], "conflict": False}

    # Search last WINDOW_SIZE chars
    window = text[-WINDOW_SIZE:]
    lang_patterns = patterns.get(language, [])

    matches = []
    for regex, label, priority, name in lang_patterns:
        if regex.search(window):
            matches.append((name, label, priority))

    if not matches:
        return {"label": None, "confidence": None, "matches": [], "conflict": False}

    # Get unique predicted labels
    labels_found = set(m[1] for m in matches)

    if len(labels_found) == 1:
        # Unambiguous — all matches agree
        label = labels_found.pop()
        best_priority = min(m[2] for m in matches)
        confidence = {1: "high", 2: "high", 3: "medium", 4: "low"}.get(best_priority, "low")
        return {"label": label, "confidence": confidence, "matches": matches, "conflict": False}

    # Conflict — both dismissal and approval patterns found
    # Resolve by priority: lower priority number wins
    dismissal_best = min((m[2] for m in matches if m[1] == "dismissal"), default=99)
    approval_best = min((m[2] for m in matches if m[1] == "approval"), default=99)

    if dismissal_best < approval_best:
        return {"label": "dismissal", "confidence": "low", "matches": matches, "conflict": True}
    elif approval_best < dismissal_best:
        return {"label": "approval", "confidence": "low", "matches": matches, "conflict": True}
    else:
        # Same priority — count matches per side
        n_dismissal = sum(1 for m in matches if m[1] == "dismissal")
        n_approval = sum(1 for m in matches if m[1] == "approval")
        if n_dismissal > n_approval:
            return {"label": "dismissal", "confidence": "low", "matches": matches, "conflict": True}
        elif n_approval > n_dismissal:
            return {"label": "approval", "confidence": "low", "matches": matches, "conflict": True}
        else:
            return {"label": None, "confidence": None, "matches": matches, "conflict": True}


def run_stage1(df: pd.DataFrame, patterns: dict) -> pd.DataFrame:
    """Run regex extraction on all rows."""
    logger.info("Stage 1: Regex extraction from considerations...")
    t0 = time.time()

    results = []
    for i, row in df.iterrows():
        text = row.get("considerations", "")
        lang = row.get("language", "de")
        result = extract_label_regex(text, lang, patterns)
        results.append({
            "label_extracted": result["label"],
            "extraction_confidence": result["confidence"],
            "extraction_conflict": result["conflict"],
            "extraction_n_matches": len(result["matches"]),
            "extraction_matches": json.dumps([(m[0], m[1]) for m in result["matches"]]),
        })

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            logger.info(f"  {i+1}/{len(df)} ({(i+1)/elapsed:.0f} rows/s)")

    result_df = pd.DataFrame(results, index=df.index)
    elapsed = time.time() - t0
    logger.info(f"  Stage 1 complete: {len(df)} rows in {elapsed:.0f}s ({len(df)/elapsed:.0f} rows/s)")
    return result_df


# ── Stage 2: LLM Extraction (Optional) ──────────────────────────

LLM_SYSTEM_PROMPT = """You analyze the ending of Swiss Federal Supreme Court decisions (Bundesgericht).
Given the last paragraph of the court's reasoning, determine the outcome.
Respond with JSON: {"outcome": "dismissal" or "approval"}
- dismissal: appeal rejected (Beschwerde abgewiesen / recours rejeté / ricorso respinto)
- approval: appeal granted, lower court overturned (gutgeheissen / admis / accolto)
- If the appeal is partially approved, classify as "approval"
- If the appeal is inadmissible (Nichteintreten), classify as "dismissal"
Be concise. Only output the JSON."""

LLM_USER_TEMPLATE = """Decision {decision_id} ({language}).
Last section of considerations:

{text}

What is the outcome? Respond with JSON only."""


def extract_label_llm(
    decision_id: str,
    text: str,
    language: str,
    omlx_url: str = "http://localhost:8000",
    model: str = "qwen3.5-35b-a3b-text-hi",
) -> dict:
    """Extract label using LLM for a single case."""
    import httpx

    # Use last 2000 chars
    excerpt = text[-2000:] if len(text) > 2000 else text

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": LLM_USER_TEMPLATE.format(
                decision_id=decision_id, language=language, text=excerpt,
            )},
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "outcome",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "enum": ["dismissal", "approval"]},
                    },
                    "required": ["outcome"],
                },
            },
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        with httpx.Client(base_url=omlx_url, timeout=60.0) as client:
            resp = client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {"label": parsed.get("outcome"), "method": "llm", "error": None}
    except Exception as e:
        return {"label": None, "method": "llm", "error": str(e)}


def run_stage2(df: pd.DataFrame, ambiguous_mask: pd.Series, omlx_url: str, model: str) -> pd.DataFrame:
    """Run LLM extraction on ambiguous cases."""
    ambiguous_df = df[ambiguous_mask]
    n = len(ambiguous_df)
    logger.info(f"Stage 2: LLM extraction for {n} ambiguous cases...")

    if n == 0:
        return pd.DataFrame(index=df.index, columns=["llm_label", "llm_error"]).fillna("")

    results = []
    t0 = time.time()
    for i, (idx, row) in enumerate(ambiguous_df.iterrows()):
        result = extract_label_llm(
            decision_id=row.get("decision_id", str(idx)),
            text=row.get("considerations", ""),
            language=row.get("language", "de"),
            omlx_url=omlx_url,
            model=model,
        )
        results.append({"idx": idx, "llm_label": result["label"], "llm_error": result.get("error", "")})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate / 60
            logger.info(f"  LLM: {i+1}/{n} ({rate:.1f}/s, ETA {eta:.0f}min)")

    llm_df = pd.DataFrame(results).set_index("idx")
    # Merge back to full index
    full_llm = pd.DataFrame(index=df.index, columns=["llm_label", "llm_error"])
    full_llm.update(llm_df)
    full_llm = full_llm.fillna("")

    elapsed = time.time() - t0
    logger.info(f"  Stage 2 complete: {n} cases in {elapsed:.0f}s")
    return full_llm


# ── Stage 3: Reconciliation ──────────────────────────────────────

def reconcile(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile original labels with extracted labels.

    Adds columns: label_cleaned, label_status, extraction_method
    """
    logger.info("Stage 3: Reconciliation...")

    label_cleaned = []
    label_status = []
    extraction_method = []

    for _, row in df.iterrows():
        original = row["label"]
        extracted = row.get("label_extracted")
        llm_label = row.get("llm_label", "")
        confidence = row.get("extraction_confidence")
        has_considerations = isinstance(row.get("considerations", ""), str) and len(row.get("considerations", "")) > 100

        # Determine best extraction
        if extracted and extracted in ("dismissal", "approval"):
            best_extracted = extracted
            method = "regex"
        elif llm_label and llm_label in ("dismissal", "approval"):
            best_extracted = llm_label
            method = "llm"
        else:
            best_extracted = None
            method = "none"

        # Reconcile
        if best_extracted is None:
            if not has_considerations:
                label_cleaned.append(original)
                label_status.append("no_considerations")
                extraction_method.append("none")
            else:
                label_cleaned.append(original)
                label_status.append("no_extraction")
                extraction_method.append("none")
        elif best_extracted == original:
            label_cleaned.append(original)
            label_status.append("verified")
            extraction_method.append(method)
        else:
            # Discordance — trust extraction if confidence is not low
            if confidence in ("high", "medium") or method == "llm":
                label_cleaned.append(best_extracted)
                label_status.append("corrected")
                extraction_method.append(method)
            else:
                # Low confidence extraction disagrees — flag but keep original
                label_cleaned.append(original)
                label_status.append("ambiguous")
                extraction_method.append(method)

    df = df.copy()
    df["label_cleaned"] = label_cleaned
    df["label_status"] = label_status
    df["extraction_method"] = extraction_method
    df["label_extracted_raw"] = df.get("label_extracted", None)

    logger.info(f"  Reconciliation complete")
    return df


# ── Report ────────────────────────────────────────────────────────

def generate_report(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate detailed statistics report."""
    lines = []
    lines.append("# ATHENA2 — Label Cleaning Report")
    lines.append(f"\nDate: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Total cases: {len(df):,}")
    lines.append("")

    # Overall status
    lines.append("## 1. Overall Status Distribution")
    lines.append("")
    status_counts = df["label_status"].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        lines.append(f"| {status} | {count:,} | {pct:.1f}% |")
    lines.append("")

    # By language
    lines.append("## 2. Status by Language")
    lines.append("")
    for lang in ["de", "fr", "it"]:
        lang_df = df[df["language"] == lang]
        lines.append(f"### {lang.upper()} ({len(lang_df):,} cases)")
        status = lang_df["label_status"].value_counts()
        for s, c in status.items():
            lines.append(f"  - {s}: {c:,} ({c/len(lang_df)*100:.1f}%)")
        lines.append("")

    # Corrections
    corrected = df[df["label_status"] == "corrected"]
    lines.append(f"## 3. Corrections ({len(corrected):,} total)")
    lines.append("")

    if len(corrected) > 0:
        # Direction of corrections
        d2a = len(corrected[(corrected["label"] == "dismissal") & (corrected["label_cleaned"] == "approval")])
        a2d = len(corrected[(corrected["label"] == "approval") & (corrected["label_cleaned"] == "dismissal")])
        lines.append(f"| Direction | Count |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| dismissal → approval | {d2a:,} |")
        lines.append(f"| approval → dismissal | {a2d:,} |")
        lines.append("")

        # By language
        lines.append("### Corrections by Language")
        for lang in ["de", "fr", "it"]:
            lang_corr = corrected[corrected["language"] == lang]
            if len(lang_corr) > 0:
                d2a_l = len(lang_corr[(lang_corr["label"] == "dismissal") & (lang_corr["label_cleaned"] == "approval")])
                a2d_l = len(lang_corr[(lang_corr["label"] == "approval") & (lang_corr["label_cleaned"] == "dismissal")])
                lines.append(f"  - {lang.upper()}: {len(lang_corr):,} corrections (D→A: {d2a_l}, A→D: {a2d_l})")
        lines.append("")

    # Noise rate comparison
    lines.append("## 4. Noise Rate Comparison (Original vs Cleaned)")
    lines.append("")
    lines.append("| Metric | Original | After Cleaning |")
    lines.append("|--------|----------|----------------|")

    original_approval = len(df[df["label"] == "approval"])
    cleaned_approval = len(df[df["label_cleaned"] == "approval"])
    original_dismissal = len(df[df["label"] == "dismissal"])
    cleaned_dismissal = len(df[df["label_cleaned"] == "dismissal"])
    n_corrected = len(corrected)

    lines.append(f"| Total corrections | — | {n_corrected:,} ({n_corrected/len(df)*100:.1f}%) |")
    lines.append(f"| Approval count | {original_approval:,} | {cleaned_approval:,} |")
    lines.append(f"| Dismissal count | {original_dismissal:,} | {cleaned_dismissal:,} |")
    lines.append(f"| Approval % | {original_approval/len(df)*100:.1f}% | {cleaned_approval/len(df)*100:.1f}% |")
    lines.append("")

    # Verified rate by original class
    lines.append("## 5. Verification Rate by Original Class")
    lines.append("")
    for label in ["dismissal", "approval"]:
        label_df = df[df["label"] == label]
        verified = len(label_df[label_df["label_status"] == "verified"])
        corrected_l = len(label_df[label_df["label_status"] == "corrected"])
        no_ext = len(label_df[label_df["label_status"].isin(["no_extraction", "no_considerations"])])
        ambig = len(label_df[label_df["label_status"] == "ambiguous"])
        lines.append(f"### {label} ({len(label_df):,})")
        lines.append(f"  - Verified: {verified:,} ({verified/len(label_df)*100:.1f}%)")
        lines.append(f"  - Corrected: {corrected_l:,} ({corrected_l/len(label_df)*100:.1f}%)")
        lines.append(f"  - Ambiguous: {ambig:,} ({ambig/len(label_df)*100:.1f}%)")
        lines.append(f"  - No extraction: {no_ext:,} ({no_ext/len(label_df)*100:.1f}%)")
        lines.append("")

    # Extraction coverage
    lines.append("## 6. Extraction Coverage")
    lines.append("")
    has_extraction = df["label_extracted_raw"].notna() & (df["label_extracted_raw"] != "")
    for lang in ["de", "fr", "it"]:
        lang_df = df[df["language"] == lang]
        lang_has = has_extraction[df["language"] == lang].sum()
        lines.append(f"  - {lang.upper()}: {lang_has:,}/{len(lang_df):,} ({lang_has/len(lang_df)*100:.1f}%)")
    lines.append("")

    # Cleanlab comparison (if available)
    noisy_path = Path("data/models/noise_analysis/noisy_indices.npy")
    if noisy_path.exists():
        lines.append("## 7. Cleanlab Cross-Reference")
        lines.append("")
        try:
            noisy_indices = np.load(noisy_path)
            # These indices are into the training set — need to map
            train_mask = df["year"] <= 2015
            train_df = df[train_mask]
            if len(noisy_indices) > 0 and max(noisy_indices) < len(train_df):
                noisy_set = set(train_df.index[noisy_indices])
                cleanlab_flagged = df.index.isin(noisy_set)
                flagged_corrected = (cleanlab_flagged & (df["label_status"] == "corrected")).sum()
                flagged_verified = (cleanlab_flagged & (df["label_status"] == "verified")).sum()
                flagged_total = cleanlab_flagged.sum()
                lines.append(f"  - Cleanlab flagged: {flagged_total:,}")
                lines.append(f"  - Of which now corrected: {flagged_corrected:,} ({flagged_corrected/max(flagged_total,1)*100:.1f}%) — true positives")
                lines.append(f"  - Of which now verified: {flagged_verified:,} ({flagged_verified/max(flagged_total,1)*100:.1f}%) — false positives")
        except Exception as e:
            lines.append(f"  - Error loading cleanlab data: {e}")
        lines.append("")

    report = "\n".join(lines)

    report_path = output_dir / "label_cleaning_report.md"
    report_path.write_text(report)
    logger.info(f"Report saved: {report_path}")
    return report


def save_validation_sample(df: pd.DataFrame, output_dir: Path, n: int = 50):
    """Save a random sample of corrected cases for manual review."""
    corrected = df[df["label_status"] == "corrected"]
    if len(corrected) == 0:
        logger.info("No corrections to sample.")
        return

    sample = corrected.sample(min(n, len(corrected)), random_state=42)

    lines = ["# Label Corrections — Validation Sample", ""]
    lines.append(f"Sample size: {len(sample)} (of {len(corrected)} total corrections)")
    lines.append("")

    for _, row in sample.iterrows():
        did = row.get("decision_id", "unknown")
        lang = row.get("language", "?")
        original = row["label"]
        cleaned = row["label_cleaned"]
        method = row.get("extraction_method", "?")
        matches = row.get("extraction_matches", "[]")

        # Last 500 chars of considerations
        text = row.get("considerations", "")
        excerpt = text[-500:] if len(text) > 500 else text

        lines.append(f"## {did} ({lang})")
        lines.append(f"- **Original**: {original}")
        lines.append(f"- **Corrected to**: {cleaned}")
        lines.append(f"- **Method**: {method}")
        lines.append(f"- **Matches**: {matches}")
        lines.append(f"- **Text (last 500 chars)**:")
        lines.append(f"```")
        lines.append(excerpt)
        lines.append(f"```")
        lines.append("")

    sample_path = output_dir / "validation_sample.md"
    sample_path.write_text("\n".join(lines))
    logger.info(f"Validation sample saved: {sample_path} ({len(sample)} cases)")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ATHENA2 Label Cleaning Pipeline")
    parser.add_argument("--input", type=Path, default=Path("data/processed/sjp_xl.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--regex-only", action="store_true", help="Skip LLM stage")
    parser.add_argument("--with-llm", action="store_true", help="Run LLM on ambiguous cases")
    parser.add_argument("--report-only", action="store_true", help="Report on existing cleaned data")
    parser.add_argument("--omlx-url", type=str, default="http://localhost:8000")
    parser.add_argument("--llm-model", type=str, default="qwen3.5-35b-a3b-text-hi")
    parser.add_argument("--sample-size", type=int, default=50, help="Validation sample size")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Report-only mode
    if args.report_only:
        cleaned_path = args.output_dir / "sjp_xl_cleaned.parquet"
        if not cleaned_path.exists():
            logger.error(f"No cleaned data at {cleaned_path}")
            return
        df = pd.read_parquet(cleaned_path)
        report = generate_report(df, args.output_dir)
        print(report)
        return

    # Load data
    logger.info(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Build patterns
    patterns = build_patterns()
    total_patterns = sum(len(v) for v in patterns.values())
    logger.info(f"Built {total_patterns} regex patterns across 3 languages")

    # Stage 1: Regex
    stage1_results = run_stage1(df, patterns)
    df = pd.concat([df, stage1_results], axis=1)

    # Report Stage 1 coverage
    has_match = df["label_extracted"].notna()
    logger.info(f"\nStage 1 Results:")
    logger.info(f"  Extracted: {has_match.sum():,}/{len(df):,} ({has_match.mean()*100:.1f}%)")
    for lang in ["de", "fr", "it"]:
        lang_mask = df["language"] == lang
        lang_match = (lang_mask & has_match).sum()
        lang_total = lang_mask.sum()
        logger.info(f"  {lang.upper()}: {lang_match:,}/{lang_total:,} ({lang_match/lang_total*100:.1f}%)")

    # Agreement with original labels
    agree = (has_match & (df["label_extracted"] == df["label"])).sum()
    disagree = (has_match & (df["label_extracted"] != df["label"])).sum()
    logger.info(f"  Agrees with original: {agree:,} ({agree/has_match.sum()*100:.1f}%)")
    logger.info(f"  Disagrees with original: {disagree:,} ({disagree/has_match.sum()*100:.1f}%)")

    # Stage 2: LLM (optional)
    if args.with_llm and not args.regex_only:
        ambiguous_mask = df["label_extracted"].isna() & (
            df["considerations"].str.len() > 100
        )
        n_ambiguous = ambiguous_mask.sum()
        logger.info(f"\n{n_ambiguous:,} cases need LLM extraction")

        if n_ambiguous > 0:
            llm_results = run_stage2(df, ambiguous_mask, args.omlx_url, args.llm_model)
            df["llm_label"] = llm_results["llm_label"]
            df["llm_error"] = llm_results["llm_error"]

    # Stage 3: Reconciliation
    df = reconcile(df)

    # Save cleaned dataset
    output_path = args.output_dir / "sjp_xl_cleaned.parquet"
    # Select columns to save
    save_cols = [c for c in df.columns if c not in ("extraction_matches",)]
    df[save_cols].to_parquet(output_path, index=False)
    logger.info(f"\nCleaned dataset saved: {output_path}")

    # Generate report
    report = generate_report(df, args.output_dir)
    print("\n" + report)

    # Save validation sample
    save_validation_sample(df, args.output_dir, n=args.sample_size)

    # Save summary stats as JSON
    stats = {
        "total_cases": len(df),
        "verified": int((df["label_status"] == "verified").sum()),
        "corrected": int((df["label_status"] == "corrected").sum()),
        "ambiguous": int((df["label_status"] == "ambiguous").sum()),
        "no_extraction": int((df["label_status"] == "no_extraction").sum()),
        "no_considerations": int((df["label_status"] == "no_considerations").sum()),
        "regex_coverage": float(has_match.mean()),
        "correction_rate": float((df["label_status"] == "corrected").mean()),
    }
    (args.output_dir / "cleaning_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats saved: {args.output_dir / 'cleaning_stats.json'}")


if __name__ == "__main__":
    main()
