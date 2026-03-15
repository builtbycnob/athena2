"""Tests for ATHENA2 feature extraction."""

import pytest

from athena2.features.regex_features import (
    RegexFeatures,
    extract_batch,
    extract_regex_features,
)
from athena2.features.citation_graph import (
    CitationGraph,
    extract_citations_from_ner,
    normalize_bge_ref,
)


# ── Regex Feature Extraction ──────────────────────────────────────

class TestBGECitations:
    def test_german_bge(self):
        text = "Wie das Bundesgericht in BGE 144 III 120 E. 5.1 entschieden hat"
        features = extract_regex_features("test-1", text)
        assert features.n_bge_citations == 1
        assert features.bge_citations[0]["volume"] == "144"
        assert features.bge_citations[0]["part"] == "III"
        assert features.bge_citations[0]["page"] == "120"
        assert features.bge_citations[0]["consideration"] == "5.1"

    def test_french_dtf(self):
        text = "conformément à la DTF 140 II 233"
        features = extract_regex_features("test-2", text)
        assert features.n_bge_citations == 1
        assert features.bge_citations[0]["type"] == "DTF"

    def test_italian_atf(self):
        text = "secondo la ATF 139 I 16"
        features = extract_regex_features("test-3", text)
        assert features.n_bge_citations == 1
        assert features.bge_citations[0]["type"] == "ATF"

    def test_multiple_citations(self):
        text = "BGE 144 III 120 und BGE 140 II 233 sowie BGE 139 I 16"
        features = extract_regex_features("test-4", text)
        assert features.n_bge_citations == 3

    def test_no_citations(self):
        text = "Der Beschwerdeführer macht geltend, die Vorinstanz habe falsch entschieden."
        features = extract_regex_features("test-5", text)
        assert features.n_bge_citations == 0


class TestSRReferences:
    def test_basic_sr(self):
        text = "gemäss Art. 8 SR 210 (ZGB)"
        features = extract_regex_features("test-1", text)
        assert features.n_sr_references == 1
        assert "210" in features.sr_references

    def test_dotted_sr(self):
        text = "Art. 146 SR 311.0 (StGB)"
        features = extract_regex_features("test-2", text)
        assert "311.0" in features.sr_references

    def test_dedup(self):
        text = "SR 210 und wiederum SR 210"
        features = extract_regex_features("test-3", text)
        assert features.n_sr_references == 1


class TestArticleReferences:
    def test_basic_article(self):
        text = "Art. 8 ZGB"
        features = extract_regex_features("test-1", text)
        assert features.n_article_references >= 1
        ref = features.article_references[0]
        assert ref["article"] == "8"
        assert ref["law"] == "ZGB"

    def test_with_paragraph(self):
        text = "Art. 29 Abs. 2 BV"
        features = extract_regex_features("test-2", text)
        assert features.n_article_references >= 1
        ref = features.article_references[0]
        assert ref["article"] == "29"
        assert ref["paragraph"] == "2"

    def test_unique_laws_count(self):
        text = "Art. 8 ZGB und Art. 41 OR sowie Art. 29 BV"
        features = extract_regex_features("test-3", text)
        assert features.n_unique_laws == 3


class TestOutcomeIndicators:
    def test_german_dismissal(self):
        features = extract_regex_features(
            "test-1", "facts", "Die Beschwerde ist abgewiesen."
        )
        assert features.has_outcome_indicator is True

    def test_german_approval(self):
        features = extract_regex_features(
            "test-2", "facts", "Die Beschwerde wird gutgeheissen."
        )
        assert features.has_outcome_indicator is True

    def test_french_rejection(self):
        features = extract_regex_features(
            "test-3", "facts", "Le recours est rejeté."
        )
        assert features.has_outcome_indicator is True

    def test_italian_approval(self):
        features = extract_regex_features(
            "test-4", "facts", "Il ricorso è accolto."
        )
        assert features.has_outcome_indicator is True

    def test_no_indicator(self):
        features = extract_regex_features(
            "test-5", "facts", "Die Erwägungen des Gerichts."
        )
        assert features.has_outcome_indicator is False


class TestBatchExtraction:
    def test_batch(self):
        rows = [
            {"decision_id": "1", "facts": "BGE 144 III 120", "considerations": ""},
            {"decision_id": "2", "facts": "Art. 8 ZGB", "considerations": ""},
        ]
        results = extract_batch(rows)
        assert len(results) == 2
        assert results[0]["n_bge_citations"] == 1
        assert results[1]["n_article_references"] >= 1


# ── Citation Graph ─────────────────────────────────────────────────

class TestNormalizeBGE:
    def test_basic(self):
        assert normalize_bge_ref("BGE 144 III 120") == "BGE_144_III_120"

    def test_french(self):
        assert normalize_bge_ref("DTF 140 II 233") == "BGE_140_II_233"

    def test_no_match(self):
        assert normalize_bge_ref("no citation here") is None


class TestNERExtraction:
    def test_basic_citation(self):
        tokens = ["see", "BGE", "144", "III", "120", "for", "details"]
        labels = [0, 1, 2, 2, 2, 0, 0]  # O, B-CIT, I-CIT, I-CIT, I-CIT, O, O
        citations = extract_citations_from_ner(tokens, labels)
        assert len(citations) == 1
        assert citations[0]["type"] == "CITATION"
        assert "BGE" in citations[0]["text"]

    def test_law_reference(self):
        tokens = ["Art.", "8", "ZGB", "provides"]
        labels = [3, 4, 4, 0]  # B-LAW, I-LAW, I-LAW, O
        citations = extract_citations_from_ner(tokens, labels)
        assert len(citations) == 1
        assert citations[0]["type"] == "LAW"

    def test_multiple(self):
        tokens = ["BGE", "144", "III", "and", "Art.", "8"]
        labels = [1, 2, 2, 0, 3, 4]
        citations = extract_citations_from_ner(tokens, labels)
        assert len(citations) == 2

    def test_empty(self):
        citations = extract_citations_from_ner([], [])
        assert citations == []


class TestCitationGraph:
    def test_build_from_regex(self):
        rows = [
            {
                "decision_id": "case-1",
                "considerations": "BGE 144 III 120 und BGE 140 II 233",
                "law_area": "civil_law",
                "year": 2020,
                "language": "de",
                "label": 0,
            },
            {
                "decision_id": "case-2",
                "considerations": "BGE 144 III 120",
                "law_area": "penal_law",
                "year": 2021,
                "language": "de",
                "label": 1,
            },
        ]
        graph = CitationGraph()
        graph.build_from_regex(rows)

        assert graph.n_nodes == 2
        assert graph.n_edges == 3  # case-1→2 cites, case-2→1 cite
        assert "BGE_144_III_120" in graph.bge_to_decisions

    def test_statistics(self):
        rows = [
            {"decision_id": "case-1", "considerations": "BGE 144 III 120",
             "law_area": "", "year": 2020, "language": "de", "label": 0},
        ]
        graph = CitationGraph()
        graph.build_from_regex(rows)
        stats = graph.compute_statistics()

        assert stats["n_nodes"] == 1
        assert stats["n_edges"] == 1
        assert stats["n_unique_bge"] == 1
