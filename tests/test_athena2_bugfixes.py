"""Tests for bug fixes in ATHENA2 codebase.

Covers:
1. citation_gat.py — string label handling in build_node_features()
2. citation_gat.py — build_edge_index() with empty/invalid edges
3. llm_features.py — integer-encoded column alignment in LLMFeatures.to_dict()
4. llm_features.py — JSON repair logic (markdown fences, trailing commas)
5. phase5_citation.py — build_edge_index() with BGE keys not in node_map
"""

import json
import re

import pytest
import torch

from athena2.features.llm_features import LLMFeatures
from athena2.models.citation_gat import build_edge_index, build_node_features


# ── Bug 1: citation_gat.py string label handling ─────────────────


class TestBuildNodeFeaturesStringLabels:
    """build_node_features() must handle string labels from SJP-XL dataset."""

    def test_string_label_dismissal(self):
        """'dismissal' string label should map to 0.0, not crash."""
        nodes = [
            {"decision_id": "case_1", "year": 2020, "label": "dismissal"},
        ]
        features, node_map = build_node_features(nodes)
        assert features[0, 10] == 0.0

    def test_string_label_approval(self):
        """'approval' string label should map to 1.0."""
        nodes = [
            {"decision_id": "case_1", "year": 2020, "label": "approval"},
        ]
        features, node_map = build_node_features(nodes)
        assert features[0, 10] == 1.0

    def test_string_label_unknown_ignored(self):
        """Unknown string labels should be ignored (feature stays 0.0)."""
        nodes = [
            {"decision_id": "case_1", "year": 2020, "label": "unknown_label"},
        ]
        features, node_map = build_node_features(nodes)
        # Unknown label maps to -1, then skipped via continue
        assert features[0, 10] == 0.0

    def test_numeric_label_still_works(self):
        """Numeric labels (legacy path) must still work."""
        nodes = [
            {"decision_id": "case_1", "year": 2020, "label": 1},
            {"decision_id": "case_2", "year": 2020, "label": 0},
        ]
        features, node_map = build_node_features(nodes)
        assert features[0, 10] == 1.0
        assert features[1, 10] == 0.0

    def test_mixed_string_and_numeric_labels(self):
        """Mix of string and numeric labels should all work."""
        nodes = [
            {"decision_id": "c1", "year": 2020, "label": "dismissal"},
            {"decision_id": "c2", "year": 2020, "label": 1},
            {"decision_id": "c3", "year": 2020, "label": "approval"},
            {"decision_id": "c4", "year": 2020, "label": 0},
        ]
        features, node_map = build_node_features(nodes)
        assert features[0, 10] == 0.0  # dismissal
        assert features[1, 10] == 1.0  # numeric 1
        assert features[2, 10] == 1.0  # approval
        assert features[3, 10] == 0.0  # numeric 0

    def test_none_label_no_crash(self):
        """None label should be skipped gracefully."""
        nodes = [
            {"decision_id": "case_1", "year": 2020, "label": None},
        ]
        features, node_map = build_node_features(nodes)
        assert features[0, 10] == 0.0

    def test_no_label_key(self):
        """Missing label key should be handled (stays 0.0)."""
        nodes = [
            {"decision_id": "case_1", "year": 2020},
        ]
        features, node_map = build_node_features(nodes)
        assert features[0, 10] == 0.0


# ── Bug 2: build_edge_index with BGE keys / invalid targets ──────


class TestBuildEdgeIndexInvalidTargets:
    """build_edge_index() drops edges whose source/target aren't in node_map."""

    def test_bge_key_target_dropped(self):
        """Edges targeting BGE keys (not in node_map) are dropped."""
        node_map = {"case_1": 0, "case_2": 1}
        edges = [
            {"source": "case_1", "target": "BGE_144_III_120"},
        ]
        edge_index = build_edge_index(edges, node_map)
        assert edge_index.shape == (2, 0)

    def test_valid_edges_kept(self):
        """Edges where both source and target are valid node IDs are kept."""
        node_map = {"case_1": 0, "case_2": 1, "case_3": 2}
        edges = [
            {"source": "case_1", "target": "case_2"},
            {"source": "case_2", "target": "case_3"},
        ]
        edge_index = build_edge_index(edges, node_map)
        assert edge_index.shape == (2, 2)
        assert edge_index[0].tolist() == [0, 1]
        assert edge_index[1].tolist() == [1, 2]

    def test_mix_valid_and_bge_edges(self):
        """Only valid edges are kept; BGE-targeted edges are dropped."""
        node_map = {"case_1": 0, "case_2": 1}
        edges = [
            {"source": "case_1", "target": "case_2"},        # valid
            {"source": "case_1", "target": "BGE_100_II_50"},  # dropped
            {"source": "case_2", "target": "BGE_99_I_10"},    # dropped
        ]
        edge_index = build_edge_index(edges, node_map)
        assert edge_index.shape == (2, 1)
        assert edge_index[0].tolist() == [0]
        assert edge_index[1].tolist() == [1]


# ── Bug 5: build_edge_index with empty edges ─────────────────────


class TestBuildEdgeIndexEmpty:
    """build_edge_index() returns shape (2, 0) when no valid edges exist."""

    def test_empty_edge_list(self):
        node_map = {"case_1": 0}
        edge_index = build_edge_index([], node_map)
        assert edge_index.shape == (2, 0)
        assert edge_index.dtype == torch.long

    def test_all_edges_invalid(self):
        node_map = {"case_1": 0}
        edges = [
            {"source": "case_1", "target": "nonexistent"},
            {"source": "ghost", "target": "case_1"},
        ]
        edge_index = build_edge_index(edges, node_map)
        assert edge_index.shape == (2, 0)


# ── Bug 3: llm_features.py column name alignment + integer encoding


class TestLLMFeaturesToDict:
    """LLMFeatures.to_dict() must produce integer-encoded keys for phase3."""

    def test_required_keys_present(self):
        """to_dict() must include 'error', 'reasoning', 'outcome' integer keys."""
        feat = LLMFeatures(decision_id="test_1")
        d = feat.to_dict()
        assert "error" in d
        assert "reasoning" in d
        assert "outcome" in d

    def test_error_key_is_int(self):
        feat = LLMFeatures(decision_id="test_1")
        d = feat.to_dict()
        assert isinstance(d["error"], int)

    def test_reasoning_key_is_int(self):
        feat = LLMFeatures(decision_id="test_1")
        d = feat.to_dict()
        assert isinstance(d["reasoning"], int)

    def test_outcome_key_is_int(self):
        feat = LLMFeatures(decision_id="test_1")
        d = feat.to_dict()
        assert isinstance(d["outcome"], int)

    def test_reasoning_de_novo_maps_to_0(self):
        feat = LLMFeatures(
            decision_id="test_1",
            reasoning_pattern="de_novo_review",
        )
        d = feat.to_dict()
        assert d["reasoning"] == 0

    def test_reasoning_arbitrariness_maps_to_1(self):
        feat = LLMFeatures(
            decision_id="test_1",
            reasoning_pattern="arbitrariness_review",
        )
        d = feat.to_dict()
        assert d["reasoning"] == 1

    def test_reasoning_unknown_defaults_to_mixed(self):
        """Unknown reasoning pattern should default to 'mixed' = 8."""
        feat = LLMFeatures(
            decision_id="test_1",
            reasoning_pattern="totally_unknown",
        )
        d = feat.to_dict()
        assert d["reasoning"] == 8

    def test_outcome_full_dismissal_maps_to_0(self):
        feat = LLMFeatures(
            decision_id="test_1",
            outcome_granular="full_dismissal",
        )
        d = feat.to_dict()
        assert d["outcome"] == 0

    def test_outcome_full_approval_maps_to_1(self):
        feat = LLMFeatures(
            decision_id="test_1",
            outcome_granular="full_approval",
        )
        d = feat.to_dict()
        assert d["outcome"] == 1

    def test_outcome_unknown_defaults_to_other(self):
        """Unknown outcome should default to 'other' = 6."""
        feat = LLMFeatures(
            decision_id="test_1",
            outcome_granular="something_weird",
        )
        d = feat.to_dict()
        assert d["outcome"] == 6

    def test_error_decisive_severity_maps_to_1(self):
        """Decisive severity error → error=1."""
        feat = LLMFeatures(
            decision_id="test_1",
            errors_identified=[{"error_type": "fact_finding", "severity": "decisive"}],
        )
        d = feat.to_dict()
        assert d["error"] == 1

    def test_error_significant_severity_maps_to_1(self):
        """Significant severity error → error=1."""
        feat = LLMFeatures(
            decision_id="test_1",
            errors_identified=[{"error_type": "procedural", "severity": "significant"}],
        )
        d = feat.to_dict()
        assert d["error"] == 1

    def test_error_minor_severity_maps_to_0(self):
        """Minor severity error → error=0."""
        feat = LLMFeatures(
            decision_id="test_1",
            errors_identified=[{"error_type": "procedural", "severity": "minor"}],
        )
        d = feat.to_dict()
        assert d["error"] == 0

    def test_error_none_severity_maps_to_0(self):
        """None severity → error=0."""
        feat = LLMFeatures(
            decision_id="test_1",
            errors_identified=[{"error_type": "none", "severity": "none"}],
        )
        d = feat.to_dict()
        assert d["error"] == 0

    def test_no_errors_maps_to_0(self):
        """Empty errors list → error=0."""
        feat = LLMFeatures(decision_id="test_1", errors_identified=[])
        d = feat.to_dict()
        assert d["error"] == 0

    def test_all_reasoning_patterns_have_valid_mapping(self):
        """All valid reasoning patterns should map to values 0-8."""
        patterns = [
            "de_novo_review", "arbitrariness_review", "proportionality_test",
            "balancing_test", "subsumption", "teleological",
            "systematic", "historical", "mixed",
        ]
        for i, pattern in enumerate(patterns):
            feat = LLMFeatures(decision_id="test", reasoning_pattern=pattern)
            d = feat.to_dict()
            assert d["reasoning"] == i, f"{pattern} should map to {i}, got {d['reasoning']}"

    def test_all_outcome_values_have_valid_mapping(self):
        """All valid outcomes should map to values 0-6."""
        outcomes = [
            "full_dismissal", "full_approval", "partial_approval",
            "remand", "inadmissible", "withdrawn", "other",
        ]
        for i, outcome in enumerate(outcomes):
            feat = LLMFeatures(decision_id="test", outcome_granular=outcome)
            d = feat.to_dict()
            assert d["outcome"] == i, f"{outcome} should map to {i}, got {d['outcome']}"


# ── Bug 4: JSON repair logic ─────────────────────────────────────


class TestJSONRepairLogic:
    """Test the JSON repair patterns used in extract_single().

    We test the repair logic directly (strip markdown fences, fix trailing commas)
    without making HTTP calls.
    """

    @staticmethod
    def _repair_json(content: str) -> str:
        """Replicate the JSON repair logic from extract_single()."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        content = re.sub(r",\s*([}\]])", r"\1", content)
        return content

    def test_clean_json_passes_through(self):
        raw = '{"key": "value"}'
        assert json.loads(self._repair_json(raw)) == {"key": "value"}

    def test_markdown_fence_stripped(self):
        raw = '```json\n{"key": "value"}\n```'
        assert json.loads(self._repair_json(raw)) == {"key": "value"}

    def test_markdown_fence_no_lang(self):
        raw = '```\n{"key": "value"}\n```'
        assert json.loads(self._repair_json(raw)) == {"key": "value"}

    def test_trailing_comma_in_object(self):
        raw = '{"a": 1, "b": 2,}'
        assert json.loads(self._repair_json(raw)) == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self):
        raw = '{"items": [1, 2, 3,]}'
        result = json.loads(self._repair_json(raw))
        assert result == {"items": [1, 2, 3]}

    def test_trailing_comma_with_whitespace(self):
        raw = '{"a": 1 ,  }'
        assert json.loads(self._repair_json(raw)) == {"a": 1}

    def test_fence_plus_trailing_comma(self):
        """Both repairs applied together."""
        raw = '```json\n{"a": 1, "b": [2, 3,],}\n```'
        result = json.loads(self._repair_json(raw))
        assert result == {"a": 1, "b": [2, 3]}

    def test_nested_trailing_commas(self):
        raw = '{"outer": {"inner": 1,}, "arr": [1,]}'
        result = json.loads(self._repair_json(raw))
        assert result == {"outer": {"inner": 1}, "arr": [1]}
