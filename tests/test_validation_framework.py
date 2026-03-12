# tests/test_validation_framework.py
"""Tests for the validation framework: extraction, validation, scoring."""

import json
from pathlib import Path

import pytest
import yaml

from athena.validation.ground_truth import (
    GroundTruth,
    SWISS_LABEL_MAP,
    load_ground_truths,
    save_ground_truth,
)
from athena.validation.enricher import (
    get_applicable_law,
    get_party_templates,
    get_procedural_rules,
)
from athena.validation.validator import validate_case_yaml, validate_case_dict
from athena.validation.scorer import (
    CaseScore,
    ValidationReport,
    _compute_outcome_probabilities,
)
from athena.validation.case_extractor import (
    extract_case_deterministic,
    _merge_extraction,
    _parse_json_response,
    extract_and_save,
)


# --- Fixtures ---

SAMPLE_RECORD = {
    "id": "12345",
    "text": "Il Tribunale federale ha respinto il ricorso del ricorrente...",
    "label": 0,
    "legal_area": "civil_law",
    "year": 2018,
    "region": "Ticino",
    "canton": "TI",
    "language": "it",
}

SAMPLE_EXTRACTION = {
    "facts_undisputed": [
        {"id": "F1", "description": "Il ricorrente ha presentato ricorso", "evidence": ["DOC1"]},
    ],
    "facts_disputed": [
        {
            "id": "D1",
            "description": "Applicabilità della norma",
            "appellant_position": "Norma inapplicabile",
            "respondent_position": "Norma applicabile",
            "depends_on_facts": ["F1"],
        },
    ],
    "evidence": [
        {
            "id": "DOC1",
            "type": "atto_pubblico",
            "description": "Decisione cantonale",
            "produced_by": "ricorrente",
            "admissibility": "uncontested",
            "supports_facts": ["F1"],
        },
    ],
    "seed_arguments_appellant": [
        {"id": "SEED_ARG1", "claim": "Violazione del diritto federale",
         "direction": "Ricorso fondato", "references_facts": ["F1", "D1"]},
    ],
    "seed_arguments_respondent": [
        {"id": "SEED_RARG1", "claim": "Decisione conforme",
         "direction": "Ricorso infondato", "references_facts": ["F1"]},
    ],
    "key_precedents": [
        {"id": "prec_1", "citation": "DTF 140 III 86", "holding": "Principio applicabile", "weight": "binding"},
    ],
    "legal_texts_cited": ["art. 8 CC", "art. 29 Cost."],
    "stakes_description": "Contestazione di una decisione cantonale in materia civile",
    "timeline": [
        {"date": "2018-01-15", "event": "Decisione cantonale"},
        {"date": "2018-03-01", "event": "Ricorso al TF"},
    ],
}


# --- Ground Truth ---

class TestGroundTruth:
    def test_swiss_label_map(self):
        assert SWISS_LABEL_MAP[0] == "rejection"
        assert SWISS_LABEL_MAP[1] == "annulment"

    def test_save_load_roundtrip(self, tmp_path):
        gt = GroundTruth(
            case_id="ch-12345",
            source="swiss_judgment_prediction",
            outcome="rejection",
            outcome_raw=0,
            extraction_confidence="high",
            legal_area="civil_law",
            year=2018,
            canton="TI",
        )
        path = save_ground_truth(gt, tmp_path / "gt")
        assert path.exists()

        loaded = load_ground_truths(tmp_path / "gt")
        assert "ch-12345" in loaded
        assert loaded["ch-12345"].outcome == "rejection"
        assert loaded["ch-12345"].year == 2018

    def test_load_empty_dir(self, tmp_path):
        gt_dir = tmp_path / "empty"
        gt_dir.mkdir()
        assert load_ground_truths(gt_dir) == {}

    def test_load_nonexistent_dir(self, tmp_path):
        assert load_ground_truths(tmp_path / "nonexistent") == {}


# --- Enricher ---

class TestEnricher:
    def test_swiss_procedural_rules(self):
        rules = get_procedural_rules("CH", "bundesgericht")
        assert rules.rite == "ricorso_tribunale_federale"
        assert "ricorso" in rules.phases

    def test_italian_procedural_rules(self):
        rules = get_procedural_rules("IT", "giudice_di_pace")
        assert rules.rite == "opposizione_sanzione_amministrativa"

    def test_applicable_law_civil(self):
        laws = get_applicable_law("CH", "civil_law")
        assert any("CC" in law for law in laws)

    def test_applicable_law_penal(self):
        laws = get_applicable_law("CH", "penal_law")
        assert any("CP" in law for law in laws)

    def test_party_templates(self):
        templates = get_party_templates("CH")
        assert "appellant" in templates
        assert "respondent" in templates
        assert templates["appellant"]["id"] == "ricorrente"


# --- Case Extractor ---

class TestCaseExtractor:
    def test_extract_deterministic(self):
        case_data, gt = extract_case_deterministic(SAMPLE_RECORD)

        assert case_data["case_id"] == "ch-12345"
        assert case_data["jurisdiction"]["country"] == "CH"
        assert case_data["jurisdiction"]["court"] == "bundesgericht"
        assert case_data["jurisdiction"]["venue"] == "TI"

        assert gt.case_id == "ch-12345"
        assert gt.outcome == "rejection"
        assert gt.outcome_raw == 0
        assert gt.source == "swiss_judgment_prediction"

    def test_extract_deterministic_approval(self):
        record = dict(SAMPLE_RECORD, label=1)
        _, gt = extract_case_deterministic(record)
        assert gt.outcome == "annulment"

    def test_merge_extraction(self):
        case_data, _ = extract_case_deterministic(SAMPLE_RECORD)
        merged = _merge_extraction(case_data, SAMPLE_EXTRACTION, SAMPLE_RECORD)

        assert len(merged["parties"]) == 2
        assert len(merged["evidence"]) == 1
        assert len(merged["facts"]["undisputed"]) == 1
        assert len(merged["facts"]["disputed"]) == 1
        assert "ricorrente" in merged["seed_arguments"]["by_party"]
        assert "controparte" in merged["seed_arguments"]["by_party"]
        assert len(merged["key_precedents"]) == 1
        assert len(merged["legal_texts"]) == 2
        assert merged["stakes"]["litigation_cost_estimate"] == 2000

    def test_merge_extraction_disputed_facts_positions(self):
        """Disputed facts are converted to positions dict."""
        case_data, _ = extract_case_deterministic(SAMPLE_RECORD)
        merged = _merge_extraction(case_data, SAMPLE_EXTRACTION, SAMPLE_RECORD)

        disputed = merged["facts"]["disputed"][0]
        assert "positions" in disputed
        assert "ricorrente" in disputed["positions"]

    def test_parse_json_response_direct(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_response_with_text(self):
        result = _parse_json_response('Here is the JSON:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_parse_json_response_invalid(self):
        result = _parse_json_response("not json at all")
        assert isinstance(result, str)

    def test_extract_and_save_no_llm(self, tmp_path):
        cases_dir = tmp_path / "cases"
        gt_dir = tmp_path / "gt"

        yaml_path, gt_path = extract_and_save(
            SAMPLE_RECORD, cases_dir, gt_dir, use_llm=False,
        )

        assert yaml_path.exists()
        assert gt_path.exists()

        # Verify YAML is parseable
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert "case" in data
        assert data["case"]["id"] == "ch-12345"

        # Verify ground truth
        gt_data = json.loads(gt_path.read_text())
        assert gt_data["outcome"] == "rejection"

    def test_extract_and_save_with_mock_llm(self, tmp_path):
        """Test full pipeline with a mock LLM that returns our sample extraction."""
        def mock_invoke(**kwargs):
            return json.dumps(SAMPLE_EXTRACTION)

        yaml_path, gt_path = extract_and_save(
            SAMPLE_RECORD,
            tmp_path / "cases",
            tmp_path / "gt",
            use_llm=True,
            invoke_fn=mock_invoke,
        )

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        case = data["case"]
        assert len(case["evidence"]) == 1
        assert len(case["facts"]["undisputed"]) == 1


# --- Validator ---

class TestValidator:
    def _make_valid_case_yaml(self, path: Path) -> Path:
        """Create a minimal valid case YAML."""
        case = {
            "case": {
                "id": "test-001",
                "jurisdiction": {
                    "country": "CH",
                    "court": "bundesgericht",
                    "venue": "Bern",
                    "applicable_law": ["LTF"],
                    "key_precedents": [],
                    "procedural_rules": {
                        "rite": "ricorso",
                        "phases": ["ricorso", "risposta"],
                        "allowed_moves": {"appellant": ["memoria"], "respondent": ["risposta"]},
                    },
                },
                "parties": [
                    {"id": "ricorrente", "role": "appellant", "type": "persona_fisica",
                     "objectives": {"primary": "accoglimento", "subordinate": "rinvio"}},
                    {"id": "controparte", "role": "respondent", "type": "autorita",
                     "objectives": {"primary": "rigetto", "subordinate": "conferma"}},
                ],
                "stakes": {
                    "current_sanction": {"norm": "decisione", "fine_range": [0, 0], "points_deducted": 0},
                    "alternative_sanction": {"norm": "esito", "fine_range": [0, 0], "points_deducted": 0},
                    "litigation_cost_estimate": 1000,
                },
                "evidence": [
                    {"id": "DOC1", "type": "atto", "description": "Decisione cantonale",
                     "produced_by": "ricorrente", "admissibility": "uncontested", "supports_facts": ["F1"]},
                ],
                "facts": {
                    "undisputed": [{"id": "F1", "description": "Fatto accertato", "evidence": ["DOC1"]}],
                    "disputed": [],
                },
                "legal_texts": [{"id": "norm_1", "norm": "Art. 1 LTF", "text": "Testo"}],
                "seed_arguments": {
                    "by_party": {
                        "ricorrente": [{"id": "SEED_ARG1", "claim": "Violazione", "direction": "Fondato",
                                        "references_facts": ["F1"]}],
                        "controparte": [{"id": "SEED_RARG1", "claim": "Conforme", "direction": "Infondato",
                                         "references_facts": ["F1"]}],
                    }
                },
                "timeline": [{"date": "2018-01-01", "event": "Decisione"}],
            }
        }
        yaml_path = path / "test-001.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(case, f, default_flow_style=False, allow_unicode=True)
        return yaml_path

    def test_valid_case(self, tmp_path):
        yaml_path = self._make_valid_case_yaml(tmp_path)
        result = validate_case_yaml(yaml_path)
        assert result.valid, f"Errors: {result.errors}"
        assert result.errors == []

    def test_missing_parties(self, tmp_path):
        yaml_path = self._make_valid_case_yaml(tmp_path)
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        data["case"]["parties"] = [data["case"]["parties"][0]]  # only 1 party
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        result = validate_case_yaml(yaml_path)
        assert not result.valid
        assert any("at least 2 parties" in e for e in result.errors)

    def test_bad_evidence_reference(self, tmp_path):
        yaml_path = self._make_valid_case_yaml(tmp_path)
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        data["case"]["facts"]["undisputed"][0]["evidence"] = ["NONEXISTENT"]
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        result = validate_case_yaml(yaml_path)
        assert any("unknown evidence" in e for e in result.errors)

    def test_validate_case_dict(self):
        case_data = {
            "case_id": "test-001",
            "jurisdiction": {
                "country": "CH", "court": "bundesgericht", "venue": "Bern",
                "applicable_law": ["LTF"],
                "key_precedents": [],
                "procedural_rules": {"rite": "ricorso", "phases": ["ricorso"], "allowed_moves": {}},
            },
            "parties": [
                {"id": "a", "role": "appellant", "type": "pf", "objectives": {"primary": "x", "subordinate": "y"}},
                {"id": "b", "role": "respondent", "type": "pa", "objectives": {"primary": "x", "subordinate": "y"}},
            ],
            "stakes": {
                "current_sanction": {"norm": "n", "fine_range": [0, 0], "points_deducted": 0},
                "alternative_sanction": {"norm": "n", "fine_range": [0, 0], "points_deducted": 0},
                "litigation_cost_estimate": 1000,
            },
            "evidence": [],
            "facts": {"undisputed": [], "disputed": []},
            "legal_texts": [],
            "seed_arguments": {"by_party": {"a": [{"id": "S1", "claim": "c", "direction": "d", "references_facts": []}]}},
            "timeline": [],
        }
        result = validate_case_dict(case_data)
        assert result.valid


# --- Scorer ---

class TestScorer:
    def _make_gt(self, case_id: str, outcome: str) -> GroundTruth:
        return GroundTruth(
            case_id=case_id,
            source="swiss_judgment_prediction",
            outcome=outcome,
            outcome_raw=0 if outcome == "rejection" else 1,
            extraction_confidence="high",
        )

    def test_case_score_correct(self):
        gt = self._make_gt("ch-1", "rejection")
        score = CaseScore("ch-1", gt, "rejection", 0.8, 0.2)
        assert score.correct
        assert score.predicted_probability == 0.8

    def test_case_score_incorrect(self):
        gt = self._make_gt("ch-1", "rejection")
        score = CaseScore("ch-1", gt, "annulment", 0.3, 0.7)
        assert not score.correct
        assert score.predicted_probability == 0.3

    def test_report_accuracy(self):
        scores = [
            CaseScore("ch-1", self._make_gt("ch-1", "rejection"), "rejection", 0.8, 0.2),
            CaseScore("ch-2", self._make_gt("ch-2", "annulment"), "annulment", 0.3, 0.7),
            CaseScore("ch-3", self._make_gt("ch-3", "rejection"), "annulment", 0.4, 0.6),
        ]
        report = ValidationReport(scores)
        assert report.accuracy == pytest.approx(2 / 3)
        assert report.n == 3

    def test_report_accuracy_ci(self):
        scores = [
            CaseScore("ch-1", self._make_gt("ch-1", "rejection"), "rejection", 0.8, 0.2),
            CaseScore("ch-2", self._make_gt("ch-2", "annulment"), "annulment", 0.3, 0.7),
        ]
        report = ValidationReport(scores)
        ci_low, ci_high = report.accuracy_ci
        assert 0.0 <= ci_low <= 1.0
        assert ci_low <= ci_high

    def test_report_log_loss(self):
        scores = [
            CaseScore("ch-1", self._make_gt("ch-1", "rejection"), "rejection", 0.9, 0.1),
        ]
        report = ValidationReport(scores)
        import math
        assert report.log_loss == pytest.approx(-math.log(0.9), abs=0.01)

    def test_report_ece(self):
        scores = [
            CaseScore("ch-1", self._make_gt("ch-1", "rejection"), "rejection", 0.9, 0.1),
            CaseScore("ch-2", self._make_gt("ch-2", "rejection"), "rejection", 0.9, 0.1),
        ]
        report = ValidationReport(scores)
        assert report.ece < 0.2

    def test_report_empty(self):
        report = ValidationReport([])
        assert report.accuracy == 0.0
        assert report.log_loss == float("inf")
        assert report.ece == 0.0

    def test_report_markdown(self):
        scores = [
            CaseScore("ch-1", self._make_gt("ch-1", "rejection"), "rejection", 0.8, 0.2),
            CaseScore("ch-2", self._make_gt("ch-2", "annulment"), "rejection", 0.6, 0.4),
        ]
        report = ValidationReport(scores)
        md = report.to_markdown()
        assert "Validation Report" in md
        assert "Accuracy" in md
        assert "Error Analysis" in md

    def test_compute_outcome_probabilities(self):
        results = [
            {"judge_decision": {"verdict": {"qualification_correct": True}}},
            {"judge_decision": {"verdict": {"qualification_correct": True}}},
            {"judge_decision": {"verdict": {"qualification_correct": False, "if_incorrect": {"consequence": "annulment"}}}},
        ]
        p_rej, p_ann = _compute_outcome_probabilities(results)
        assert p_rej == pytest.approx(2 / 3)
        assert p_ann == pytest.approx(1 / 3)

    def test_compute_outcome_probabilities_empty(self):
        p_rej, p_ann = _compute_outcome_probabilities([])
        assert p_rej == 0.5
        assert p_ann == 0.5

    def test_stratify_by_legal_area(self):
        gt1 = GroundTruth(case_id="ch-1", source="s", outcome="rejection", outcome_raw=0,
                          extraction_confidence="high", legal_area="civil_law")
        gt2 = GroundTruth(case_id="ch-2", source="s", outcome="annulment", outcome_raw=1,
                          extraction_confidence="high", legal_area="penal_law")
        scores = [
            CaseScore("ch-1", gt1, "rejection", 0.8, 0.2),
            CaseScore("ch-2", gt2, "annulment", 0.3, 0.7),
        ]
        report = ValidationReport(scores)
        by_area = report.stratify_by("legal_area")
        assert "civil_law" in by_area
        assert "penal_law" in by_area
        assert by_area["civil_law"].n == 1

    def test_error_analysis(self):
        scores = [
            CaseScore("ch-1", self._make_gt("ch-1", "rejection"), "annulment", 0.4, 0.6),
        ]
        report = ValidationReport(scores)
        errors = report.error_analysis()
        assert len(errors) == 1
        assert errors[0]["case_id"] == "ch-1"
        assert errors[0]["expected"] == "rejection"
        assert errors[0]["predicted"] == "annulment"

    def test_score_results_e2e(self, tmp_path):
        """End-to-end: create results + ground truth, score them."""
        gt_dir = tmp_path / "gt"
        gt = self._make_gt("ch-1", "rejection")
        save_ground_truth(gt, gt_dir)

        results_dir = tmp_path / "results"
        case_dir = results_dir / "ch-1"
        case_dir.mkdir(parents=True)
        raw_results = [
            {"judge_decision": {"verdict": {"qualification_correct": True}}},
            {"judge_decision": {"verdict": {"qualification_correct": True}}},
            {"judge_decision": {"verdict": {"qualification_correct": False}}},
        ]
        (case_dir / "raw_results.json").write_text(json.dumps(raw_results))

        from athena.validation.scorer import score_results
        report = score_results(results_dir, gt_dir)
        assert report.n == 1
        assert report.scores[0].correct  # 2/3 rejection → predicted rejection
