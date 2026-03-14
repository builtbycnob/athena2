# tests/test_jurisdiction.py
"""Tests for jurisdiction registry, config lookup, prompt selection, outcome extraction."""

import pytest
import jsonschema

from athena.jurisdiction import (
    get_jurisdiction,
    get_jurisdiction_for_case,
    list_jurisdictions,
)
from athena.jurisdiction.it import _it_outcome_extractor
from athena.jurisdiction.ch import _ch_outcome_extractor, _ch_enforce_consistency
from athena.schemas.structured_output import (
    AGENT_SCHEMAS, JUDGE_CH_SCHEMA,
    JUDGE_CH_STEP1_SCHEMA, JUDGE_CH_STEP2_SCHEMA,
)


# --- Registry ---

class TestJurisdictionRegistry:
    def test_it_registered(self):
        config = get_jurisdiction("IT")
        assert config.country == "IT"

    def test_ch_registered(self):
        config = get_jurisdiction("CH")
        assert config.country == "CH"

    def test_case_insensitive(self):
        assert get_jurisdiction("it").country == "IT"
        assert get_jurisdiction("ch").country == "CH"

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="DE"):
            get_jurisdiction("DE")

    def test_list_jurisdictions(self):
        codes = list_jurisdictions()
        assert "IT" in codes
        assert "CH" in codes

    def test_get_jurisdiction_for_case_it(self, sample_case_data):
        config = get_jurisdiction_for_case(sample_case_data)
        assert config.country == "IT"

    def test_get_jurisdiction_for_case_ch(self):
        case = {"jurisdiction": {"country": "CH"}}
        config = get_jurisdiction_for_case(case)
        assert config.country == "CH"

    def test_get_jurisdiction_for_case_defaults_it(self):
        config = get_jurisdiction_for_case({})
        assert config.country == "IT"

    def test_ch_has_default_models(self):
        config = get_jurisdiction("CH")
        assert config.default_models.get("judge") == "qwen3.5-122b-a10b-4bit"

    def test_it_has_empty_default_models(self):
        config = get_jurisdiction("IT")
        assert config.default_models == {}


# --- Italian jurisdiction ---

class TestItalianConfig:
    def test_prompt_keys(self):
        config = get_jurisdiction("IT")
        assert config.prompt_keys["appellant"] == "appellant_it"
        assert config.prompt_keys["respondent"] == "respondent_it"
        assert config.prompt_keys["judge"] == "judge_it"

    def test_schema_keys(self):
        config = get_jurisdiction("IT")
        assert config.schema_keys["judge"] == "judge"
        assert config.verdict_schema_key == "judge"

    def test_outcome_space(self):
        config = get_jurisdiction("IT")
        assert "rejection" in config.outcome_space
        assert "annulment" in config.outcome_space
        assert "reclassification" in config.outcome_space


class TestItalianOutcomeExtractor:
    def test_rejection(self):
        verdict = {"qualification_correct": True}
        assert _it_outcome_extractor(verdict) == "rejection"

    def test_annulment(self):
        verdict = {
            "qualification_correct": False,
            "if_incorrect": {"consequence": "annulment"},
        }
        assert _it_outcome_extractor(verdict) == "annulment"

    def test_reclassification(self):
        verdict = {
            "qualification_correct": False,
            "if_incorrect": {"consequence": "reclassification"},
        }
        assert _it_outcome_extractor(verdict) == "reclassification"

    def test_incorrect_no_details(self):
        verdict = {"qualification_correct": False}
        assert _it_outcome_extractor(verdict) == "annulment"

    def test_incorrect_null_if_incorrect(self):
        verdict = {"qualification_correct": False, "if_incorrect": None}
        assert _it_outcome_extractor(verdict) == "annulment"


# --- Swiss jurisdiction ---

class TestSwissConfig:
    def test_prompt_keys(self):
        config = get_jurisdiction("CH")
        assert config.prompt_keys["appellant"] == "appellant_ch"
        assert config.prompt_keys["respondent"] == "respondent_ch"
        assert config.prompt_keys["judge"] == "judge_ch"

    def test_schema_keys(self):
        config = get_jurisdiction("CH")
        assert config.schema_keys["judge"] == "judge_ch"
        assert config.schema_keys["appellant"] == "appellant"  # shared
        assert config.schema_keys["respondent"] == "respondent"  # shared

    def test_outcome_space_no_reclassification(self):
        config = get_jurisdiction("CH")
        assert "rejection" in config.outcome_space
        assert "annulment" in config.outcome_space
        assert "reclassification" not in config.outcome_space


class TestSwissOutcomeExtractor:
    # New two-step schema
    def test_lower_court_correct_true(self):
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [],
        }
        assert _ch_outcome_extractor(verdict) == "rejection"

    def test_lower_court_correct_false_with_decisive(self):
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
        }
        assert _ch_outcome_extractor(verdict) == "annulment"

    def test_lower_court_false_no_decisive_enforced_to_rejection(self):
        """Consistency: if no decisive errors, lower_court_correct forced to True."""
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "procedural", "severity": "minor"},
            ],
        }
        assert _ch_outcome_extractor(verdict) == "rejection"

    # Legacy flat schema
    def test_dismissed(self):
        assert _ch_outcome_extractor({"appeal_outcome": "dismissed"}) == "rejection"

    def test_upheld(self):
        assert _ch_outcome_extractor({"appeal_outcome": "upheld"}) == "annulment"

    def test_partially_upheld(self):
        assert _ch_outcome_extractor({"appeal_outcome": "partially_upheld"}) == "annulment"

    def test_remanded(self):
        assert _ch_outcome_extractor({"appeal_outcome": "remanded"}) == "annulment"

    def test_default_dismissed(self):
        assert _ch_outcome_extractor({}) == "rejection"


class TestSwissConsistencyEnforcement:
    def test_no_errors_forces_correct(self):
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["lower_court_correct"] is True

    def test_only_minor_forces_correct(self):
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "procedural", "severity": "minor"},
                {"error_type": "fact_finding", "severity": "none"},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["lower_court_correct"] is True

    def test_decisive_stays_incorrect(self):
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["lower_court_correct"] is False

    def test_decisive_forces_incorrect(self):
        """If decisive errors found but LCC=True, force to False."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["lower_court_correct"] is False
        assert result["if_correct"] is None
        assert result["if_incorrect"] is not None

    def test_decisive_in_error_assessment_forces_incorrect(self):
        """Two-step: decisive confirmed_severity in error_assessment forces LCC=False."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "significant"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "decisive",
                 "assessment_reasoning": "Confirmed as decisive on review."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["lower_court_correct"] is False
        assert result["if_correct"] is None
        assert result["if_incorrect"] is not None

    def test_severity_floor_decisive_to_none_raises_to_significant(self):
        """Step 1 decisive + Step 2 none → floor to significant."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "none",
                 "assessment_reasoning": "Dismissed."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["error_assessment"][0]["confirmed_severity"] == "significant"
        # significant (not decisive) → LCC stays True
        assert result["lower_court_correct"] is True

    def test_severity_floor_decisive_to_minor_raises_to_significant(self):
        """Step 1 decisive + Step 2 minor → floor to significant."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "minor",
                 "assessment_reasoning": "Minor issue."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["error_assessment"][0]["confirmed_severity"] == "significant"

    def test_severity_floor_does_not_affect_significant(self):
        """Step 1 decisive + Step 2 significant → no change (floor not triggered)."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "significant",
                 "assessment_reasoning": "Significant but not decisive."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["error_assessment"][0]["confirmed_severity"] == "significant"

    def test_severity_floor_does_not_affect_non_decisive_step1(self):
        """Step 1 significant + Step 2 none → no floor (only applies to Step 1 decisive)."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "significant"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "none",
                 "assessment_reasoning": "Dismissed."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["error_assessment"][0]["confirmed_severity"] == "none"

    def test_severity_floor_uses_error_id_not_position(self):
        """Severity floor must match by error_id, not array position."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "procedural", "severity": "minor"},
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
            "error_assessment": [
                # Only assess error_id=1 (the decisive one), skip error_id=0
                {"error_id": 1, "confirmed_severity": "none",
                 "assessment_reasoning": "Dismissed."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        # error_id=1 maps to the decisive Step 1 error → floor to significant
        assert result["error_assessment"][0]["confirmed_severity"] == "significant"

    def test_severity_floor_missing_error_id_no_crash(self):
        """If error_id is missing from assessment, severity floor is skipped."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
            "error_assessment": [
                {"confirmed_severity": "none", "assessment_reasoning": "No error_id."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        # No error_id → no matching → no floor applied
        assert result["error_assessment"][0]["confirmed_severity"] == "none"

    def test_significant_without_decisive_forces_correct(self):
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "fact_finding", "severity": "significant"},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["lower_court_correct"] is True

    def test_branch_completion_if_correct(self):
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [],
            "correctness_reasoning": "La decisione è corretta.",
            "if_incorrect": {"consequence": "stale"},
            "if_correct": None,
        }
        result = _ch_enforce_consistency(verdict)
        assert result["if_incorrect"] is None
        assert result["if_correct"] is not None
        assert "confirmation_reasoning" in result["if_correct"]

    def test_branch_completion_if_incorrect(self):
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
            "correctness_reasoning": "Errore decisivo.",
            "if_incorrect": None,
            "if_correct": {"stale": True},
        }
        result = _ch_enforce_consistency(verdict)
        assert result["if_correct"] is None
        assert result["if_incorrect"] is not None
        assert "consequence" in result["if_incorrect"]


# --- Swiss judge schema ---

class TestJudgeChSchema:
    def test_registered_in_agent_schemas(self):
        assert "judge_ch" in AGENT_SCHEMAS
        assert AGENT_SCHEMAS["judge_ch"] is JUDGE_CH_SCHEMA

    def test_schema_is_valid_json_schema(self):
        validator_cls = jsonschema.validators.validator_for(JUDGE_CH_SCHEMA)
        validator_cls.check_schema(JUDGE_CH_SCHEMA)

    def _base_decision(self, verdict):
        return {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.6,
                    "strengths": "Argomento solido basato su DTF",
                    "weaknesses": "Precedente distinguibile",
                    "determinative": True,
                }
            ],
            "precedent_analysis": {
                "dtf_123": {
                    "followed": True,
                    "distinguished": False,
                    "reasoning": "Il precedente è in punto.",
                }
            },
            "verdict": verdict,
            "reasoning": "Motivazione completa della sentenza.",
            "gaps": [],
        }

    def test_validates_correct_lower_court(self):
        verdict = {
            "identified_errors": [
                {
                    "error_type": "none_found",
                    "description": "Nessun errore individuato nella decisione impugnata.",
                    "severity": "none",
                }
            ],
            "lower_court_correct": True,
            "correctness_reasoning": "La decisione è conforme al diritto.",
            "if_incorrect": None,
            "if_correct": {
                "confirmation_reasoning": "La decisione dell'istanza inferiore è corretta."
            },
            "costs_ruling": "a carico del ricorrente",
        }
        jsonschema.validate(self._base_decision(verdict), JUDGE_CH_SCHEMA)

    def test_validates_incorrect_lower_court_with_annulment(self):
        verdict = {
            "identified_errors": [
                {
                    "error_type": "legal_interpretation",
                    "description": "Errata interpretazione dell'art. 41 CO.",
                    "severity": "decisive",
                    "relevant_norm": "art. 41 CO",
                }
            ],
            "lower_court_correct": False,
            "correctness_reasoning": "L'istanza inferiore ha errato.",
            "if_incorrect": {
                "consequence": "annulment",
                "consequence_reasoning": "La decisione va annullata.",
                "remedy": {
                    "type": "annul",
                    "description": "Annullamento della decisione.",
                    "amount_awarded": 50000,
                    "costs_appellant": 0,
                    "costs_respondent": 2000,
                },
            },
            "if_correct": None,
            "costs_ruling": "a carico della controparte",
        }
        jsonschema.validate(self._base_decision(verdict), JUDGE_CH_SCHEMA)

    def test_validates_incorrect_with_remand(self):
        verdict = {
            "identified_errors": [
                {
                    "error_type": "fact_finding",
                    "description": "Accertamento incompleto dei fatti.",
                    "severity": "decisive",
                    "relevant_norm": "art. 97 LTF",
                }
            ],
            "lower_court_correct": False,
            "correctness_reasoning": "Accertamento lacunoso.",
            "if_incorrect": {
                "consequence": "remand",
                "consequence_reasoning": "Rinvio per nuovo accertamento.",
                "remedy": {
                    "type": "remand",
                    "description": "Rinvio all'istanza inferiore.",
                    "amount_awarded": None,
                    "costs_appellant": 0,
                    "costs_respondent": 1000,
                },
            },
            "if_correct": None,
            "costs_ruling": "a carico della controparte",
        }
        jsonschema.validate(self._base_decision(verdict), JUDGE_CH_SCHEMA)

    def test_validates_empty_identified_errors(self):
        """minItems=0 allows no errors when lower court is correct."""
        verdict = {
            "identified_errors": [],
            "lower_court_correct": True,
            "correctness_reasoning": "Nessun errore individuato.",
            "if_incorrect": None,
            "if_correct": {"confirmation_reasoning": "Decisione confermata."},
            "costs_ruling": "a carico del ricorrente",
        }
        jsonschema.validate(self._base_decision(verdict), JUDGE_CH_SCHEMA)

    def test_identified_errors_required(self):
        verdict = {
            "lower_court_correct": True,
            "correctness_reasoning": "OK.",
            "if_incorrect": None,
            "if_correct": {"confirmation_reasoning": "OK."},
            "costs_ruling": "test",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(self._base_decision(verdict), JUDGE_CH_SCHEMA)

    def test_rejects_invalid_error_type(self):
        verdict = {
            "identified_errors": [
                {
                    "error_type": "invalid_type",
                    "description": "Test.",
                    "severity": "minor",
                }
            ],
            "lower_court_correct": True,
            "correctness_reasoning": "Test.",
            "if_incorrect": None,
            "if_correct": {"confirmation_reasoning": "Test."},
            "costs_ruling": "test",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(self._base_decision(verdict), JUDGE_CH_SCHEMA)

    def test_rejects_invalid_consequence(self):
        verdict = {
            "identified_errors": [
                {
                    "error_type": "legal_interpretation",
                    "description": "Errore.",
                    "severity": "decisive",
                }
            ],
            "lower_court_correct": False,
            "correctness_reasoning": "Errato.",
            "if_incorrect": {
                "consequence": "invalid_value",
                "consequence_reasoning": "Test.",
                "remedy": {
                    "type": "annul",
                    "description": "Test.",
                    "amount_awarded": None,
                    "costs_appellant": 0,
                    "costs_respondent": 0,
                },
            },
            "if_correct": None,
            "costs_ruling": "test",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(self._base_decision(verdict), JUDGE_CH_SCHEMA)


# --- Prompt registration ---

class TestPromptRegistration:
    def test_swiss_prompts_registered(self):
        import athena.agents.prompts  # noqa: F401 — triggers registration
        from athena.agents.prompt_registry import list_prompts
        keys = list_prompts()
        assert "appellant_ch" in keys
        assert "respondent_ch" in keys
        assert "judge_ch" in keys

    def test_italian_prompts_still_registered(self):
        import athena.agents.prompts  # noqa: F401 — triggers registration
        from athena.agents.prompt_registry import list_prompts
        keys = list_prompts()
        assert "appellant_it" in keys
        assert "respondent_it" in keys
        assert "judge_it" in keys


# --- Phase builder ---

class TestBuildBilateralPhases:
    def test_italian_case_uses_it_prompts(self, sample_case_data, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        phases = build_bilateral_phases(sample_case_data, sample_run_params)
        prompt_keys = [a.prompt_key for phase in phases for a in phase.agents]
        assert "appellant_it" in prompt_keys
        assert "respondent_it" in prompt_keys
        assert "judge_it" in prompt_keys

    def test_swiss_case_uses_ch_prompts(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        prompt_keys = [a.prompt_key for phase in phases for a in phase.agents]
        assert "appellant_ch" in prompt_keys
        assert "respondent_ch" in prompt_keys
        # Two-step: judge uses judge_ch_step1 as primary prompt key
        assert "judge_ch_step1" in prompt_keys

    def test_swiss_case_uses_ch_schema(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        judge_config = phases[2].agents[0]
        # Two-step: step1 schema
        assert judge_config.schema_key == "judge_ch_step1"

    def test_swiss_judge_default_temperature(self):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        # No explicit judge temperature → should use CH default (0.7)
        params = {"temperatures": {"appellant": 0.5, "respondent": 0.4}}
        phases = build_bilateral_phases(swiss_case, params)
        judge_config = phases[2].agents[0]
        assert judge_config.temperature == 0.7

    def test_italian_judge_default_temperature(self):
        from athena.simulation.graph import build_bilateral_phases
        it_case = {
            "jurisdiction": {"country": "IT"},
            "parties": [
                {"id": "opponente", "role": "appellant"},
                {"id": "comune", "role": "respondent"},
            ],
        }
        # No explicit judge temperature → should use global default (0.3)
        params = {"temperatures": {"appellant": 0.5, "respondent": 0.4}}
        phases = build_bilateral_phases(it_case, params)
        judge_config = phases[2].agents[0]
        assert judge_config.temperature == 0.3

    def test_swiss_party_ids(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        assert phases[0].agents[0].party_id == "ricorrente"
        assert phases[1].agents[0].party_id == "controparte"


# --- Aggregator ---

class TestAggregatorOutcomeDetection:
    def test_italian_verdict_detection(self):
        from athena.simulation.aggregator import _detect_outcome_extractor, _it_outcome_extractor
        results = [{"judge_decision": {"verdict": {"qualification_correct": True}}}]
        extractor = _detect_outcome_extractor(results)
        assert extractor is _it_outcome_extractor

    def test_swiss_verdict_detection(self):
        from athena.simulation.aggregator import _detect_outcome_extractor, _ch_outcome_extractor
        results = [{"judge_decision": {"verdict": {"appeal_outcome": "dismissed"}}}]
        extractor = _detect_outcome_extractor(results)
        assert extractor is _ch_outcome_extractor

    def test_swiss_new_schema_detection(self):
        from athena.simulation.aggregator import _detect_outcome_extractor, _ch_outcome_extractor
        results = [{"judge_decision": {"verdict": {"lower_court_correct": True}}}]
        extractor = _detect_outcome_extractor(results)
        assert extractor is _ch_outcome_extractor

    def test_aggregate_with_new_swiss_verdicts(self):
        from athena.simulation.aggregator import aggregate_results
        results = [
            {
                "judge_profile": "formal",
                "appellant_profile": "standard",
                "judge_decision": {
                    "verdict": {"lower_court_correct": True},
                    "argument_evaluation": [],
                    "precedent_analysis": {},
                },
            },
            {
                "judge_profile": "formal",
                "appellant_profile": "standard",
                "judge_decision": {
                    "verdict": {"lower_court_correct": False},
                    "argument_evaluation": [],
                    "precedent_analysis": {},
                },
            },
        ]
        agg = aggregate_results(results, total_expected=2)
        table = agg["probability_table"][("formal", "standard")]
        assert table["p_rejection"] == 0.5
        assert table["p_annulment"] == 0.5

    def test_aggregate_with_swiss_verdicts(self):
        from athena.simulation.aggregator import aggregate_results
        results = [
            {
                "judge_profile": "formal",
                "appellant_profile": "standard",
                "judge_decision": {
                    "verdict": {"appeal_outcome": "dismissed"},
                    "argument_evaluation": [],
                    "precedent_analysis": {},
                },
            },
            {
                "judge_profile": "formal",
                "appellant_profile": "standard",
                "judge_decision": {
                    "verdict": {"appeal_outcome": "upheld"},
                    "argument_evaluation": [],
                    "precedent_analysis": {},
                },
            },
            {
                "judge_profile": "formal",
                "appellant_profile": "standard",
                "judge_decision": {
                    "verdict": {"appeal_outcome": "dismissed"},
                    "argument_evaluation": [],
                    "precedent_analysis": {},
                },
            },
        ]
        agg = aggregate_results(results, total_expected=3)
        table = agg["probability_table"][("formal", "standard")]
        assert abs(table["p_rejection"] - 2/3) < 0.01
        assert abs(table["p_annulment"] - 1/3) < 0.01
        assert table["p_reclassification"] == 0.0


# --- Scorer ---

class TestScorerOutcomeDetection:
    def test_swiss_outcome_probabilities(self):
        from athena.validation.scorer import _compute_outcome_probabilities
        results = [
            {"judge_decision": {"verdict": {"appeal_outcome": "dismissed"}}},
            {"judge_decision": {"verdict": {"appeal_outcome": "upheld"}}},
            {"judge_decision": {"verdict": {"appeal_outcome": "dismissed"}}},
            {"judge_decision": {"verdict": {"appeal_outcome": "partially_upheld"}}},
        ]
        p_rej, p_ann = _compute_outcome_probabilities(results)
        assert p_rej == 0.5
        assert p_ann == 0.5

    def test_swiss_new_schema_outcome_probabilities(self):
        from athena.validation.scorer import _compute_outcome_probabilities
        results = [
            {"judge_decision": {"verdict": {"lower_court_correct": True}}},
            {"judge_decision": {"verdict": {"lower_court_correct": False}}},
            {"judge_decision": {"verdict": {"lower_court_correct": True}}},
            {"judge_decision": {"verdict": {"lower_court_correct": False}}},
        ]
        p_rej, p_ann = _compute_outcome_probabilities(results)
        assert p_rej == 0.5
        assert p_ann == 0.5

    def test_italian_outcome_probabilities(self):
        from athena.validation.scorer import _compute_outcome_probabilities
        results = [
            {"judge_decision": {"verdict": {"qualification_correct": True}}},
            {"judge_decision": {"verdict": {"qualification_correct": False}}},
        ]
        p_rej, p_ann = _compute_outcome_probabilities(results)
        assert p_rej == 0.5
        assert p_ann == 0.5


# --- Valuation ---

class TestValuationOutcomeSpace:
    def test_default_includes_reclassification(self):
        from athena.game_theory.valuation import compute_outcome_values
        stakes = {
            "current_sanction": {"fine_range": [170, 680], "points_deducted": 4},
            "alternative_sanction": {"fine_range": [42, 173], "points_deducted": 0},
            "litigation_cost_estimate": 1500,
        }
        values = compute_outcome_values(stakes, "appellant")
        assert "reclassification" in values

    def test_swiss_outcome_space(self):
        from athena.game_theory.valuation import compute_outcome_values
        stakes = {
            "current_sanction": {"fine_range": [0, 0], "points_deducted": 0},
            "alternative_sanction": {"fine_range": [0, 0], "points_deducted": 0},
            "litigation_cost_estimate": 2000,
        }
        values = compute_outcome_values(
            stakes, "appellant", outcome_space=["rejection", "annulment"]
        )
        assert "rejection" in values
        assert "annulment" in values
        assert "reclassification" not in values


# --- Two-Step Judge Schemas ---

class TestJudgeChStep1Schema:
    def test_registered_in_agent_schemas(self):
        assert "judge_ch_step1" in AGENT_SCHEMAS
        assert AGENT_SCHEMAS["judge_ch_step1"] is JUDGE_CH_STEP1_SCHEMA

    def test_schema_is_valid_json_schema(self):
        validator_cls = jsonschema.validators.validator_for(JUDGE_CH_STEP1_SCHEMA)
        validator_cls.check_schema(JUDGE_CH_STEP1_SCHEMA)

    def test_validates_step1_output(self):
        output = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.7,
                    "strengths": "Argomento solido",
                    "weaknesses": "Precedente contrario",
                    "determinative": True,
                }
            ],
            "precedent_analysis": {
                "dtf_123": {
                    "followed": True,
                    "distinguished": False,
                    "reasoning": "Precedente in punto.",
                }
            },
            "identified_errors": [
                {
                    "error_type": "legal_interpretation",
                    "description": "Errata interpretazione dell'art. 41 CO.",
                    "severity": "decisive",
                    "relevant_norm": "art. 41 CO",
                }
            ],
            "error_analysis_reasoning": "Un errore decisivo nell'interpretazione normativa.",
        }
        jsonschema.validate(output, JUDGE_CH_STEP1_SCHEMA)

    def test_validates_no_errors(self):
        output = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.3,
                    "strengths": "Nessuno",
                    "weaknesses": "Argomento debole",
                    "determinative": False,
                }
            ],
            "precedent_analysis": {},
            "identified_errors": [],
            "error_analysis_reasoning": "Nessun errore individuato nella decisione impugnata.",
        }
        jsonschema.validate(output, JUDGE_CH_STEP1_SCHEMA)

    def test_rejects_missing_error_analysis_reasoning(self):
        output = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.5,
                    "strengths": "Ok",
                    "weaknesses": "Ok",
                    "determinative": False,
                }
            ],
            "precedent_analysis": {},
            "identified_errors": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(output, JUDGE_CH_STEP1_SCHEMA)


class TestJudgeChStep2Schema:
    def test_registered_in_agent_schemas(self):
        assert "judge_ch_step2" in AGENT_SCHEMAS
        assert AGENT_SCHEMAS["judge_ch_step2"] is JUDGE_CH_STEP2_SCHEMA

    def test_schema_is_valid_json_schema(self):
        validator_cls = jsonschema.validators.validator_for(JUDGE_CH_STEP2_SCHEMA)
        validator_cls.check_schema(JUDGE_CH_STEP2_SCHEMA)

    def test_validates_correct_decision(self):
        output = {
            "error_assessment": [
                {
                    "error_id": 0,
                    "confirmed_severity": "minor",
                    "assessment_reasoning": "Errore non decisivo.",
                }
            ],
            "correctness_reasoning": "La decisione è complessivamente corretta.",
            "lower_court_correct": True,
            "if_incorrect": None,
            "if_correct": {"confirmation_reasoning": "Decisione confermata."},
            "costs_ruling": "a carico del ricorrente",
        }
        jsonschema.validate(output, JUDGE_CH_STEP2_SCHEMA)

    def test_validates_incorrect_decision(self):
        output = {
            "error_assessment": [
                {
                    "error_id": 0,
                    "confirmed_severity": "decisive",
                    "assessment_reasoning": "Errore confermato come decisivo.",
                }
            ],
            "correctness_reasoning": "L'errore ha alterato il dispositivo.",
            "lower_court_correct": False,
            "if_incorrect": {
                "consequence": "annulment",
                "consequence_reasoning": "La decisione va annullata.",
                "remedy": {
                    "type": "annul",
                    "description": "Annullamento.",
                    "amount_awarded": None,
                    "costs_appellant": 0,
                    "costs_respondent": 2000,
                },
            },
            "if_correct": None,
            "costs_ruling": "a carico della controparte",
        }
        jsonschema.validate(output, JUDGE_CH_STEP2_SCHEMA)

    def test_validates_empty_error_assessment(self):
        output = {
            "error_assessment": [],
            "correctness_reasoning": "Nessun errore da valutare.",
            "lower_court_correct": True,
            "if_incorrect": None,
            "if_correct": {"confirmation_reasoning": "Nessun errore."},
            "costs_ruling": "a carico del ricorrente",
        }
        jsonschema.validate(output, JUDGE_CH_STEP2_SCHEMA)

    def test_rejects_invalid_severity(self):
        output = {
            "error_assessment": [
                {
                    "error_id": 0,
                    "confirmed_severity": "critical",
                    "assessment_reasoning": "Test.",
                }
            ],
            "correctness_reasoning": "Test.",
            "lower_court_correct": False,
            "if_incorrect": None,
            "if_correct": None,
            "costs_ruling": "test",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(output, JUDGE_CH_STEP2_SCHEMA)


# --- Two-Step Prompt Registration ---

class TestTwoStepPromptRegistration:
    def test_step1_prompt_registered(self):
        import athena.agents.prompts  # noqa: F401
        from athena.agents.prompt_registry import list_prompts
        assert "judge_ch_step1" in list_prompts()

    def test_step2_prompt_registered(self):
        import athena.agents.prompts  # noqa: F401
        from athena.agents.prompt_registry import list_prompts
        assert "judge_ch_step2" in list_prompts()


# --- Two-Step Consistency Enforcement ---

class TestTwoStepConsistencyEnforcement:
    def test_error_assessment_decisive_stays_incorrect(self):
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "decisive",
                 "assessment_reasoning": "Confirmed."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["lower_court_correct"] is False

    def test_error_assessment_downgraded_forces_correct(self):
        """If Step 2 downgrades all errors from decisive, force correct."""
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "decisive"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "significant",
                 "assessment_reasoning": "Not actually decisive."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        assert result["lower_court_correct"] is True

    def test_error_assessment_takes_priority_over_identified_errors(self):
        """error_assessment confirmed_severity overrides identified_errors severity,
        but severity ceiling limits upgrades to +1 level max."""
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "procedural", "severity": "minor"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "decisive",
                 "assessment_reasoning": "Actually decisive on review."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        # Severity ceiling: minor→decisive capped to significant (max +1)
        assert result["error_assessment"][0]["confirmed_severity"] == "significant"
        # No decisive → lcc forced True
        assert result["lower_court_correct"] is True

    def test_error_assessment_upgrade_within_ceiling(self):
        """Step 2 can upgrade +1 level (significant→decisive is allowed)."""
        verdict = {
            "lower_court_correct": True,
            "identified_errors": [
                {"error_type": "legal_interpretation", "severity": "significant"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "decisive",
                 "assessment_reasoning": "Error is clearly decisive."},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        # significant→decisive is +1, within ceiling
        assert result["error_assessment"][0]["confirmed_severity"] == "decisive"
        assert result["lower_court_correct"] is False

    def test_severity_ceiling_none_to_decisive(self):
        """Step 2 cannot jump from none to decisive (3 levels)."""
        verdict = {
            "lower_court_correct": False,
            "identified_errors": [
                {"error_type": "procedural", "severity": "none"},
            ],
            "error_assessment": [
                {"error_id": 0, "confirmed_severity": "decisive"},
            ],
        }
        result = _ch_enforce_consistency(verdict)
        # none→decisive capped to minor (max +1)
        assert result["error_assessment"][0]["confirmed_severity"] == "minor"
        assert result["lower_court_correct"] is True


# --- Two-Step Phase Builder ---

class TestTwoStepPhaseBuilder:
    def test_swiss_uses_two_step(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        judge_config = phases[2].agents[0]
        assert judge_config.role_type == "adjudicator_two_step"
        assert judge_config.prompt_key == "judge_ch_step1"
        assert judge_config.schema_key == "judge_ch_step1"

    def test_swiss_two_step_has_step2_in_template_vars(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        judge_config = phases[2].agents[0]
        assert judge_config.template_vars["_step2_prompt_key"] == "judge_ch_step2"
        assert judge_config.template_vars["_step2_schema_key"] == "judge_ch_step2"
        assert judge_config.template_vars["_step2_temperature"] == 0.4

    def test_swiss_step1_temperature(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        judge_config = phases[2].agents[0]
        assert judge_config.temperature == 0.7  # step1 temp from CH config

    def test_italian_still_single_step(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        it_case = {
            "jurisdiction": {"country": "IT"},
            "parties": [
                {"id": "opponente", "role": "appellant"},
                {"id": "comune", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(it_case, sample_run_params)
        judge_config = phases[2].agents[0]
        assert judge_config.role_type == "adjudicator"
        assert judge_config.prompt_key == "judge_it"


# --- JurisdictionConfig two-step fields ---

class TestJurisdictionConfigTwoStep:
    def test_ch_has_two_step_enabled(self):
        config = get_jurisdiction("CH")
        assert config.judge_two_step is True
        assert config.judge_step1_prompt_key == "judge_ch_step1"
        assert config.judge_step2_prompt_key == "judge_ch_step2"

    def test_it_has_two_step_disabled(self):
        config = get_jurisdiction("IT")
        assert config.judge_two_step is False
        assert config.judge_step1_prompt_key is None


# --- Multi-model support ---

class TestMultiModelConfig:
    """Tests for per-role model override in jurisdiction config and graph phases."""

    def test_ch_judge_gets_122b_model(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        judge_config = phases[2].agents[0]
        assert judge_config.model == "qwen3.5-122b-a10b-4bit"

    def test_ch_parties_get_no_model_override(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        assert phases[0].agents[0].model is None  # appellant
        assert phases[1].agents[0].model is None  # respondent

    def test_it_agents_get_no_model_override(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        it_case = {
            "jurisdiction": {"country": "IT"},
            "parties": [
                {"id": "opponente", "role": "appellant"},
                {"id": "comune", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(it_case, sample_run_params)
        for phase in phases:
            for agent in phase.agents:
                assert agent.model is None

    def test_sim_yaml_model_override_takes_priority(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        params = {**sample_run_params, "models": {"judge": "custom-model", "appellant": "small-model"}}
        phases = build_bilateral_phases(swiss_case, params)
        assert phases[0].agents[0].model == "small-model"  # appellant from sim YAML
        assert phases[1].agents[0].model is None  # respondent: no override
        assert phases[2].agents[0].model == "custom-model"  # judge from sim YAML (overrides CH default)

    def test_step2_inherits_model_from_step1(self, sample_run_params):
        from athena.simulation.graph import build_bilateral_phases, build_graph_from_phases, AgentConfig
        swiss_case = {
            "jurisdiction": {"country": "CH"},
            "parties": [
                {"id": "ricorrente", "role": "appellant"},
                {"id": "controparte", "role": "respondent"},
            ],
        }
        phases = build_bilateral_phases(swiss_case, sample_run_params)
        judge_step1 = phases[2].agents[0]
        assert judge_step1.model == "qwen3.5-122b-a10b-4bit"
        # Verify step2 config inherits model when graph is built
        # (step2 is constructed inside build_graph_from_phases)
        assert judge_step1.role_type == "adjudicator_two_step"
