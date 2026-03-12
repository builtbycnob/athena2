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
from athena.jurisdiction.ch import _ch_outcome_extractor
from athena.schemas.structured_output import AGENT_SCHEMAS, JUDGE_CH_SCHEMA


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


# --- Swiss judge schema ---

class TestJudgeChSchema:
    def test_registered_in_agent_schemas(self):
        assert "judge_ch" in AGENT_SCHEMAS
        assert AGENT_SCHEMAS["judge_ch"] is JUDGE_CH_SCHEMA

    def test_schema_is_valid_json_schema(self):
        validator_cls = jsonschema.validators.validator_for(JUDGE_CH_SCHEMA)
        validator_cls.check_schema(JUDGE_CH_SCHEMA)

    def test_validates_sample_swiss_decision(self):
        decision = {
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
            "verdict": {
                "appeal_outcome": "dismissed",
                "outcome_reasoning": "Il ricorso è infondato nel merito.",
                "remedy": {
                    "type": "confirm",
                    "description": "Decisione confermata.",
                    "amount_awarded": None,
                    "costs_appellant": 2000,
                    "costs_respondent": 0,
                },
                "costs_ruling": "a carico del ricorrente",
            },
            "reasoning": "Motivazione completa della sentenza.",
            "gaps": [],
        }
        jsonschema.validate(decision, JUDGE_CH_SCHEMA)

    def test_validates_upheld_with_remand(self):
        decision = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.8,
                    "strengths": "Forte",
                    "weaknesses": "Nessuna",
                    "determinative": True,
                }
            ],
            "precedent_analysis": {},
            "verdict": {
                "appeal_outcome": "remanded",
                "outcome_reasoning": "La causa va rinviata.",
                "remedy": {
                    "type": "remand",
                    "description": "Rinvio all'istanza inferiore.",
                    "amount_awarded": None,
                    "costs_appellant": 0,
                    "costs_respondent": 1000,
                },
                "costs_ruling": "a carico della controparte",
            },
            "reasoning": "Motivazione.",
            "gaps": [],
        }
        jsonschema.validate(decision, JUDGE_CH_SCHEMA)

    def test_rejects_invalid_appeal_outcome(self):
        decision = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.5,
                    "strengths": "OK",
                    "weaknesses": "Debole",
                    "determinative": False,
                }
            ],
            "precedent_analysis": {},
            "verdict": {
                "appeal_outcome": "invalid_value",
                "outcome_reasoning": "Test.",
                "remedy": {
                    "type": "confirm",
                    "description": "Test.",
                    "amount_awarded": None,
                    "costs_appellant": 0,
                    "costs_respondent": 0,
                },
                "costs_ruling": "test",
            },
            "reasoning": "Test.",
            "gaps": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(decision, JUDGE_CH_SCHEMA)

    def test_rejects_invalid_remedy_type(self):
        decision = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.5,
                    "strengths": "OK",
                    "weaknesses": "OK",
                    "determinative": False,
                }
            ],
            "precedent_analysis": {},
            "verdict": {
                "appeal_outcome": "dismissed",
                "outcome_reasoning": "Test.",
                "remedy": {
                    "type": "invalid_type",
                    "description": "Test.",
                    "amount_awarded": None,
                    "costs_appellant": 0,
                    "costs_respondent": 0,
                },
                "costs_ruling": "test",
            },
            "reasoning": "Test.",
            "gaps": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(decision, JUDGE_CH_SCHEMA)


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
        assert "judge_ch" in prompt_keys

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
        assert judge_config.schema_key == "judge_ch"

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
