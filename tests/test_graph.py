# tests/test_graph.py
import pytest
from unittest.mock import patch, call
from athena.simulation.graph import (
    build_graph_from_phases, build_bilateral_phases, run_single, _MAX_TOKENS,
)
from athena.schemas.structured_output import AGENT_SCHEMAS


class TestBuildGraph:
    def test_graph_compiles(self, sample_case_data, sample_run_params):
        phases = build_bilateral_phases(sample_case_data, sample_run_params)
        graph = build_graph_from_phases(phases)
        assert graph is not None


class TestRunSingle:
    @patch("athena.simulation.graph.invoke_llm")
    def test_end_to_end_with_mock(
        self,
        mock_llm,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        """Full run with mocked LLM returning valid outputs."""
        mock_llm.side_effect = [
            sample_appellant_brief,   # appellant
            sample_respondent_brief,  # respondent
            {                         # judge
                "preliminary_objections_ruling": [],
                "case_reaches_merits": True,
                "argument_evaluation": [
                    {
                        "argument_id": "ARG1",
                        "party": "appellant",
                        "persuasiveness": 0.7,
                        "strengths": "Forte",
                        "weaknesses": "Cassazione contraria",
                        "determinative": True,
                    },
                    {
                        "argument_id": "RARG1",
                        "party": "respondent",
                        "persuasiveness": 0.5,
                        "strengths": "Precedente",
                        "weaknesses": "Testo contrario",
                        "determinative": False,
                    },
                ],
                "precedent_analysis": {
                    "cass_16515_2005": {
                        "followed": False,
                        "distinguished": True,
                        "reasoning": "Caso distinguibile.",
                    }
                },
                "verdict": {
                    "qualification_correct": False,
                    "qualification_reasoning": "La qualificazione è errata.",
                    "if_incorrect": {
                        "consequence": "reclassification",
                        "consequence_reasoning": "Va riqualificata.",
                        "applied_norm": "artt. 6-7 CdS",
                        "sanction_determined": 87,
                        "points_deducted": 0,
                    },
                    "costs_ruling": "a carico del Comune",
                },
                "reasoning": "Motivazione della sentenza.",
                "gaps": [],
            },
        ]

        result = run_single(sample_case_data, sample_run_params)

        assert result["error"] is None
        assert result["appellant_brief"] is not None
        assert result["respondent_brief"] is not None
        assert result["judge_decision"] is not None
        assert result["judge_decision"]["verdict"]["qualification_correct"] is False

    @patch("athena.simulation.graph.invoke_llm")
    def test_passes_json_schema_and_max_tokens(
        self,
        mock_llm,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        """Verify invoke_llm receives json_schema and max_tokens kwargs."""
        judge_decision = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.7,
                    "strengths": "Forte",
                    "weaknesses": "Cassazione contraria",
                    "determinative": True,
                },
                {
                    "argument_id": "RARG1",
                    "party": "respondent",
                    "persuasiveness": 0.5,
                    "strengths": "Precedente",
                    "weaknesses": "Testo contrario",
                    "determinative": False,
                },
            ],
            "precedent_analysis": {
                "cass_16515_2005": {
                    "followed": False,
                    "distinguished": True,
                    "reasoning": "Caso distinguibile.",
                }
            },
            "verdict": {
                "qualification_correct": False,
                "qualification_reasoning": "La qualificazione è errata.",
                "if_incorrect": {
                    "consequence": "reclassification",
                    "consequence_reasoning": "Va riqualificata.",
                    "applied_norm": "artt. 6-7 CdS",
                    "sanction_determined": 87,
                    "points_deducted": 0,
                },
                "costs_ruling": "a carico del Comune",
            },
            "reasoning": "Motivazione della sentenza.",
            "gaps": [],
        }
        mock_llm.side_effect = [
            sample_appellant_brief,
            sample_respondent_brief,
            judge_decision,
        ]

        run_single(sample_case_data, sample_run_params)

        # Check each call got the right schema and max_tokens
        # Schemas are now dynamic deep copies (with enum constraints), not identity
        assert mock_llm.call_count == 3
        for i, agent in enumerate(["appellant", "respondent", "judge"]):
            _, kwargs = mock_llm.call_args_list[i]
            assert kwargs["json_schema"]["type"] == AGENT_SCHEMAS[agent]["type"]
            assert kwargs["json_schema"]["required"] == AGENT_SCHEMAS[agent]["required"]
            assert kwargs["max_tokens"] == _MAX_TOKENS[agent]
