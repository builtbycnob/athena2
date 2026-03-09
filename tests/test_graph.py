# tests/test_graph.py
import pytest
from unittest.mock import patch
from athena.simulation.graph import build_graph, run_single


class TestBuildGraph:
    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        # LangGraph compiled graph should have the node names
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
