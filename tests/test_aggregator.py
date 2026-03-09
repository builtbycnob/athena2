# tests/test_aggregator.py
import pytest
from athena.simulation.aggregator import aggregate_results, wilson_ci


class TestWilsonCI:
    def test_all_successes(self):
        low, high = wilson_ci(5, 5)
        assert low > 0.5
        assert high <= 1.0

    def test_no_successes(self):
        low, high = wilson_ci(0, 5)
        assert low >= 0.0
        assert high < 0.5

    def test_zero_trials(self):
        low, high = wilson_ci(0, 0)
        assert low == 0.0 and high == 0.0


class TestAggregateResults:
    def test_basic_aggregation(self):
        results = [
            {
                "run_id": "jp1__style1__000",
                "judge_profile": "formalista_pro_cass",
                "appellant_profile": "aggressivo",
                "judge_decision": {
                    "verdict": {"qualification_correct": False, "if_incorrect": {"consequence": "reclassification"}},
                    "argument_evaluation": [
                        {"argument_id": "ARG1", "persuasiveness": 0.7, "determinative": True, "party": "appellant"},
                    ],
                    "precedent_analysis": {"cass_16515_2005": {"followed": False, "distinguished": True}},
                },
            },
            {
                "run_id": "jp1__style1__001",
                "judge_profile": "formalista_pro_cass",
                "appellant_profile": "aggressivo",
                "judge_decision": {
                    "verdict": {"qualification_correct": True},
                    "argument_evaluation": [
                        {"argument_id": "ARG1", "persuasiveness": 0.4, "determinative": False, "party": "appellant"},
                    ],
                    "precedent_analysis": {"cass_16515_2005": {"followed": True, "distinguished": False}},
                },
            },
        ]

        agg = aggregate_results(results, total_expected=2)
        key = ("formalista_pro_cass", "aggressivo")
        assert key in agg["probability_table"]
        assert agg["probability_table"][key]["n_runs"] == 2
        assert agg["probability_table"][key]["p_rejection"] == 0.5
        assert agg["total_runs"] == 2
        assert agg["failed_runs"] == 0

    def test_argument_effectiveness(self):
        results = [
            {
                "run_id": "test__test__000",
                "judge_profile": "jp1",
                "appellant_profile": "s1",
                "judge_decision": {
                    "verdict": {"qualification_correct": False, "if_incorrect": {"consequence": "annulment"}},
                    "argument_evaluation": [
                        {"argument_id": "ARG1", "persuasiveness": 0.9, "determinative": True, "party": "appellant"},
                        {"argument_id": "ARG2", "persuasiveness": 0.3, "determinative": False, "party": "appellant"},
                    ],
                    "precedent_analysis": {},
                },
            },
        ]
        agg = aggregate_results(results, total_expected=1)
        assert agg["argument_effectiveness"]["ARG1"]["mean_persuasiveness"] == 0.9
        assert agg["argument_effectiveness"]["ARG2"]["mean_persuasiveness"] == 0.3
        assert agg["argument_effectiveness"]["ARG1"]["determinative_rate"] == 1.0
