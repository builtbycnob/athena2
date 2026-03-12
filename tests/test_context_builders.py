# tests/test_context_builders.py
import pytest
from athena.simulation.context import (
    build_party_context,
    build_adjudicator_context,
    _sanitize_brief,
)


class TestBuildPartyContextAppellant:
    def test_includes_own_seed_arguments(self, sample_case_data, sample_run_params):
        ctx = build_party_context(sample_case_data, sample_run_params, "opponente")
        assert "seed_arguments" in ctx
        assert ctx["seed_arguments"][0]["id"] == "SEED_ARG1"

    def test_excludes_respondent_seeds(self, sample_case_data, sample_run_params):
        ctx = build_party_context(sample_case_data, sample_run_params, "opponente")
        seed_ids = {s["id"] for s in ctx["seed_arguments"]}
        assert "SEED_RARG1" not in seed_ids

    def test_excludes_judge_profile(self, sample_case_data, sample_run_params):
        ctx = build_party_context(sample_case_data, sample_run_params, "opponente")
        assert "judge_profile" not in ctx


class TestBuildPartyContextRespondent:
    def test_includes_prior_briefs(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        prior = {"opponente": _sanitize_brief(sample_appellant_brief)}
        ctx = build_party_context(
            sample_case_data, sample_run_params, "comune_milano",
            prior_briefs=prior,
        )
        assert "prior_briefs" in ctx
        assert "opponente" in ctx["prior_briefs"]
        # Sanitized: no internal_analysis
        assert "internal_analysis" not in ctx["prior_briefs"]["opponente"]

    def test_excludes_judge_profile(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_party_context(sample_case_data, sample_run_params, "comune_milano")
        assert "judge_profile" not in ctx


class TestBuildAdjudicatorContext:
    def test_includes_all_briefs_sanitized(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        briefs = {
            "opponente": sample_appellant_brief,
            "comune_milano": sample_respondent_brief,
        }
        ctx = build_adjudicator_context(sample_case_data, sample_run_params, briefs)
        assert "all_briefs" in ctx
        assert len(ctx["all_briefs"]) == 2
        # Sanitized: no internal_analysis
        for pid, brief in ctx["all_briefs"].items():
            assert "internal_analysis" not in brief
            assert "arguments" in brief or "responses_to_opponent" in brief

    def test_includes_judge_profile(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        briefs = {
            "opponente": sample_appellant_brief,
            "comune_milano": sample_respondent_brief,
        }
        ctx = build_adjudicator_context(sample_case_data, sample_run_params, briefs)
        assert "judge_profile" in ctx

    def test_excludes_seed_arguments(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        briefs = {
            "opponente": sample_appellant_brief,
            "comune_milano": sample_respondent_brief,
        }
        ctx = build_adjudicator_context(sample_case_data, sample_run_params, briefs)
        assert "seed_arguments" not in ctx
