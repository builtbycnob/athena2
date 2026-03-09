# tests/test_context_builders.py
import pytest
from athena.simulation.context import (
    build_context_appellant,
    build_context_respondent,
    build_context_judge,
)


class TestBuildContextAppellant:
    def test_includes_own_seed_arguments(self, sample_case_data, sample_run_params):
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        assert "seed_arguments" in ctx
        assert ctx["seed_arguments"][0]["id"] == "SEED_ARG1"

    def test_excludes_respondent_seeds(self, sample_case_data, sample_run_params):
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        seed_ids = {s["id"] for s in ctx["seed_arguments"]}
        assert "SEED_RARG1" not in seed_ids

    def test_includes_advocacy_style(self, sample_case_data, sample_run_params):
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        assert "advocacy_style" in ctx

    def test_excludes_judge_profile(self, sample_case_data, sample_run_params):
        ctx = build_context_appellant(sample_case_data, sample_run_params)
        assert "judge_profile" not in ctx


class TestBuildContextRespondent:
    def test_includes_only_filed_brief(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_context_respondent(
            sample_case_data, sample_run_params, sample_appellant_brief
        )
        # Should have filed_brief fields but NOT internal_analysis
        assert "arguments" in ctx["appellant_brief"]
        assert "requests" in ctx["appellant_brief"]
        assert "internal_analysis" not in ctx.get("appellant_brief", {})
        assert "key_vulnerabilities" not in ctx.get("appellant_brief", {})

    def test_excludes_judge_profile(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_context_respondent(
            sample_case_data, sample_run_params, sample_appellant_brief
        )
        assert "judge_profile" not in ctx

    def test_excludes_advocacy_style(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_context_respondent(
            sample_case_data, sample_run_params, sample_appellant_brief
        )
        assert "advocacy_style" not in ctx


class TestBuildContextJudge:
    def test_includes_both_filed_briefs(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        assert "appellant_brief" in ctx
        assert "respondent_brief" in ctx
        # Only filed_brief content
        assert "arguments" in ctx["appellant_brief"]
        assert "internal_analysis" not in ctx["appellant_brief"]

    def test_includes_judge_profile(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        assert "judge_profile" in ctx

    def test_excludes_advocacy_style(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        assert "advocacy_style" not in ctx

    def test_excludes_seed_arguments(
        self,
        sample_case_data,
        sample_run_params,
        sample_appellant_brief,
        sample_respondent_brief,
    ):
        ctx = build_context_judge(
            sample_case_data,
            sample_run_params,
            sample_appellant_brief,
            sample_respondent_brief,
        )
        assert "seed_arguments" not in ctx
