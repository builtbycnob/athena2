# tests/test_prompts.py
import json
from athena.agents.prompts import (
    build_appellant_prompt,
    build_respondent_prompt,
    build_judge_prompt,
)
from athena.simulation.context import (
    build_party_context,
    build_adjudicator_context,
    _sanitize_brief,
)


class TestAppellantPrompt:
    def test_includes_advocacy_style(self, sample_case_data, sample_run_params):
        ctx = build_party_context(sample_case_data, sample_run_params, "opponente")
        # Inject advocacy_style as the generic graph node does
        profile = sample_run_params.get("party_profiles", {}).get("opponente", {})
        ctx["advocacy_style"] = profile.get("parameters", {}).get("style", "")
        system, user = build_appellant_prompt(ctx)
        assert "Attacca frontalmente" in system

    def test_includes_legal_texts(self, sample_case_data, sample_run_params):
        ctx = build_party_context(sample_case_data, sample_run_params, "opponente")
        ctx["advocacy_style"] = ""
        system, user = build_appellant_prompt(ctx)
        assert "art_143_cds" in user or "Art. 143" in user

    def test_returns_system_and_user(self, sample_case_data, sample_run_params):
        ctx = build_party_context(sample_case_data, sample_run_params, "opponente")
        ctx["advocacy_style"] = ""
        system, user = build_appellant_prompt(ctx)
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 100
        assert len(user) > 100


class TestRespondentPrompt:
    def test_includes_appellant_arguments(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_party_context(sample_case_data, sample_run_params, "comune_milano")
        ctx["appellant_brief"] = _sanitize_brief(sample_appellant_brief)
        system, user = build_respondent_prompt(ctx)
        assert "ARG1" in user

    def test_no_internal_analysis_in_prompt(
        self, sample_case_data, sample_run_params, sample_appellant_brief
    ):
        ctx = build_party_context(sample_case_data, sample_run_params, "comune_milano")
        ctx["appellant_brief"] = _sanitize_brief(sample_appellant_brief)
        system, user = build_respondent_prompt(ctx)
        assert "key_vulnerabilities" not in user
        assert "strongest_point" not in user


class TestJudgePrompt:
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
        ctx["appellant_brief"] = _sanitize_brief(sample_appellant_brief)
        ctx["respondent_brief"] = _sanitize_brief(sample_respondent_brief)
        system, user = build_judge_prompt(ctx)
        assert "follows_cassazione" in system

    def test_no_advocacy_style_in_prompt(
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
        ctx["appellant_brief"] = _sanitize_brief(sample_appellant_brief)
        ctx["respondent_brief"] = _sanitize_brief(sample_respondent_brief)
        system, user = build_judge_prompt(ctx)
        assert "advocacy_style" not in user
        assert "Attacca frontalmente" not in user
