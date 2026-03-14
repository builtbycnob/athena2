# src/athena/jurisdiction/ch.py
"""Swiss jurisdiction — Tribunale federale (Bundesgericht), ricorso in materia civile."""

from athena.jurisdiction.registry import JurisdictionConfig, register_jurisdiction


def _ch_enforce_consistency(verdict: dict) -> dict:
    """Enforce cross-field consistency that constrained decoding cannot.

    Works with both single-step schema (identified_errors in verdict)
    and two-step merged output (error_assessment from Step 2).

    Rules:
    1. If no decisive errors → force lower_court_correct=True
       - Two-step: uses confirmed_severity from error_assessment
       - Single-step: uses severity from identified_errors
    2. If lower_court_correct=True but if_correct is None → populate stub
    3. If lower_court_correct=False but if_incorrect is None → populate stub
    """
    # Severity calibration: floor + ceiling.
    # Floor: Step 1 decisive can't be downgraded below significant.
    # Ceiling: Step 2 can upgrade at most +1 level (prevents false decisive).
    _SEV_ORDER = {"none": 0, "minor": 1, "significant": 2, "decisive": 3}
    _SEV_NAMES = {v: k for k, v in _SEV_ORDER.items()}
    error_assessment = verdict.get("error_assessment", [])
    errors = verdict.get("identified_errors", [])
    if error_assessment and errors:
        errors_by_idx = {i: err for i, err in enumerate(errors)}
        for ea in error_assessment:
            eid = ea.get("error_id")
            if eid is not None and eid in errors_by_idx:
                s1 = errors_by_idx[eid].get("severity", "none")
                s2 = ea.get("confirmed_severity", "none")
                s1_level = _SEV_ORDER.get(s1, 0)
                s2_level = _SEV_ORDER.get(s2, 0)
                # Floor: decisive can't drop below significant
                if s1 == "decisive" and s2 in ("minor", "none"):
                    ea["confirmed_severity"] = "significant"
                # Ceiling: max +1 level upgrade
                elif s2_level > s1_level + 1:
                    ea["confirmed_severity"] = _SEV_NAMES[s1_level + 1]

    # Two-step: use error_assessment confirmed severities if available
    if error_assessment:
        severities = [ea.get("confirmed_severity", "none") for ea in error_assessment]
    else:
        severities = [e.get("severity", "none") for e in errors]

    has_decisive = "decisive" in severities

    lcc = verdict.get("lower_court_correct")

    # Rule 1a: decisive errors present → lower_court_correct must be False
    if has_decisive and lcc is True:
        verdict["lower_court_correct"] = False
        lcc = False

    # Rule 1b: no decisive errors → lower_court_correct should be True
    if not has_decisive and lcc is False:
        verdict["lower_court_correct"] = True
        lcc = True

    # Rule 2: branch completion
    if lcc is True:
        verdict["if_incorrect"] = None
        if not verdict.get("if_correct"):
            verdict["if_correct"] = {
                "confirmation_reasoning": verdict.get("correctness_reasoning", "Decisione confermata.")
            }
    elif lcc is False:
        verdict["if_correct"] = None
        if not verdict.get("if_incorrect"):
            verdict["if_incorrect"] = {
                "consequence": "annulment",
                "consequence_reasoning": verdict.get("correctness_reasoning", "Errore decisivo individuato."),
                "remedy": {
                    "type": "annul",
                    "description": "Annullamento della decisione impugnata.",
                    "amount_awarded": None,
                    "costs_appellant": 0,
                    "costs_respondent": 0,
                },
            }

    return verdict


def _ch_outcome_extractor(verdict: dict) -> str:
    """Extract outcome from Swiss judge verdict.

    Applies consistency enforcement before extracting outcome.
    Supports both new two-step schema (lower_court_correct) and
    legacy flat schema (appeal_outcome).
    """
    # Apply consistency enforcement for new schema
    if "lower_court_correct" in verdict:
        verdict = _ch_enforce_consistency(verdict)
        if verdict.get("lower_court_correct"):
            return "rejection"
        return "annulment"
    # Legacy fallback for old appeal_outcome results
    outcome = verdict.get("appeal_outcome", "dismissed")
    if outcome == "dismissed":
        return "rejection"
    return "annulment"


_CH_CONFIG = JurisdictionConfig(
    country="CH",
    prompt_keys={
        "appellant": "appellant_ch",
        "respondent": "respondent_ch",
        "judge": "judge_ch",
    },
    schema_keys={
        "appellant": "appellant",       # same schema, different prompt
        "respondent": "respondent",     # same schema, different prompt
        "judge": "judge_ch",            # different schema
    },
    verdict_schema_key="judge_ch",
    outcome_extractor=_ch_outcome_extractor,
    outcome_space=["rejection", "annulment"],
    source_hierarchy=(
        "Costituzione federale > Leggi federali (CO, CC, LEF, LTF) > "
        "Ordinanze del Consiglio federale > Diritto cantonale > "
        "Giurisprudenza del Tribunale federale (DTF/BGE)"
    ),
    respondent_brief_label="Memoria della controparte (depositata)",
    default_temperatures={"judge": 0.7},
    # Multi-model: 122B MoE for judge (higher accuracy on complex reasoning)
    default_models={"judge": "qwen3.5-122b-a10b-4bit"},
    # Two-step judge architecture (v1.1 bias fix)
    judge_two_step=True,
    judge_step1_prompt_key="judge_ch_step1",
    judge_step1_schema_key="judge_ch_step1",
    judge_step1_temperature=0.7,
    judge_step2_prompt_key="judge_ch_step2",
    judge_step2_schema_key="judge_ch_step2",
    judge_step2_temperature=0.4,
)

register_jurisdiction("CH", _CH_CONFIG)
