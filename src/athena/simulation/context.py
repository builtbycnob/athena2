# src/athena/simulation/context.py


def _sanitize_brief(brief: dict) -> dict:
    """Strip internal_analysis, return only filed_brief contents."""
    return brief["filed_brief"]


def _find_party(case_data: dict, party_id: str) -> dict:
    """Find a party by ID in the case data."""
    for p in case_data["parties"]:
        if p["id"] == party_id:
            return p
    raise ValueError(f"Party '{party_id}' not found in case data")


def build_party_context(
    case_data: dict,
    run_params: dict,
    party_id: str,
    prior_briefs: dict[str, dict] | None = None,
    visibility: dict | None = None,
) -> dict:
    """Generic context for any party agent.

    Args:
        case_data: Full case data dict.
        run_params: Run parameters with profiles and temperatures.
        party_id: ID of the party this context is for.
        prior_briefs: party_id → sanitized brief from earlier phases.
        visibility: Optional visibility rules from Party.visibility.
    """
    party = _find_party(case_data, party_id)
    if prior_briefs is None:
        prior_briefs = {}

    # Evidence filtering based on visibility rules
    vis = visibility or (party.get("visibility") if isinstance(party, dict) else None) or {}
    evidence_vis = vis.get("evidence_visibility", "own_and_uncontested")
    if evidence_vis == "all":
        evidence = case_data["evidence"]
    else:  # "own_and_uncontested"
        evidence = [
            e for e in case_data["evidence"]
            if e["produced_by"] == party_id or e["admissibility"] == "uncontested"
        ]

    # Seed arguments for this party
    seed_args = case_data["seed_arguments"]["by_party"].get(party_id, [])

    # Party profile from run_params
    profile = run_params.get("party_profiles", {}).get(party_id, {})

    return {
        "own_party": party,
        "facts": case_data["facts"],
        "evidence": evidence,
        "legal_texts": case_data["legal_texts"],
        "precedents": case_data["key_precedents"],
        "seed_arguments": seed_args,
        "stakes": case_data["stakes"],
        "procedural_rules": case_data["jurisdiction"]["procedural_rules"],
        "prior_briefs": prior_briefs,
        "profile": profile,
    }


def build_adjudicator_context(
    case_data: dict,
    run_params: dict,
    all_briefs: dict[str, dict],
) -> dict:
    """Context for judge/arbitrator — sees ALL evidence and ALL briefs."""
    sanitized = {pid: _sanitize_brief(b) for pid, b in all_briefs.items()}
    return {
        "facts": case_data["facts"],
        "evidence": case_data["evidence"],
        "legal_texts": case_data["legal_texts"],
        "precedents": case_data["key_precedents"],
        "stakes": case_data["stakes"],
        "procedural_rules": case_data["jurisdiction"]["procedural_rules"],
        "all_briefs": sanitized,
        "judge_profile": run_params["judge_profile"],
    }


# --- Backward-compatible wrappers (used during transition) ---

def _sanitize_brief_for_opponent(brief: dict) -> dict:
    return _sanitize_brief(brief)


def _sanitize_brief_for_judge(brief: dict) -> dict:
    return _sanitize_brief(brief)


def build_context_appellant(case_data: dict, run_params: dict) -> dict:
    own_id = next(p["id"] for p in case_data["parties"] if p["role"] == "appellant")
    ctx = build_party_context(case_data, run_params, own_id)
    # Legacy format: advocacy_style at top level
    profile = run_params.get("party_profiles", {}).get(own_id, {})
    style = profile.get("parameters", {}).get("style", "")
    if not style:
        # Fallback for old run_params format
        style = run_params.get("appellant_profile", {}).get("style", "")
    ctx["advocacy_style"] = style
    return ctx


def build_context_respondent(
    case_data: dict, run_params: dict, appellant_brief: dict
) -> dict:
    own_id = next(p["id"] for p in case_data["parties"] if p["role"] == "respondent")
    sanitized = _sanitize_brief(appellant_brief)
    ctx = build_party_context(
        case_data, run_params, own_id,
        prior_briefs={},
        visibility={"evidence_visibility": "all"},
    )
    ctx["appellant_brief"] = sanitized
    return ctx


def build_context_judge(
    case_data: dict,
    run_params: dict,
    appellant_brief: dict,
    respondent_brief: dict,
) -> dict:
    appellant_id = next(p["id"] for p in case_data["parties"] if p["role"] == "appellant")
    respondent_id = next(p["id"] for p in case_data["parties"] if p["role"] == "respondent")
    ctx = build_adjudicator_context(
        case_data, run_params,
        {appellant_id: appellant_brief, respondent_id: respondent_brief},
    )
    # Legacy format: separate brief keys
    ctx["appellant_brief"] = _sanitize_brief(appellant_brief)
    ctx["respondent_brief"] = _sanitize_brief(respondent_brief)
    return ctx
