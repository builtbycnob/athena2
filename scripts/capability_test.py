#!/usr/bin/env python3
"""
Run: python scripts/capability_test.py
      python scripts/capability_test.py --dry-run

Tests the MLX model's ability to:
1. Produce valid JSON
2. Follow the agent prompt structure
3. Differentiate between judge profiles
4. Maintain referential integrity
"""
import argparse
import json
import sys
import time

# ---------------------------------------------------------------------------
# Test fixtures (minimal versions from tests/conftest.py)
# ---------------------------------------------------------------------------

SAMPLE_CASE_DATA = {
    "case_id": "gdp-milano-17928-2025",
    "jurisdiction": {
        "country": "IT",
        "court": "giudice_di_pace",
        "venue": "Milano",
        "applicable_law": ["D.Lgs. 285/1992", "L. 689/1981"],
        "key_precedents": [
            {
                "id": "cass_16515_2005",
                "citation": "Cass. civ. n. 16515/2005",
                "holding": "Equiparazione contromano/controsenso",
                "weight": "contested",
            }
        ],
        "procedural_rules": {
            "rite": "opposizione_sanzione_amministrativa",
            "phases": ["ricorso", "costituzione_resistente", "udienza", "decisione"],
            "allowed_moves": {
                "appellant": ["memoria", "produzione_documenti"],
                "respondent": ["memoria_costituzione", "produzione_documenti"],
            },
        },
    },
    "parties": [
        {
            "id": "opponente",
            "role": "appellant",
            "type": "persona_fisica",
            "objectives": {
                "primary": "annullamento_verbale",
                "subordinate": "riqualificazione_artt_6_7",
            },
        },
        {
            "id": "comune_milano",
            "role": "respondent",
            "type": "pubblica_amministrazione",
            "entity": "Comune di Milano -- Polizia Locale",
            "objectives": {
                "primary": "conferma_verbale",
                "subordinate": "conferma_anche_con_riduzione",
            },
        },
    ],
    "stakes": {
        "current_sanction": {
            "norm": "art. 143 CdS",
            "fine_range": [170, 680],
            "points_deducted": 4,
        },
        "alternative_sanction": {
            "norm": "artt. 6-7 CdS",
            "fine_range": [42, 173],
            "points_deducted": 0,
        },
        "litigation_cost_estimate": 1500,
    },
    "evidence": [
        {
            "id": "DOC1",
            "type": "atto_pubblico",
            "description": "Verbale Polizia Locale",
            "produced_by": "comune_milano",
            "admissibility": "uncontested",
            "supports_facts": ["F1", "F2", "F3"],
        },
        {
            "id": "DOC2",
            "type": "prova_documentale",
            "description": "Documentazione segnaletica",
            "produced_by": "opponente",
            "admissibility": "uncontested",
            "supports_facts": ["F3"],
        },
    ],
    "facts": {
        "undisputed": [
            {"id": "F1", "description": "Transito in senso vietato", "evidence": ["DOC1"]},
            {"id": "F2", "description": "Verbale ex art. 143 CdS", "evidence": ["DOC1"]},
            {"id": "F3", "description": "Strada a senso unico", "evidence": ["DOC1", "DOC2"]},
        ],
        "disputed": [
            {
                "id": "D1",
                "description": "Correttezza qualificazione giuridica",
                "appellant_position": "Art. 143 inapplicabile",
                "respondent_position": "Art. 143 applicabile per Cass. 16515/2005",
                "depends_on_facts": ["F1", "F3"],
            }
        ],
    },
    "legal_texts": [
        {
            "id": "art_143_cds",
            "norm": "Art. 143 D.Lgs. 285/1992",
            "text": (
                "I veicoli devono circolare sulla parte destra della carreggiata "
                "e in prossimita del margine destro della medesima, anche quando "
                "la strada e libera. [testo di esempio per test]"
            ),
        },
        {
            "id": "art_6_cds",
            "norm": "Art. 6 D.Lgs. 285/1992",
            "text": (
                "Il prefetto puo, per motivi di sicurezza pubblica o inerenti "
                "alla sicurezza della circolazione... [testo di esempio per test]"
            ),
        },
        {
            "id": "art_1_l689",
            "norm": "Art. 1 L. 689/1981",
            "text": (
                "Nessuno puo essere assoggettato a sanzioni amministrative se non "
                "in forza di una legge che sia entrata in vigore prima della "
                "commissione della violazione. Le leggi che prevedono sanzioni "
                "amministrative si applicano soltanto nei casi e per i tempi "
                "in esse considerati."
            ),
        },
    ],
    "seed_arguments": {
        "appellant": [
            {
                "id": "SEED_ARG1",
                "claim": "Errata qualificazione giuridica",
                "direction": "Art. 143 non copre la fattispecie",
                "references_facts": ["F1", "F3", "D1"],
            },
            {
                "id": "SEED_ARG2",
                "claim": "Contraddizione interna del verbale",
                "direction": "Verbale descrive senso unico, applica norma da doppio senso",
                "references_facts": ["F3"],
            },
        ],
        "respondent": [
            {
                "id": "SEED_RARG1",
                "claim": "Legittimita ex Cass. 16515/2005",
                "direction": "Cassazione equipara le due condotte",
                "references_facts": ["F1", "D1"],
            },
        ],
    },
    "key_precedents": [
        {
            "id": "cass_16515_2005",
            "citation": "Cass. civ. n. 16515/2005",
            "holding": "Equiparazione contromano/controsenso",
            "weight": "contested",
        }
    ],
    "timeline": [],
}

SAMPLE_RUN_PARAMS = {
    "run_id": "test__aggressivo__000",
    "judge_profile": {
        "id": "formalista_pro_cass",
        "jurisprudential_orientation": "follows_cassazione",
        "formalism": "high",
    },
    "appellant_profile": {
        "id": "aggressivo",
        "style": "Attacca frontalmente la giurisprudenza sfavorevole.",
    },
    "temperature": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
    "language": "it",
}

SAMPLE_APPELLANT_BRIEF = {
    "filed_brief": {
        "arguments": [
            {
                "id": "ARG1",
                "type": "derived",
                "derived_from": "SEED_ARG1",
                "claim": "Errata qualificazione giuridica del fatto",
                "legal_reasoning": (
                    "L'art. 143 disciplina la marcia contromano su strada a "
                    "doppio senso. Il fatto e avvenuto su senso unico."
                ),
                "norm_text_cited": ["art_143_cds"],
                "facts_referenced": ["F1", "F3"],
                "evidence_cited": ["DOC1"],
                "precedents_addressed": [
                    {
                        "id": "cass_16515_2005",
                        "strategy": "distinguish",
                        "reasoning": "Il precedente non e in punto.",
                    }
                ],
                "supports": None,
            },
        ],
        "requests": {
            "primary": "Annullamento del verbale",
            "subordinate": "Riqualificazione sotto artt. 6-7 CdS",
        },
    },
    "internal_analysis": {
        "strength_self_assessments": {"ARG1": 0.7},
        "key_vulnerabilities": ["Cassazione 16515/2005 contraria"],
        "strongest_point": "Testo letterale art. 143 non copre senso unico",
        "gaps": [],
    },
}

SAMPLE_RESPONDENT_BRIEF = {
    "filed_brief": {
        "preliminary_objections": [],
        "responses_to_opponent": [
            {
                "to_argument": "ARG1",
                "counter_strategy": "rebut",
                "counter_reasoning": (
                    "La Cassazione ha equiparato le due fattispecie."
                ),
                "norm_text_cited": ["art_143_cds"],
                "precedents_cited": [
                    {"id": "cass_16515_2005", "relevance": "Direttamente in punto."}
                ],
            }
        ],
        "affirmative_defenses": [
            {
                "id": "RARG1",
                "type": "derived",
                "derived_from": "SEED_RARG1",
                "claim": "Legittimita del verbale ex Cass. 16515/2005",
                "legal_reasoning": (
                    "La Cassazione equipara contromano e controsenso."
                ),
                "norm_text_cited": ["art_143_cds"],
                "facts_referenced": ["F1"],
                "evidence_cited": ["DOC1"],
            }
        ],
        "requests": {
            "primary": "Rigetto dell'opposizione",
            "fallback": "Conferma sanzione anche in caso di riqualificazione",
        },
    },
    "internal_analysis": {
        "strength_self_assessments": {"response_to_ARG1": 0.6, "RARG1": 0.6},
        "key_vulnerabilities": ["Testo letterale art. 143 non chiarissimo"],
        "opponent_strongest_point": "Argomento testuale sull'art. 143",
        "gaps": [],
    },
}

# Alternative judge profile for differentiation test
ALT_JUDGE_PROFILE = {
    "id": "sostanzialista_critico",
    "jurisprudential_orientation": "distinguishes_cassazione",
    "formalism": "low",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 72
SUBSEP = "-" * 72


def banner(text: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)


def section(text: str) -> None:
    print(f"\n{SUBSEP}")
    print(f"  {text}")
    print(SUBSEP)


def print_json_pretty(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def reasoning_quality_indicators(output: dict, role: str) -> dict[str, object]:
    """Compute simple reasoning quality indicators from agent output."""
    indicators: dict[str, object] = {}

    if role in ("appellant", "respondent"):
        brief = output.get("filed_brief", {})
        args = brief.get("arguments", brief.get("affirmative_defenses", []))
        indicators["num_arguments"] = len(args)
        # Check if legal_reasoning fields are substantive (> 50 chars)
        reasoning_lengths = [
            len(a.get("legal_reasoning", "")) for a in args
        ]
        indicators["reasoning_lengths"] = reasoning_lengths
        indicators["avg_reasoning_length"] = (
            sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
        )
        # Self-assessment spread
        assessments = output.get("internal_analysis", {}).get(
            "strength_self_assessments", {}
        )
        if assessments:
            vals = list(assessments.values())
            indicators["self_assessment_range"] = [min(vals), max(vals)]
            indicators["all_above_0.8"] = all(v > 0.8 for v in vals)
        indicators["has_vulnerabilities"] = bool(
            output.get("internal_analysis", {}).get("key_vulnerabilities")
        )

    elif role == "judge":
        evals = output.get("argument_evaluation", [])
        indicators["num_evaluations"] = len(evals)
        persuasiveness = [e.get("persuasiveness", 0) for e in evals]
        indicators["persuasiveness_scores"] = persuasiveness
        reasoning = output.get("reasoning", "")
        indicators["reasoning_word_count"] = len(reasoning.split())
        verdict = output.get("verdict", {})
        indicators["has_qualification_reasoning"] = bool(
            verdict.get("qualification_reasoning")
        )
        indicators["has_consequence"] = verdict.get("if_incorrect") is not None

    return indicators


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_agent_test(
    agent_name: str,
    role: str,
    build_context_fn,
    build_prompt_fn,
    temperature: float,
    case,
    dry_run: bool,
    appellant_brief: dict | None = None,
    respondent_brief: dict | None = None,
) -> dict | None:
    """Run a single agent test. Returns parsed output or None on failure."""
    from athena.schemas.case import CaseFile
    from athena.simulation.validation import validate_agent_output

    banner(f"Testing: {agent_name} (role={role})")

    # Build context
    context = build_context_fn()
    system_prompt, user_prompt = build_prompt_fn(context)

    section("System prompt (first 500 chars)")
    print(system_prompt[:500])
    print("..." if len(system_prompt) > 500 else "")

    section("User prompt (first 1000 chars)")
    print(user_prompt[:1000])
    print("..." if len(user_prompt) > 1000 else "")

    if dry_run:
        print("\n  [DRY RUN] Skipping model invocation.\n")
        return None

    # Invoke model
    from athena.agents.llm import parse_json_response

    section("Invoking model...")
    t0 = time.time()
    from athena.agents.llm import _call_model
    raw = _call_model(system_prompt, user_prompt, temperature)
    elapsed = time.time() - t0

    section(f"Raw response ({len(raw)} chars, {elapsed:.1f}s)")
    print(raw[:3000])
    if len(raw) > 3000:
        print(f"\n... ({len(raw) - 3000} more chars)")

    # JSON parse
    section("JSON parse")
    try:
        parsed = parse_json_response(raw)
        print("  JSON parse: SUCCESS")
        print_json_pretty(parsed)
    except ValueError as e:
        print(f"  JSON parse: FAILED -- {e}")
        return None

    # Validation
    section("Validation")
    result = validate_agent_output(
        parsed, role, case,
        appellant_brief=appellant_brief,
        respondent_brief=respondent_brief,
    )
    print(f"  Valid:    {result.valid}")
    print(f"  Errors:   {result.errors}")
    print(f"  Warnings: {result.warnings}")

    # Quality indicators
    section("Reasoning quality indicators")
    indicators = reasoning_quality_indicators(parsed, role)
    for k, v in indicators.items():
        print(f"  {k}: {v}")

    return parsed


def run_judge_profile_differentiation(
    case,
    appellant_brief: dict,
    respondent_brief: dict,
    dry_run: bool,
) -> None:
    """Run judge with two profiles and compare outputs."""
    from athena.simulation.context import build_context_judge
    from athena.agents.prompts import build_judge_prompt
    from athena.agents.llm import _call_model, parse_json_response

    banner("Profile Differentiation Test: Judge")

    profiles = [
        ("formalista_pro_cass", SAMPLE_RUN_PARAMS["judge_profile"]),
        ("sostanzialista_critico", ALT_JUDGE_PROFILE),
    ]

    results = {}
    for profile_name, profile in profiles:
        section(f"Judge profile: {profile_name}")
        print(json.dumps(profile, indent=2))

        params = {**SAMPLE_RUN_PARAMS, "judge_profile": profile}
        ctx = build_context_judge(
            SAMPLE_CASE_DATA, params, appellant_brief, respondent_brief
        )
        system_prompt, user_prompt = build_judge_prompt(ctx)

        if dry_run:
            print(f"\n  [DRY RUN] Would invoke model with profile: {profile_name}")
            section("System prompt diff snippet")
            # Show the profile-dependent portion
            for line in system_prompt.split("\n"):
                if "follows_cassazione" in line or "distinguishes_cassazione" in line:
                    print(f"    {line}")
                if "high" in line or "low" in line:
                    print(f"    {line}")
            continue

        t0 = time.time()
        raw = _call_model(system_prompt, user_prompt, 0.3)
        elapsed = time.time() - t0

        try:
            parsed = parse_json_response(raw)
            results[profile_name] = parsed
            print(f"  Response: {len(raw)} chars, {elapsed:.1f}s, JSON: OK")
        except ValueError:
            print(f"  Response: {len(raw)} chars, {elapsed:.1f}s, JSON: FAILED")
            results[profile_name] = None

    if dry_run:
        print("\n  [DRY RUN] Skipping comparison.\n")
        return

    # Compare
    section("Profile Comparison")
    if all(v is not None for v in results.values()):
        for profile_name, parsed in results.items():
            verdict = parsed.get("verdict", {})
            prec = parsed.get("precedent_analysis", {})
            print(f"\n  Profile: {profile_name}")
            print(f"    qualification_correct: {verdict.get('qualification_correct')}")
            print(f"    consequence: {verdict.get('if_incorrect', {}).get('consequence') if verdict.get('if_incorrect') else 'N/A'}")
            for prec_id, prec_data in prec.items():
                print(f"    precedent {prec_id}: followed={prec_data.get('followed')}, distinguished={prec_data.get('distinguished')}")
            indicators = reasoning_quality_indicators(parsed, "judge")
            print(f"    reasoning_word_count: {indicators.get('reasoning_word_count')}")
            print(f"    persuasiveness_scores: {indicators.get('persuasiveness_scores')}")
    else:
        print("  Cannot compare -- one or both parses failed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capability test for ATHENA MLX model"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts only, skip model loading and invocation.",
    )
    args = parser.parse_args()

    from athena.simulation.context import (
        build_context_appellant,
        build_context_respondent,
        build_context_judge,
    )
    from athena.agents.prompts import (
        build_appellant_prompt,
        build_respondent_prompt,
        build_judge_prompt,
    )
    from athena.schemas.case import CaseFile

    case = CaseFile(**SAMPLE_CASE_DATA)

    # Track aggregate metrics
    json_results: dict[str, bool] = {}
    validation_results: dict[str, bool] = {}

    banner("ATHENA Capability Test")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Case:    {SAMPLE_CASE_DATA['case_id']}")

    # --- 1. Appellant ---
    appellant_output = run_agent_test(
        agent_name="Appellant",
        role="appellant",
        build_context_fn=lambda: build_context_appellant(
            SAMPLE_CASE_DATA, SAMPLE_RUN_PARAMS
        ),
        build_prompt_fn=build_appellant_prompt,
        temperature=SAMPLE_RUN_PARAMS["temperature"]["appellant"],
        case=case,
        dry_run=args.dry_run,
    )
    # Use real output if available, fall back to fixture
    appellant_brief = (
        appellant_output if appellant_output is not None else SAMPLE_APPELLANT_BRIEF
    )
    json_results["appellant"] = appellant_output is not None
    if appellant_output is not None:
        from athena.simulation.validation import validate_agent_output
        vr = validate_agent_output(appellant_output, "appellant", case)
        validation_results["appellant"] = vr.valid

    # --- 2. Respondent ---
    respondent_output = run_agent_test(
        agent_name="Respondent",
        role="respondent",
        build_context_fn=lambda: build_context_respondent(
            SAMPLE_CASE_DATA, SAMPLE_RUN_PARAMS, appellant_brief
        ),
        build_prompt_fn=build_respondent_prompt,
        temperature=SAMPLE_RUN_PARAMS["temperature"]["respondent"],
        case=case,
        dry_run=args.dry_run,
        appellant_brief=appellant_brief,
    )
    respondent_brief = (
        respondent_output if respondent_output is not None else SAMPLE_RESPONDENT_BRIEF
    )
    json_results["respondent"] = respondent_output is not None
    if respondent_output is not None:
        from athena.simulation.validation import validate_agent_output
        vr = validate_agent_output(
            respondent_output, "respondent", case, appellant_brief=appellant_brief
        )
        validation_results["respondent"] = vr.valid

    # --- 3. Judge ---
    judge_output = run_agent_test(
        agent_name="Judge",
        role="judge",
        build_context_fn=lambda: build_context_judge(
            SAMPLE_CASE_DATA, SAMPLE_RUN_PARAMS, appellant_brief, respondent_brief
        ),
        build_prompt_fn=build_judge_prompt,
        temperature=SAMPLE_RUN_PARAMS["temperature"]["judge"],
        case=case,
        dry_run=args.dry_run,
        appellant_brief=appellant_brief,
        respondent_brief=respondent_brief,
    )
    json_results["judge"] = judge_output is not None
    if judge_output is not None:
        from athena.simulation.validation import validate_agent_output
        vr = validate_agent_output(
            judge_output, "judge", case,
            appellant_brief=appellant_brief,
            respondent_brief=respondent_brief,
        )
        validation_results["judge"] = vr.valid

    # --- 4. Judge profile differentiation ---
    run_judge_profile_differentiation(
        case=case,
        appellant_brief=appellant_brief,
        respondent_brief=respondent_brief,
        dry_run=args.dry_run,
    )

    # --- Summary ---
    banner("Summary")
    if args.dry_run:
        print("  [DRY RUN] No model calls were made.")
        print("  Prompts were generated successfully for all 3 agents + 2 judge profiles.")
    else:
        print("  JSON valid rate:")
        for agent, ok in json_results.items():
            status = "OK" if ok else "FAILED"
            print(f"    {agent:>12}: {status}")
        total = len(json_results)
        passed = sum(1 for v in json_results.values() if v)
        print(f"    {'TOTAL':>12}: {passed}/{total}")

        print("\n  Validation results:")
        for agent, ok in validation_results.items():
            status = "PASS" if ok else "FAIL"
            print(f"    {agent:>12}: {status}")

        print("\n  Note: review reasoning quality indicators above for each agent.")

    print()


if __name__ == "__main__":
    main()
