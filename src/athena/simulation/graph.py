# src/athena/simulation/graph.py
import time
from dataclasses import dataclass, field
from functools import partial

from langfuse import observe
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

from athena.simulation.context import (
    build_party_context,
    build_adjudicator_context,
    _sanitize_brief,
)
from athena.simulation.validation import validate_agent_output
from athena.agents.prompts import (
    build_appellant_prompt,
    build_respondent_prompt,
    build_judge_prompt,
)
from athena.agents.llm import invoke_llm
from athena.schemas.case import CaseFile
from athena.schemas.structured_output import AGENT_SCHEMAS
from athena.schemas.schema_builder import build_schema_for_agent


def _log(msg: str) -> None:
    print(msg, flush=True)


def _inject_rag_context(ctx: dict, case_data: dict) -> None:
    """Inject RAG-retrieved legal norms into adjudicator context if enabled."""
    from athena.rag import is_rag_enabled, retrieve_norms
    if not is_rag_enabled():
        return

    jurisdiction = case_data.get("jurisdiction", {})
    country = jurisdiction.get("country", "IT") if isinstance(jurisdiction, dict) else "IT"

    # Collect seed arguments from all parties
    all_seeds = []
    by_party = case_data.get("seed_arguments", {}).get("by_party", {})
    for party_args in by_party.values():
        if isinstance(party_args, list):
            all_seeds.extend(party_args)

    norms = retrieve_norms(
        seed_arguments=all_seeds,
        facts=case_data.get("facts", {}),
        existing_legal_texts=case_data.get("legal_texts", []),
        jurisdiction=country,
    )
    if norms:
        ctx["rag_legal_texts"] = [
            {"sr_number": n.get("sr_number", ""), "article": n.get("article_number", ""),
             "text": n.get("text", ""), "breadcrumb": n.get("section_breadcrumb", "")}
            for n in norms
        ]


# --- AgentConfig & Phase dataclasses ---

@dataclass
class AgentConfig:
    party_id: str
    role_type: str          # "advocate" | "adjudicator"
    prompt_key: str         # registry lookup
    schema_key: str         # AGENT_SCHEMAS lookup
    max_tokens: int
    temperature: float
    template_vars: dict = field(default_factory=dict)
    model: str | None = None  # per-agent model override (None = use OMLX_MODEL default)


@dataclass
class Phase:
    name: str
    agents: list[AgentConfig]


# --- State reducers ---

def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b overwriting a."""
    merged = dict(a)
    merged.update(b)
    return merged


class GraphState(TypedDict):
    case: dict
    params: dict
    briefs: Annotated[dict, _merge_dicts]        # party_id → brief
    validations: Annotated[dict, _merge_dicts]    # party_id → validation
    decision: dict | None
    decision_validation: dict | None
    error: str | None


MAX_RETRIES = 2

_MAX_TOKENS = {
    "appellant": 4096,
    "respondent": 4096,
    "judge": 6144,
    "advocate_filing": 4096,
    "advocate_response": 4096,
    "adjudicator": 6144,
}


def _run_agent_with_retry(
    system: str,
    user: str,
    temp: float,
    validate_fn,
    max_retries: int = MAX_RETRIES,
    json_schema: dict | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> tuple[dict, dict]:
    """Run LLM with validation and retry. Returns (output, validation_dump)."""
    kwargs: dict = {}
    if json_schema is not None:
        kwargs["json_schema"] = json_schema
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if model is not None:
        kwargs["model"] = model
    output = invoke_llm(system, user, temp, **kwargs)
    validation = validate_fn(output)
    for _ in range(max_retries):
        if validation.valid:
            break
        error_feedback = "\n".join(validation.errors)
        retry_user = f"{user}\n\n## ERRORI DA CORREGGERE\n{error_feedback}\n\nRiproduci l'output completo corretto."
        output = invoke_llm(system, retry_user, temp, **kwargs)
        validation = validate_fn(output)
    return output, validation.model_dump()


# --- Generic node functions ---

@observe(name="party_agent")
def _node_party(state: GraphState, *, config: AgentConfig) -> dict:
    """Generic node for any advocate party agent."""
    if state.get("error"):
        return {}
    run_id = state["params"].get("run_id", "?")
    _log(f"[{run_id}]   {config.party_id}: generating...")
    t0 = time.time()

    # Collect prior briefs (sanitized)
    prior_briefs = {}
    for pid, brief in state["briefs"].items():
        if brief is not None:
            prior_briefs[pid] = _sanitize_brief(brief)

    # Build context
    ctx = build_party_context(state["case"], state["params"], config.party_id, prior_briefs)

    # Inject advocacy_style for prompt template substitution
    if "advocacy_style" not in ctx and config.template_vars.get("advocacy_style"):
        ctx["advocacy_style"] = config.template_vars["advocacy_style"]
    # Inject appellant_brief for respondent prompts (both IT and CH)
    if config.prompt_key in ("respondent_it", "respondent_ch") and prior_briefs:
        parties = state["case"].get("parties", [])
        appellant_id = next((p["id"] for p in parties if p["role"] == "appellant"), None)
        if appellant_id and appellant_id in prior_briefs:
            ctx["appellant_brief"] = prior_briefs[appellant_id]
        elif prior_briefs:
            ctx["appellant_brief"] = next(iter(prior_briefs.values()))

    system, user = _get_prompt_for_config(config, ctx)
    temp = config.temperature

    try:
        case = CaseFile(**state["case"])
        all_prior_briefs = {pid: state["briefs"][pid] for pid in state["briefs"]
                           if state["briefs"][pid] is not None}
        output, val = _run_agent_with_retry(
            system, user, temp,
            lambda o: validate_agent_output(
                o, config.role_type, case, prior_briefs=all_prior_briefs
            ),
            json_schema=build_schema_for_agent(
                config.schema_key, state["case"], prior_briefs=all_prior_briefs
            ),
            max_tokens=config.max_tokens,
            model=config.model,
        )
        _log(f"[{run_id}]   {config.party_id}: done ({time.time()-t0:.1f}s, valid={val['valid']})")
        return {
            "briefs": {config.party_id: output},
            "validations": {config.party_id: val},
        }
    except Exception as e:
        _log(f"[{run_id}]   {config.party_id}: FAILED ({time.time()-t0:.1f}s) — {e}")
        return {"error": f"{config.party_id} failed: {e}"}


@observe(name="adjudicator_agent")
def _node_adjudicator(state: GraphState, *, config: AgentConfig) -> dict:
    """Node for judge/arbitrator."""
    if state.get("error"):
        return {}
    run_id = state["params"].get("run_id", "?")
    _log(f"[{run_id}]   {config.party_id}: generating...")
    t0 = time.time()

    # Build adjudicator context with all briefs
    all_briefs = {pid: brief for pid, brief in state["briefs"].items()
                  if brief is not None}
    ctx = build_adjudicator_context(state["case"], state["params"], all_briefs)

    # RAG: inject retrieved legal norms
    _inject_rag_context(ctx, state["case"])

    # Provide separate brief keys for judge prompts (IT and CH)
    if config.prompt_key in ("judge_it", "judge_ch"):
        parties = state["case"].get("parties", [])
        for p in parties:
            if p["role"] == "appellant" and p["id"] in all_briefs:
                ctx["appellant_brief"] = _sanitize_brief(all_briefs[p["id"]])
            if p["role"] == "respondent" and p["id"] in all_briefs:
                ctx["respondent_brief"] = _sanitize_brief(all_briefs[p["id"]])

    system, user = _get_prompt_for_config(config, ctx)
    temp = config.temperature

    try:
        case = CaseFile(**state["case"])
        output, val = _run_agent_with_retry(
            system, user, temp,
            lambda o: validate_agent_output(
                o, config.role_type, case, prior_briefs=all_briefs
            ),
            json_schema=build_schema_for_agent(
                config.schema_key, state["case"], prior_briefs=all_briefs
            ),
            max_tokens=config.max_tokens,
            model=config.model,
        )
        _log(f"[{run_id}]   {config.party_id}: done ({time.time()-t0:.1f}s, valid={val['valid']})")
        return {
            "decision": output,
            "decision_validation": val,
        }
    except Exception as e:
        _log(f"[{run_id}]   {config.party_id}: FAILED ({time.time()-t0:.1f}s) — {e}")
        return {"error": f"{config.party_id} failed: {e}"}


@observe(name="adjudicator_two_step")
def _node_adjudicator_two_step(
    state: GraphState,
    *,
    step1_config: AgentConfig,
    step2_config: AgentConfig,
) -> dict:
    """Two-step judge node: Step 1 identifies errors, Step 2 decides outcome.

    Breaks cascading bias by making Step 1 errors INPUT to Step 2
    instead of self-generated context under constrained decoding.
    """
    if state.get("error"):
        return {}
    run_id = state["params"].get("run_id", "?")
    _log(f"[{run_id}]   judge (two-step): starting step 1 (error identification)...")
    t0 = time.time()

    # Build adjudicator context with all briefs
    all_briefs = {pid: brief for pid, brief in state["briefs"].items()
                  if brief is not None}
    ctx = build_adjudicator_context(state["case"], state["params"], all_briefs)

    # RAG: inject retrieved legal norms
    _inject_rag_context(ctx, state["case"])

    # Provide separate brief keys for judge prompts
    parties = state["case"].get("parties", [])
    for p in parties:
        if p["role"] == "appellant" and p["id"] in all_briefs:
            ctx["appellant_brief"] = _sanitize_brief(all_briefs[p["id"]])
        if p["role"] == "respondent" and p["id"] in all_briefs:
            ctx["respondent_brief"] = _sanitize_brief(all_briefs[p["id"]])

    try:
        case = CaseFile(**state["case"])

        # --- Step 1: Error Identification ---
        system1, user1 = _get_prompt_for_config(step1_config, ctx)
        step1_output, val1 = _run_agent_with_retry(
            system1, user1, step1_config.temperature,
            lambda o: validate_agent_output(
                o, step1_config.role_type, case, prior_briefs=all_briefs
            ),
            json_schema=build_schema_for_agent(
                step1_config.schema_key, state["case"], prior_briefs=all_briefs
            ),
            max_tokens=step1_config.max_tokens,
            model=step1_config.model,
        )
        t1 = time.time()
        _log(f"[{run_id}]   judge step 1: done ({t1-t0:.1f}s, valid={val1['valid']})")

        # --- Step 2: Outcome Decision ---
        _log(f"[{run_id}]   judge (two-step): starting step 2 (outcome decision)...")

        # Format Step 1 errors as text for injection into Step 2 prompt
        import json as _json
        errors = step1_output.get("identified_errors", [])
        if errors:
            errors_text = _json.dumps(errors, indent=2, ensure_ascii=False)
        else:
            errors_text = "Nessun errore identificato nella decisione impugnata."

        # Also include argument evaluation summary
        arg_eval = step1_output.get("argument_evaluation", [])
        arg_eval_text = _json.dumps(arg_eval, indent=2, ensure_ascii=False) if arg_eval else "[]"

        error_analysis = step1_output.get("error_analysis_reasoning", "")

        step1_errors_block = (
            f"### Errori identificati\n{errors_text}\n\n"
            f"### Sintesi dell'analisi errori\n{error_analysis}\n\n"
            f"### Valutazione argomenti\n{arg_eval_text}"
        )

        # Inject into Step 2 context via template_vars
        step2_vars = dict(step2_config.template_vars)
        step2_vars["step1_errors_text"] = step1_errors_block

        # Also inject judge profile vars from step1
        step2_vars.setdefault("jurisprudential_orientation",
                              step1_config.template_vars.get("jurisprudential_orientation", "follows_cassazione"))
        step2_vars.setdefault("formalism",
                              step1_config.template_vars.get("formalism", "high"))

        step2_config_with_vars = AgentConfig(
            party_id=step2_config.party_id,
            role_type=step2_config.role_type,
            prompt_key=step2_config.prompt_key,
            schema_key=step2_config.schema_key,
            max_tokens=step2_config.max_tokens,
            temperature=step2_config.temperature,
            template_vars=step2_vars,
        )

        system2, user2 = _get_prompt_for_config(step2_config_with_vars, ctx)
        step2_output, val2 = _run_agent_with_retry(
            system2, user2, step2_config.temperature,
            lambda o: validate_agent_output(
                o, step2_config.role_type, case, prior_briefs=all_briefs
            ),
            json_schema=build_schema_for_agent(
                step2_config.schema_key, state["case"], prior_briefs=all_briefs,
                step1_error_count=len(step1_output.get("identified_errors", [])),
            ),
            max_tokens=step2_config.max_tokens,
            model=step2_config.model,
        )
        t2 = time.time()
        _log(f"[{run_id}]   judge step 2: done ({t2-t1:.1f}s, valid={val2['valid']})")

        # --- Severity calibration: floor + ceiling ---
        _SEV_ORDER = {"none": 0, "minor": 1, "significant": 2, "decisive": 3}
        _SEV_NAMES = {v: k for k, v in _SEV_ORDER.items()}
        step1_errors = step1_output.get("identified_errors", [])
        step2_assessments = step2_output.get("error_assessment", [])
        step1_by_idx = {i: err for i, err in enumerate(step1_errors)}
        for ea in step2_assessments:
            eid = ea.get("error_id")
            if eid is not None and eid in step1_by_idx:
                s1_sev = step1_by_idx[eid].get("severity", "none")
                s2_sev = ea.get("confirmed_severity", "none")
                s1_level = _SEV_ORDER.get(s1_sev, 0)
                s2_level = _SEV_ORDER.get(s2_sev, 0)
                # Floor: Step 1 decisive can't be downgraded below significant
                if s1_sev == "decisive" and s2_sev in ("minor", "none"):
                    _log(f"[{run_id}]   severity floor: err{eid} Step1=decisive, Step2={s2_sev} → raising to significant")
                    ea["confirmed_severity"] = "significant"
                # Ceiling: Step 2 can upgrade at most +1 level
                elif s2_level > s1_level + 1:
                    capped = _SEV_NAMES[s1_level + 1]
                    _log(f"[{run_id}]   severity ceiling: err{eid} Step1={s1_sev}, Step2={s2_sev} → capping to {capped}")
                    ea["confirmed_severity"] = capped

        # --- Merge: Step 1 analysis + Step 2 decision ---
        # Consistency override: if Step 2 confirmed any error as "decisive",
        # lower_court_correct MUST be False (logical invariant).
        lcc = step2_output.get("lower_court_correct", True)
        has_decisive = any(
            e.get("confirmed_severity") == "decisive"
            for e in step2_assessments
        )
        if has_decisive and lcc:
            _log(f"[{run_id}]   consistency fix: decisive error confirmed but LCC=True → forcing False")
            lcc = False

        # Build output compatible with JUDGE_CH_SCHEMA format for downstream
        merged = {
            "preliminary_objections_ruling": step1_output.get("preliminary_objections_ruling", []),
            "case_reaches_merits": step1_output.get("case_reaches_merits", True),
            "argument_evaluation": step1_output.get("argument_evaluation", []),
            "precedent_analysis": step1_output.get("precedent_analysis", {}),
            "verdict": {
                "identified_errors": step1_errors,
                "error_assessment": step2_assessments,
                "lower_court_correct": lcc,
                "correctness_reasoning": step2_output.get("correctness_reasoning", ""),
                "if_incorrect": step2_output.get("if_incorrect"),
                "if_correct": step2_output.get("if_correct"),
                "costs_ruling": step2_output.get("costs_ruling", ""),
            },
            "reasoning": step1_output.get("error_analysis_reasoning", ""),
            "gaps": [],
        }

        _log(f"[{run_id}]   judge (two-step): total {t2-t0:.1f}s")
        return {
            "decision": merged,
            "decision_validation": val2,
        }
    except Exception as e:
        _log(f"[{run_id}]   judge (two-step): FAILED ({time.time()-t0:.1f}s) — {e}")
        return {"error": f"judge (two-step) failed: {e}"}


def _get_prompt_for_config(config: AgentConfig, ctx: dict) -> tuple[str, str]:
    """Get prompt for an agent config — uses legacy builders for IT, registry for others."""
    if config.prompt_key == "appellant_it":
        return build_appellant_prompt(ctx)
    elif config.prompt_key == "respondent_it":
        return build_respondent_prompt(ctx)
    elif config.prompt_key == "judge_it":
        return build_judge_prompt(ctx)
    else:
        from athena.agents.prompt_registry import build_party_prompt
        return build_party_prompt(ctx, config.prompt_key, config.template_vars)


# --- Graph construction ---

def build_graph_from_phases(phases: list[Phase]) -> object:
    """Build LangGraph from phase definitions."""
    graph = StateGraph(GraphState)
    prev_nodes: list[str] = []

    for i, phase in enumerate(phases):
        current_nodes: list[str] = []
        for agent in phase.agents:
            node_name = f"phase{i}_{agent.party_id}"
            if agent.role_type == "adjudicator_two_step":
                # Build Step 2 config from template_vars metadata
                step2_config = AgentConfig(
                    party_id=agent.party_id,
                    role_type="adjudicator",
                    prompt_key=agent.template_vars["_step2_prompt_key"],
                    schema_key=agent.template_vars["_step2_schema_key"],
                    max_tokens=agent.template_vars.get("_step2_max_tokens", agent.max_tokens),
                    temperature=agent.template_vars["_step2_temperature"],
                    template_vars={k: v for k, v in agent.template_vars.items()
                                   if not k.startswith("_step2_")},
                    model=agent.model,  # inherit model from step1 config
                )
                graph.add_node(node_name, partial(
                    _node_adjudicator_two_step,
                    step1_config=agent,
                    step2_config=step2_config,
                ))
            elif agent.role_type == "adjudicator":
                graph.add_node(node_name, partial(_node_adjudicator, config=agent))
            else:
                graph.add_node(node_name, partial(_node_party, config=agent))
            current_nodes.append(node_name)

        # Wire edges
        if not prev_nodes:
            # First phase: first node is entry point, rest chained sequentially
            graph.set_entry_point(current_nodes[0])
            for j in range(1, len(current_nodes)):
                graph.add_edge(current_nodes[j-1], current_nodes[j])
        else:
            # Connect last node of previous phase to first of current
            graph.add_edge(prev_nodes[-1], current_nodes[0])
            for j in range(1, len(current_nodes)):
                graph.add_edge(current_nodes[j-1], current_nodes[j])

        prev_nodes = current_nodes

    # Final node → END
    if prev_nodes:
        graph.add_edge(prev_nodes[-1], END)

    return graph.compile()


def build_bilateral_phases(case_data: dict, run_params: dict) -> list[Phase]:
    """Build phases for the standard bilateral case (jurisdiction-aware)."""
    from athena.jurisdiction import get_jurisdiction_for_case

    jconfig = get_jurisdiction_for_case(case_data)

    # Find party IDs
    appellant_id = None
    respondent_id = None
    for p in case_data.get("parties", []):
        if p["role"] == "appellant":
            appellant_id = p["id"]
        elif p["role"] == "respondent":
            respondent_id = p["id"]

    appellant_id = appellant_id or "opponente"
    respondent_id = respondent_id or "controparte"

    # Get temperatures
    temps = run_params.get("temperatures", run_params.get("temperature", {}))

    # Get per-role model overrides: simulation YAML > jurisdiction defaults > None (global)
    sim_models = run_params.get("models", {})
    jur_models = jconfig.default_models

    def _model_for_role(role: str) -> str | None:
        """Resolve model for a role: sim YAML override > jurisdiction default > None."""
        return sim_models.get(role) or jur_models.get(role) or None

    # Get advocacy style
    appellant_profile = run_params.get("party_profiles", {}).get(appellant_id, {})
    style = appellant_profile.get("parameters", {}).get("style", "")
    if not style:
        style = run_params.get("appellant_profile", {}).get("style", "")

    phases = [
        Phase("filing", [AgentConfig(
            party_id=appellant_id,
            role_type="advocate",
            prompt_key=jconfig.prompt_keys["appellant"],
            schema_key=jconfig.schema_keys["appellant"],
            max_tokens=4096,
            temperature=temps.get("appellant", temps.get(appellant_id, 0.5)),
            template_vars={"advocacy_style": style},
            model=_model_for_role("appellant"),
        )]),
        Phase("response", [AgentConfig(
            party_id=respondent_id,
            role_type="advocate",
            prompt_key=jconfig.prompt_keys["respondent"],
            schema_key=jconfig.schema_keys["respondent"],
            max_tokens=4096,
            temperature=temps.get("respondent", temps.get(respondent_id, 0.4)),
            model=_model_for_role("respondent"),
        )]),
    ]

    if jconfig.judge_two_step:
        # Two-step judge: separate error identification from outcome decision
        judge_temp = temps.get("judge", jconfig.default_temperatures.get("judge", 0.7))
        # Get judge profile for template_vars
        judge_profile = run_params.get("judge_profile", {})
        if isinstance(judge_profile, str):
            judge_profile = {"jurisprudential_orientation": "follows_cassazione", "formalism": judge_profile}
        tvars = {
            "jurisprudential_orientation": judge_profile.get("jurisprudential_orientation", "follows_cassazione"),
            "formalism": judge_profile.get("formalism", "high"),
        }
        phases.append(Phase("decision", [AgentConfig(
            party_id="judge",
            role_type="adjudicator_two_step",
            prompt_key=jconfig.judge_step1_prompt_key,
            schema_key=jconfig.judge_step1_schema_key,
            max_tokens=8192,
            temperature=jconfig.judge_step1_temperature or judge_temp,
            template_vars={
                **tvars,
                "_step2_prompt_key": jconfig.judge_step2_prompt_key,
                "_step2_schema_key": jconfig.judge_step2_schema_key,
                "_step2_temperature": jconfig.judge_step2_temperature or 0.4,
                "_step2_max_tokens": 6144,
            },
            model=_model_for_role("judge"),
        )]))
    else:
        phases.append(Phase("decision", [AgentConfig(
            party_id="judge",
            role_type="adjudicator",
            prompt_key=jconfig.prompt_keys["judge"],
            schema_key=jconfig.schema_keys["judge"],
            max_tokens=6144,
            temperature=temps.get("judge", jconfig.default_temperatures.get("judge", 0.3)),
            model=_model_for_role("judge"),
        )]))

    return phases



def run_single(case_data: dict, run_params: dict) -> dict:
    """Run a single simulation and return results.

    Returns a dict with legacy-compatible keys:
        appellant_brief, respondent_brief, judge_decision,
        appellant_validation, respondent_validation, judge_validation,
        error.
    """
    phases = build_bilateral_phases(case_data, run_params)
    graph = build_graph_from_phases(phases)
    initial_state: GraphState = {
        "case": case_data,
        "params": run_params,
        "briefs": {},
        "validations": {},
        "decision": None,
        "decision_validation": None,
        "error": None,
    }
    final = graph.invoke(initial_state)

    # Map GraphState to legacy-compatible result dict
    briefs = final.get("briefs", {})
    validations = final.get("validations", {})

    appellant_id = next(
        (p["id"] for p in case_data.get("parties", []) if p["role"] == "appellant"),
        "opponente",
    )
    respondent_id = next(
        (p["id"] for p in case_data.get("parties", []) if p["role"] == "respondent"),
        "controparte",
    )

    return {
        "case": case_data,
        "params": run_params,
        "appellant_brief": briefs.get(appellant_id),
        "appellant_validation": validations.get(appellant_id),
        "respondent_brief": briefs.get(respondent_id),
        "respondent_validation": validations.get(respondent_id),
        "judge_decision": final.get("decision"),
        "judge_validation": final.get("decision_validation"),
        "error": final.get("error"),
    }
