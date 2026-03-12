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


def _log(msg: str) -> None:
    print(msg, flush=True)


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
) -> tuple[dict, dict]:
    """Run LLM with validation and retry. Returns (output, validation_dump)."""
    kwargs: dict = {}
    if json_schema is not None:
        kwargs["json_schema"] = json_schema
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
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
            json_schema=AGENT_SCHEMAS.get(config.schema_key),
            max_tokens=config.max_tokens,
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
            json_schema=AGENT_SCHEMAS.get(config.schema_key),
            max_tokens=config.max_tokens,
        )
        _log(f"[{run_id}]   {config.party_id}: done ({time.time()-t0:.1f}s, valid={val['valid']})")
        return {
            "decision": output,
            "decision_validation": val,
        }
    except Exception as e:
        _log(f"[{run_id}]   {config.party_id}: FAILED ({time.time()-t0:.1f}s) — {e}")
        return {"error": f"{config.party_id} failed: {e}"}


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
            if agent.role_type == "adjudicator":
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

    # Get advocacy style
    appellant_profile = run_params.get("party_profiles", {}).get(appellant_id, {})
    style = appellant_profile.get("parameters", {}).get("style", "")
    if not style:
        style = run_params.get("appellant_profile", {}).get("style", "")

    return [
        Phase("filing", [AgentConfig(
            party_id=appellant_id,
            role_type="advocate",
            prompt_key=jconfig.prompt_keys["appellant"],
            schema_key=jconfig.schema_keys["appellant"],
            max_tokens=4096,
            temperature=temps.get("appellant", temps.get(appellant_id, 0.5)),
            template_vars={"advocacy_style": style},
        )]),
        Phase("response", [AgentConfig(
            party_id=respondent_id,
            role_type="advocate",
            prompt_key=jconfig.prompt_keys["respondent"],
            schema_key=jconfig.schema_keys["respondent"],
            max_tokens=4096,
            temperature=temps.get("respondent", temps.get(respondent_id, 0.4)),
        )]),
        Phase("decision", [AgentConfig(
            party_id="judge",
            role_type="adjudicator",
            prompt_key=jconfig.prompt_keys["judge"],
            schema_key=jconfig.schema_keys["judge"],
            max_tokens=6144,
            temperature=temps.get("judge", 0.3),
        )]),
    ]



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
