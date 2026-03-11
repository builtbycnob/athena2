# src/athena/simulation/graph.py
import time

from langfuse import observe
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any

from athena.simulation.context import (
    build_context_appellant,
    build_context_respondent,
    build_context_judge,
)
from athena.simulation.validation import validate_agent_output
from athena.agents.prompts import (
    build_appellant_prompt,
    build_respondent_prompt,
    build_judge_prompt,
)
from athena.agents.llm import invoke_llm
from athena.schemas.case import CaseFile


def _log(msg: str) -> None:
    print(msg, flush=True)


class GraphState(TypedDict):
    case: dict
    params: dict
    appellant_brief: dict | None
    appellant_validation: dict | None
    respondent_brief: dict | None
    respondent_validation: dict | None
    judge_decision: dict | None
    judge_validation: dict | None
    error: str | None


MAX_RETRIES = 2


def _run_agent_with_retry(
    system: str, user: str, temp: float, validate_fn, max_retries: int = MAX_RETRIES
) -> tuple[dict, dict]:
    """Run LLM with validation and retry. Returns (output, validation_dump)."""
    output = invoke_llm(system, user, temp)
    validation = validate_fn(output)
    for _ in range(max_retries):
        if validation.valid:
            break
        error_feedback = "\n".join(validation.errors)
        retry_user = f"{user}\n\n## ERRORI DA CORREGGERE\n{error_feedback}\n\nRiproduci l'output completo corretto."
        output = invoke_llm(system, retry_user, temp)
        validation = validate_fn(output)
    return output, validation.model_dump()


@observe(name="appellant")
def _node_appellant(state: GraphState) -> dict:
    run_id = state["params"].get("run_id", "?")
    _log(f"[{run_id}]   Appellant: generating...")
    t0 = time.time()
    ctx = build_context_appellant(state["case"], state["params"])
    system, user = build_appellant_prompt(ctx)
    temp = state["params"]["temperature"]["appellant"]
    try:
        case = CaseFile(**state["case"])
        output, val = _run_agent_with_retry(
            system, user, temp,
            lambda o: validate_agent_output(o, "appellant", case),
        )
        _log(f"[{run_id}]   Appellant: done ({time.time()-t0:.1f}s, valid={val['valid']})")
        return {"appellant_brief": output, "appellant_validation": val}
    except Exception as e:
        _log(f"[{run_id}]   Appellant: FAILED ({time.time()-t0:.1f}s) — {e}")
        return {"error": f"Appellant failed: {e}"}


@observe(name="respondent")
def _node_respondent(state: GraphState) -> dict:
    if state.get("error"):
        return {}
    run_id = state["params"].get("run_id", "?")
    _log(f"[{run_id}]   Respondent: generating...")
    t0 = time.time()
    ctx = build_context_respondent(
        state["case"], state["params"], state["appellant_brief"]
    )
    system, user = build_respondent_prompt(ctx)
    temp = state["params"]["temperature"]["respondent"]
    try:
        case = CaseFile(**state["case"])
        output, val = _run_agent_with_retry(
            system, user, temp,
            lambda o: validate_agent_output(
                o, "respondent", case, appellant_brief=state["appellant_brief"]
            ),
        )
        _log(f"[{run_id}]   Respondent: done ({time.time()-t0:.1f}s, valid={val['valid']})")
        return {"respondent_brief": output, "respondent_validation": val}
    except Exception as e:
        _log(f"[{run_id}]   Respondent: FAILED ({time.time()-t0:.1f}s) — {e}")
        return {"error": f"Respondent failed: {e}"}


@observe(name="judge")
def _node_judge(state: GraphState) -> dict:
    if state.get("error"):
        return {}
    run_id = state["params"].get("run_id", "?")
    _log(f"[{run_id}]   Judge: generating...")
    t0 = time.time()
    ctx = build_context_judge(
        state["case"],
        state["params"],
        state["appellant_brief"],
        state["respondent_brief"],
    )
    system, user = build_judge_prompt(ctx)
    temp = state["params"]["temperature"]["judge"]
    try:
        case = CaseFile(**state["case"])
        output, val = _run_agent_with_retry(
            system, user, temp,
            lambda o: validate_agent_output(
                o, "judge", case,
                appellant_brief=state["appellant_brief"],
                respondent_brief=state["respondent_brief"],
            ),
        )
        _log(f"[{run_id}]   Judge: done ({time.time()-t0:.1f}s, valid={val['valid']})")
        return {"judge_decision": output, "judge_validation": val}
    except Exception as e:
        _log(f"[{run_id}]   Judge: FAILED ({time.time()-t0:.1f}s) — {e}")
        return {"error": f"Judge failed: {e}"}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("appellant", _node_appellant)
    graph.add_node("respondent", _node_respondent)
    graph.add_node("judge", _node_judge)

    graph.set_entry_point("appellant")
    graph.add_edge("appellant", "respondent")
    graph.add_edge("respondent", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


def run_single(case_data: dict, run_params: dict) -> dict:
    """Run a single simulation and return results."""
    graph = build_graph()
    initial_state: GraphState = {
        "case": case_data,
        "params": run_params,
        "appellant_brief": None,
        "appellant_validation": None,
        "respondent_brief": None,
        "respondent_validation": None,
        "judge_decision": None,
        "judge_validation": None,
        "error": None,
    }
    return graph.invoke(initial_state)
