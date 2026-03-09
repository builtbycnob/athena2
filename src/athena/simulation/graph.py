# src/athena/simulation/graph.py
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


class GraphState(TypedDict):
    case: dict
    params: dict
    appellant_brief: dict | None
    appellant_validation: dict | None
    respondent_brief: dict | None
    respondent_validation: dict | None
    judge_decision: dict | None
    judge_validation: dict | None
    retry_count: int
    error: str | None


MAX_RETRIES = 2


def _node_appellant(state: GraphState) -> dict:
    ctx = build_context_appellant(state["case"], state["params"])
    system, user = build_appellant_prompt(ctx)
    temp = state["params"]["temperature"]["appellant"]
    try:
        output = invoke_llm(system, user, temp)
        case = CaseFile(**state["case"])
        validation = validate_agent_output(output, "appellant", case)
        if not validation.valid and state["retry_count"] < MAX_RETRIES:
            error_feedback = "\n".join(validation.errors)
            retry_user = f"{user}\n\n## ERRORI DA CORREGGERE\n{error_feedback}\n\nRiproduci l'output completo corretto."
            output = invoke_llm(system, retry_user, temp)
            validation = validate_agent_output(output, "appellant", case)
            return {
                "appellant_brief": output,
                "appellant_validation": validation.model_dump(),
                "retry_count": state["retry_count"] + 1,
            }
        return {
            "appellant_brief": output,
            "appellant_validation": validation.model_dump(),
        }
    except Exception as e:
        return {"error": f"Appellant failed: {e}"}


def _node_respondent(state: GraphState) -> dict:
    if state.get("error"):
        return {}
    ctx = build_context_respondent(
        state["case"], state["params"], state["appellant_brief"]
    )
    system, user = build_respondent_prompt(ctx)
    temp = state["params"]["temperature"]["respondent"]
    try:
        output = invoke_llm(system, user, temp)
        case = CaseFile(**state["case"])
        validation = validate_agent_output(
            output, "respondent", case,
            appellant_brief=state["appellant_brief"],
        )
        if not validation.valid and state["retry_count"] < MAX_RETRIES:
            error_feedback = "\n".join(validation.errors)
            retry_user = f"{user}\n\n## ERRORI DA CORREGGERE\n{error_feedback}\n\nRiproduci l'output completo corretto."
            output = invoke_llm(system, retry_user, temp)
            validation = validate_agent_output(
                output, "respondent", case,
                appellant_brief=state["appellant_brief"],
            )
            return {
                "respondent_brief": output,
                "respondent_validation": validation.model_dump(),
                "retry_count": state["retry_count"] + 1,
            }
        return {
            "respondent_brief": output,
            "respondent_validation": validation.model_dump(),
        }
    except Exception as e:
        return {"error": f"Respondent failed: {e}"}


def _node_judge(state: GraphState) -> dict:
    if state.get("error"):
        return {}
    ctx = build_context_judge(
        state["case"],
        state["params"],
        state["appellant_brief"],
        state["respondent_brief"],
    )
    system, user = build_judge_prompt(ctx)
    temp = state["params"]["temperature"]["judge"]
    try:
        output = invoke_llm(system, user, temp)
        case = CaseFile(**state["case"])
        validation = validate_agent_output(
            output, "judge", case,
            appellant_brief=state["appellant_brief"],
            respondent_brief=state["respondent_brief"],
        )
        if not validation.valid and state["retry_count"] < MAX_RETRIES:
            error_feedback = "\n".join(validation.errors)
            retry_user = f"{user}\n\n## ERRORI DA CORREGGERE\n{error_feedback}\n\nRiproduci l'output completo corretto."
            output = invoke_llm(system, retry_user, temp)
            validation = validate_agent_output(
                output, "judge", case,
                appellant_brief=state["appellant_brief"],
                respondent_brief=state["respondent_brief"],
            )
            return {
                "judge_decision": output,
                "judge_validation": validation.model_dump(),
                "retry_count": state["retry_count"] + 1,
            }
        return {
            "judge_decision": output,
            "judge_validation": validation.model_dump(),
        }
    except Exception as e:
        return {"error": f"Judge failed: {e}"}


def _should_continue(state: GraphState) -> str:
    if state.get("error"):
        return END
    return "next"


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
        "retry_count": 0,
        "error": None,
    }
    return graph.invoke(initial_state)
