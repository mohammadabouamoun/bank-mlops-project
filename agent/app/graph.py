from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import connect


DB_URI = "postgresql://agent:agentpass@localhost:5432/agentdb"

conn = connect(DB_URI)

checkpointer = PostgresSaver(conn)


class AgentState(dict):
    pass


def triage_node(state):
    severity = state.get("severity", "low")

    if severity == "high":
        state["triage"] = "action_needed"
    elif severity == "medium":
        state["triage"] = "monitor"
    else:
        state["triage"] = "ignore"

    return state


def action_node(state):
    triage = state.get("triage")

    if triage == "action_needed":
        state["action"] = "retrain_model"
        state["requires_approval"] = True

    elif triage == "monitor":
        state["action"] = "log_only"
        state["requires_approval"] = False

    else:
        state["action"] = "do_nothing"
        state["requires_approval"] = False

    return state


def comms_node(state):
    state["message"] = (
        f"Triage: {state['triage']} | "
        f"Action: {state['action']}"
    )

    return state


workflow = StateGraph(AgentState)

workflow.add_node("triage", triage_node)
workflow.add_node("action", action_node)
workflow.add_node("comms", comms_node)

workflow.set_entry_point("triage")

workflow.add_edge("triage", "action")
workflow.add_edge("action", "comms")
workflow.add_edge("comms", END)

graph = workflow.compile(
    checkpointer=checkpointer
)