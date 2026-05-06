from langgraph.graph import StateGraph, END


class AgentState(dict):
    pass


def triage_node(state):
    state["triage"] = "action_needed"
    return state


def action_node(state):
    state["action"] = "retrain_model"
    return state


def comms_node(state):
    state["message"] = "Retrain required"
    return state


builder = StateGraph(AgentState)

builder.add_node("triage", triage_node)
builder.add_node("action", action_node)
builder.add_node("comms", comms_node)

builder.set_entry_point("triage")

builder.add_edge("triage", "action")
builder.add_edge("action", "comms")
builder.add_edge("comms", END)

graph = builder.compile()