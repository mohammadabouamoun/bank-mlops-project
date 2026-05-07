from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"


def load_prompt(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


TRIAGE_PROMPT = load_prompt("triage.txt")
ACTION_PROMPT = load_prompt("action.txt")
COMMS_PROMPT = load_prompt("comms.txt")


def triage_agent(severity: str):
    if severity == "high":
        return {
            "triage_decision": "action_needed",
            "reason": "High drift detected"
        }

    if severity == "medium":
        return {
            "triage_decision": "monitor",
            "reason": "Medium drift detected"
        }

    return {
        "triage_decision": "ignore",
        "reason": "Low drift detected"
    }


def action_agent(triage_result: dict):
    decision = triage_result["triage_decision"]

    if decision == "action_needed":
        return {
            "action": "retrain_model",
            "requires_approval": True
        }

    if decision == "monitor":
        return {
            "action": "log_only",
            "requires_approval": False
        }

    return {
        "action": "do_nothing",
        "requires_approval": False
    }


def comms_agent(triage_result: dict, action_result: dict):
    return {
        "message": (
            f"Triage result: {triage_result['triage_decision']}. "
            f"Action selected: {action_result['action']}."
        )
    }


def supervisor_flow(severity: str):
    triage_result = triage_agent(severity)
    action_result = action_agent(triage_result)
    comms_result = comms_agent(triage_result, action_result)

    return {
        "decision": triage_result["triage_decision"],
        "reason": triage_result["reason"],
        "action": action_result["action"],
        "requires_approval": action_result["requires_approval"],
        "message": comms_result["message"]
    }