def triage_decision(severity: str):
    if severity == "high":
        return "action_needed"
    elif severity == "medium":
        return "monitor"
    else:
        return "ignore"