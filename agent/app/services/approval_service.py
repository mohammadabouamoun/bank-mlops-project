pending_approvals = []

def create_approval(action: str, severity: str):
    approval = {
        "id": len(pending_approvals) + 1,
        "action": action,
        "severity": severity,
        "status": "pending"
    }
    pending_approvals.append(approval)
    return approval