from fastapi import APIRouter
from contracts.v1 import DriftPayload
from agent.app.services.supervisor import supervisor_flow
from agent.app.services.approval_service import create_approval

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/drift")
def receive_drift_alert(payload: DriftPayload):
    result = supervisor_flow(payload.severity)

    approval = None
    if result["action"] in ["retrain_model", "rollback_model"]:
        approval = create_approval(
            action=result["action"],
            severity=payload.severity
        )

    return {
        "message": "Drift alert received",
        "severity": payload.severity,
        **result,
        "approval": approval
    }