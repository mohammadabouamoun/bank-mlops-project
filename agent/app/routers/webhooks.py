from fastapi import APIRouter
from contracts.v1 import DriftPayload
from agent.app.services.supervisor import triage_decision

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

@router.post("/drift")
def receive_drift_alert(payload: DriftPayload):
    
    decision = triage_decision(payload.severity)

    return {
        "message": "Drift alert received",
        "severity": payload.severity,
        "decision": decision
    }