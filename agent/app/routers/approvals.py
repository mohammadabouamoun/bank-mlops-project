from fastapi import APIRouter
from agent.app.services.approval_service import pending_approvals
from agent.app.services.queue_service import add_to_queue

router = APIRouter(prefix="/approvals", tags=["approvals"])

@router.get("/pending")
def get_pending():
    return {"pending_approvals": pending_approvals}

@router.post("/{id}/approve")
def approve(id: int):
    for approval in pending_approvals:
        if approval["id"] == id:
            approval["status"] = "approved"

            job = add_to_queue(
                action=approval["action"],
                investigation_id=str(id)
            )

            return {
                "message": "Approved and queued successfully",
                "approval": approval,
                "job": job
            }

    return {"error": "Approval not found"}