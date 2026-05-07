from fastapi import APIRouter
from agent.app.services.queue_service import get_queue, get_dlq

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("/status")
def get_queue_status():
    queued_jobs = get_queue()
    dlq_jobs = get_dlq()

    return {
        "queue_depth": len(queued_jobs),
        "dlq_depth": len(dlq_jobs),
        "queued_jobs": queued_jobs,
        "dlq_jobs": dlq_jobs
    }