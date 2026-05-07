from fastapi import FastAPI
from agent.app.routers.webhooks import router as webhooks_router
from agent.app.routers.approvals import router as approvals_router
from agent.app.routers.queue import router as queue_router

app = FastAPI(title="Drift Triage Agent")

app.include_router(webhooks_router)
app.include_router(approvals_router)
app.include_router(queue_router)


@app.get("/")
def root():
    return {"message": "Drift Triage Agent is running"}