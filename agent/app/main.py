from fastapi import FastAPI
from agent.app.routers.webhooks import router as webhooks_router

app = FastAPI(title="Drift Triage Agent")

app.include_router(webhooks_router)

@app.get("/")
def root():
    return {"status": "Agent service is running"}