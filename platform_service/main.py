from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
import httpx
import structlog
import asyncio

from contracts.v1 import (
    PredictRequest,
    PredictResponse,
    PromoteRequest,
    PromoteResponse,
    ErrorResponse,
)
from .dependencies import get_model, get_http_client, get_settings_dep

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings_dep()
    # Placeholder: you'll load your trained model here later
    app.state.model = None  # replace with joblib.load(...)
    app.state.http_client = httpx.AsyncClient(timeout=10.0)
    log.info("platform.startup", mlflow_uri=settings.mlflow_tracking_uri)
    yield
    await app.state.http_client.aclose()
    log.info("platform.shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Placeholder endpoints – will be fully implemented after model training
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, model=Depends(get_model)):
    # Will run model.predict() inside asyncio.to_thread later
    raise HTTPException(status_code=501, detail="Not yet implemented – train the model first")


@app.post("/promote", response_model=PromoteResponse)
async def promote(request: PromoteRequest, settings=Depends(get_settings_dep)):
    # Will enforce promotion checklist and auth header later
    raise HTTPException(status_code=501, detail="Not yet implemented")