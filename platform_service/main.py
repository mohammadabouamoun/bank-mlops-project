from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
import httpx
import structlog
import asyncio
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from contracts.v1 import (
    PredictRequest,
    PredictResponse,
    PromoteRequest,
    PromoteResponse,
    ErrorResponse,
)
from .dependencies import get_http_client, get_settings_dep

log = structlog.get_logger()


# ------------------------------------------------------------------
# Model manager – keeps pipeline + threshold + metadata together
# ------------------------------------------------------------------
class ModelManager:
    def __init__(self, pipeline, threshold, feature_names, version: str):
        self.pipeline = pipeline
        self.threshold = threshold
        self.feature_names = feature_names
        self.version = version


# ------------------------------------------------------------------
# Dependency injection helpers
# ------------------------------------------------------------------
async def get_model_manager(request: Request) -> ModelManager:
    """Dependency that yields the loaded ModelManager."""
    return request.app.state.model_manager


# ------------------------------------------------------------------
# App lifespan – load the model once
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings_dep()
    model_path = Path("models/model.pkl")

    # Load the artifact dict and wrap it in a manager
    try:
        artifact = await asyncio.to_thread(joblib.load, model_path)
        app.state.model_manager = ModelManager(
            pipeline=artifact["pipeline"],
            threshold=artifact["threshold"],
            feature_names=artifact["feature_names"],
            version="1",   # later you'll pull this from MLflow
        )
        log.info(
            "platform.startup.model_loaded",
            version=app.state.model_manager.version,
            threshold=app.state.model_manager.threshold,
        )
    except Exception:
        log.exception("platform.startup.model_load_failed")
        raise RuntimeError("Could not load model") from None

    # Shared HTTP client for webhook calls
    app.state.http_client = httpx.AsyncClient(timeout=10.0)
    log.info("platform.startup", mlflow_uri=settings.mlflow_tracking_uri)

    yield

    await app.state.http_client.aclose()
    log.info("platform.shutdown")


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------
# Prediction endpoint – fully typed, fully async, fully DI
# ------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    model: ModelManager = Depends(get_model_manager),
):
    """Accept a raw feature vector and return a prediction."""
    if len(body.features) != len(model.feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(model.feature_names)} features, got {len(body.features)}",
        )

    # Build DataFrame with correct column names
    X = pd.DataFrame([body.features], columns=model.feature_names)

    # CPU‑bound inference in thread
    proba = await asyncio.to_thread(model.pipeline.predict_proba, X)
    positive_proba = float(proba[0, 1])
    pred = int(positive_proba >= model.threshold)

    log.info("prediction", prediction=pred, probability=positive_proba, model_version=model.version)

    return PredictResponse(
        prediction=pred,
        probability=positive_proba,
        model_version=model.version,
    )


# ------------------------------------------------------------------
# Promotion endpoint – placeholder (will implement later)
# ------------------------------------------------------------------
@app.post("/promote", response_model=PromoteResponse)
async def promote(
    body: PromoteRequest,
    settings = Depends(get_settings_dep),
):
    # Will verify authorization header and MLflow promotion checklist
    raise HTTPException(status_code=501, detail="Promotion logic not implemented yet")