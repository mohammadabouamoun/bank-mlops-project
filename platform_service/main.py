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
from .drift import DriftDetector

log = structlog.get_logger()


# ------------------------------------------------------------------
# ModelManager
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
    return request.app.state.model_manager


async def get_drift_detector(request: Request) -> DriftDetector:
    return request.app.state.drift_detector


# ------------------------------------------------------------------
# Helper: safe fire‑and‑forget webhook sender
# ------------------------------------------------------------------
async def _send_webhook_safe(client: httpx.AsyncClient, url: str, payload: dict) -> None:
    """Send drift webhook, log failures but never raise."""
    try:
        resp = await client.post(url, json=payload)
        log.info("webhook_sent", status_code=resp.status_code)
    except httpx.ConnectError:
        log.warning("webhook_failed", url=url, reason="Agent unreachable")
    except Exception:
        log.exception("webhook_unexpected_error", url=url)


# ------------------------------------------------------------------
# App lifespan
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings_dep()
    model_path = Path("models/model.pkl")

    # Load model
    try:
        artifact = await asyncio.to_thread(joblib.load, model_path)
        app.state.model_manager = ModelManager(
            pipeline=artifact["pipeline"],
            threshold=artifact["threshold"],
            feature_names=artifact["feature_names"],
            version="1",
        )
        log.info("model_loaded", threshold=app.state.model_manager.threshold)
    except Exception:
        log.exception("model_load_failed")
        raise RuntimeError("Could not load model") from None

    # Drift detector (change _last_severity initialisation to "low" in drift.py!)
    app.state.drift_detector = DriftDetector(
        reference_path=Path("models/reference.json"),
        window_path=Path("data/prediction_window.csv"),
        window_size=500,
        min_window_size=100,
    )
    log.info("drift_detector_initialised")

    # HTTP client for webhooks
    app.state.http_client = httpx.AsyncClient(timeout=10.0)
    log.info("platform_startup")

    yield

    await app.state.http_client.aclose()
    log.info("platform_shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    request: Request,
    model: ModelManager = Depends(get_model_manager),
    drift_detector: DriftDetector = Depends(get_drift_detector),
    settings = Depends(get_settings_dep),
):
    """Make a prediction and check for drift."""
    if len(body.features) != len(model.feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(model.feature_names)} features, got {len(body.features)}",
        )

    # Build DataFrame with correct column names
    X = pd.DataFrame([body.features], columns=model.feature_names)

    # CPU‑bound inference
    proba = await asyncio.to_thread(model.pipeline.predict_proba, X)
    positive_proba = float(proba[0, 1])
    pred = int(positive_proba >= model.threshold)

    # Drift check
    features_dict = dict(zip(model.feature_names, body.features))
    drift_report = drift_detector.check_and_report(features_dict, positive_proba)

    if drift_report is not None:
        agent_url = settings.agent_webhook_url
        log.info("drift_severity_changed", severity=drift_report["severity"], agent_url=agent_url)
        # Safe fire‑and‑forget – never crash the platform
        asyncio.create_task(
            _send_webhook_safe(request.app.state.http_client, agent_url, drift_report)
        )

    log.info("prediction", prediction=pred, probability=positive_proba, model_version=model.version)

    return PredictResponse(
        prediction=pred,
        probability=positive_proba,
        model_version=model.version,
    )


@app.post("/promote", response_model=PromoteResponse)
async def promote(
    body: PromoteRequest,
    settings = Depends(get_settings_dep),
):
    raise HTTPException(status_code=501, detail="Promotion logic not implemented yet")