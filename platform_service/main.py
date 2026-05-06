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

    # ---- Load the pipeline + threshold + feature names from disk ----
    try:
        artifact = await asyncio.to_thread(joblib.load, model_path)
        app.state.model_manager = ModelManager(
            pipeline=artifact["pipeline"],
            threshold=artifact["threshold"],
            feature_names=artifact["feature_names"],
            # Temporary version – will be overwritten below
            version="1",
        )
        log.info("model_loaded", threshold=app.state.model_manager.threshold)
    except Exception:
        log.exception("model_load_failed")
        raise RuntimeError("Could not load model") from None

    # ---- Query MLflow for the current Production version number ----
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        prod_versions = client.get_latest_versions(
            "bank_marketing_classifier", stages=["Production"]
        )
        if prod_versions:
            prod_version = str(prod_versions[0].version)   # ensure it's a string
            app.state.model_manager.version = prod_version
            log.info("production_version_resolved", version=prod_version)
        else:
            log.warning("no_production_version_found", default_version="1")
            app.state.model_manager.version = "1"   # fallback
    except Exception:
        log.exception("mlflow_version_lookup_failed")
        # Keep the default version – the server will still work

    # ---- Drift detector & HTTP client (unchanged) ----
    app.state.drift_detector = DriftDetector(
        reference_path=Path("models/reference.json"),
        window_path=Path("data/prediction_window.csv"),
        window_size=500,
        min_window_size=100,
    )
    log.info("drift_detector_initialised")

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

@app.get("/drift/latest")
async def get_latest_drift(
    request: Request,
    detector: DriftDetector = Depends(get_drift_detector),
):
    report = detector.compute_drift_report()
    return report

@app.post("/promote", response_model=PromoteResponse)
async def promote(
    body: PromoteRequest,
    request: Request,
    settings = Depends(get_settings_dep),
):
    """
    Programmatic promotion gate.
    1. Verify Authorization header.
    2. Fetch the requested model version and its run metrics/tags.
    3. Absolute checks: recall >= 0.75, AUC >= 0.70, threshold tag present.
    4. (Optional) Regression check: compare against current Production model.
    5. Transition the new version to Production in MLflow.
    """
    # ---------- Auth ----------
    auth_header = request.headers.get("Authorization", "")
    expected_key = f"Bearer {settings.promotion_api_key}"
    if auth_header != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ---------- Fetch model version from MLflow ----------
    import mlflow
    from mlflow.tracking import MlflowClient
    import json

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    try:
        mv = client.get_model_version("bank_marketing_classifier", body.model_version)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Model version {body.model_version} not found")

    run_id = mv.run_id
    if not run_id:
        raise HTTPException(status_code=400, detail="Model version has no run attached")

    # ---------- Retrieve stored metrics & tags from the run ----------
    run = client.get_run(run_id)
    metrics = run.data.metrics
    tags = run.data.tags

    model_recall = metrics.get("test_recall")
    model_auc = metrics.get("test_auc")
    stored_threshold = float(tags.get("threshold", 0)) if tags.get("threshold") else None

    if model_recall is None or model_auc is None or stored_threshold is None:
        raise HTTPException(
            status_code=400,
            detail="Promotion checklist failed: missing required metrics or threshold tag in model run",
        )

    # ---------- Absolute Day‑4 checklist ----------
    if model_recall < settings.promotion_min_recall:
        raise HTTPException(
            status_code=422,
            detail=f"Recall {model_recall:.4f} below required {settings.promotion_min_recall}",
        )
    if model_auc < settings.promotion_min_auc:
        raise HTTPException(
            status_code=422,
            detail=f"AUC {model_auc:.4f} below required {settings.promotion_min_auc}",
        )

    # ---------- Optional: Regression check against current Production ----------
    prod_versions = client.get_latest_versions(
        "bank_marketing_classifier", stages=["Production"]
    )
    if prod_versions:
        prod_mv = prod_versions[0]
        prod_run = client.get_run(prod_mv.run_id)
        prod_metrics = prod_run.data.metrics
        prod_recall = prod_metrics.get("test_recall")
        prod_auc = prod_metrics.get("test_auc")

        if prod_recall is not None and model_recall < prod_recall:
            raise HTTPException(
                status_code=422,
                detail=f"Recall {model_recall:.4f} is worse than current Production ({prod_recall:.4f})",
            )
        if prod_auc is not None and model_auc < prod_auc:
            raise HTTPException(
                status_code=422,
                detail=f"AUC {model_auc:.4f} is worse than current Production ({prod_auc:.4f})",
            )

    # ---------- Transition to Production ----------
    try:
        client.transition_model_version_stage(
            name="bank_marketing_classifier",
            version=body.model_version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow stage transition failed: {str(e)}")
    log.info("model_promoted", version=body.model_version, investigation_id=body.investigation_id)

    return PromoteResponse(
        message=f"Model version {body.model_version} promoted to Production",
        new_production_version=body.model_version,
    )