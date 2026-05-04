# platform/dependencies.py
from contracts.settings import get_settings
from fastapi import Depends, Request
import joblib

def get_model(request: Request):
    return request.app.state.model

def get_settings_dep():
    return get_settings()

# platform/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
import httpx

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.model = joblib.load("model.pkl")
    app.state.mlflow_uri = settings.mlflow_tracking_uri
    # async client for webhook
    app.state.http_client = httpx.AsyncClient(timeout=10.0)
    yield
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: PredictRequest, model=Depends(get_model)):
    # CPU-bound: run in thread
    import asyncio
    prediction = await asyncio.to_thread(model.predict, request.features)
    return {"prediction": prediction}