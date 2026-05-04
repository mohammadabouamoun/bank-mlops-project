# contracts/v1.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# --------------------- Prediction endpoint ---------------------
class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Preprocessed feature vector for one sample")


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: Optional[float] = Field(None, description="Probability of the positive class")
    model_version: Optional[str] = Field(None, description="MLflow model version in Production")


# --------------------- Drift webhook ---------------------
class DriftPayload(BaseModel):
    timestamp: str
    severity: Severity
    drifted_features: Dict[str, List[str]]   # {"numeric": [...], "categorical": [...]}
    psi_values: Optional[Dict[str, float]] = {}
    chi2_values: Optional[Dict[str, float]] = {}
    window_size: int
    reference_run_id: str


class InvestigationCreated(BaseModel):
    investigation_id: str
    status: str = "triaging"


# --------------------- Promotion endpoint ---------------------
class PromoteRequest(BaseModel):
    model_version: str
    investigation_id: str


class PromoteResponse(BaseModel):
    message: str
    new_production_version: str


# --------------------- Error response ---------------------
class ErrorResponse(BaseModel):
    detail: str