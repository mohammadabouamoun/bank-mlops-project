from pydantic import BaseModel, Field
from typing import Optional, List, Dict ,Union
from enum import Enum


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class PredictRequest(BaseModel):
    features: List[Union[float, str]] = Field(
        ..., description="Raw feature values (numbers and strings) for one sample"
    )

class PredictResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    model_version: Optional[str] = None


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


class PromoteRequest(BaseModel):
    model_version: str
    investigation_id: str


class PromoteResponse(BaseModel):
    message: str
    new_production_version: str


class ErrorResponse(BaseModel):
    detail: str