# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import numpy as np

from platform_service.main import app, get_model_manager, get_drift_detector


@pytest.fixture
def fake_model_manager():
    mgr = MagicMock()
    mgr.pipeline.predict_proba.return_value = np.array([[0.6, 0.4]])
    mgr.threshold = 0.35
    mgr.feature_names = [
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "campaign", "previous", "poutcome",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
        "nr.employed", "pdays_was_999"
    ]
    mgr.version = "1"
    return mgr


@pytest.fixture
def fake_drift_detector():
    detector = MagicMock()
    detector.check_and_report.return_value = None   # no drift alert
    return detector


@pytest.fixture
def client(fake_model_manager, fake_drift_detector):
    # Override the actual dependency functions with our mocks
    app.dependency_overrides[get_model_manager] = lambda: fake_model_manager
    app.dependency_overrides[get_drift_detector] = lambda: fake_drift_detector

    with TestClient(app) as test_client:
        yield test_client

    # Clean up overrides to avoid affecting other tests
    app.dependency_overrides.clear()