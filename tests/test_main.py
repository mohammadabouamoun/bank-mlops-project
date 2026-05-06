# tests/test_main.py

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_predict_valid_input(client):
    features = [
        56, "housemaid", "married", "basic.4y", "no", "no", "no",
        "telephone", "may", "mon", 1, 0, "nonexistent",
        1.1, 93.994, -36.4, 4.857, 5191.0, 1
    ]
    resp = client.post("/predict", json={"features": features})
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 1     # fake model returns proba 0.4 >= 0.35
    assert 0 < data["probability"] < 1
    assert data["model_version"] == "1"

def test_predict_invalid_length(client):
    # Too few features
    resp = client.post("/predict", json={"features": [1, 2, 3]})
    assert resp.status_code == 422
    # The detail should mention "Expected 19 features"
    assert "Expected 19 features" in resp.json()["detail"]

def test_promote_unauthorized(client):
    resp = client.post("/promote", json={
        "model_version": "4", "investigation_id": "test-123"
    })
    assert resp.status_code == 401
    assert "Unauthorized" in resp.json()["detail"]

def test_promote_with_mocked_mlflow(client, monkeypatch):
    """Mock MLflow so all checks pass and the promotion succeeds."""
    import mlflow
    import mlflow.tracking
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.run_id = "run123"
    mock_run = MagicMock()
    mock_run.data.metrics = {"test_recall": 0.76, "test_auc": 0.82}
    mock_run.data.tags = {"threshold": "0.3527"}
    mock_client.get_model_version.return_value = mock_mv
    mock_client.get_run.return_value = mock_run
    mock_client.get_latest_versions.return_value = []

    # Patch MlflowClient and set_tracking_uri
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda *a, **kw: mock_client)
    monkeypatch.setattr(mlflow, "set_tracking_uri", MagicMock())

    resp = client.post(
        "/promote",
        json={"model_version": "4", "investigation_id": "test-123"},
        headers={"Authorization": "Bearer dev-secret-123"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "promoted to Production" in data["message"]
    assert data["new_production_version"] == "4"