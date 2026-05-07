import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import requests
import mlflow
from mlflow.tracking import MlflowClient
from contracts.settings import get_settings

settings = get_settings()

PLATFORM_URL = "http://127.0.0.1:8000"
AGENT_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="Bank MLOps Dashboard", layout="wide")
st.title("Drift Triage Co-Pilot – Dashboard")

st.header("Platform Health")
try:
    resp = requests.get(f"{PLATFORM_URL}/health", timeout=5)
    if resp.status_code == 200:
        st.success("Platform is healthy")
    else:
        st.error(f"Platform returned {resp.status_code}")
except Exception as e:
    st.error(f"Cannot reach platform: {e}")

st.header("Model Registry")
try:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    prod_versions = client.get_latest_versions(
        "bank_marketing_classifier",
        stages=["Production"]
    )

    if prod_versions:
        prod = prod_versions[0]
        st.write(f"**Production version:** {prod.version}")
        run = client.get_run(prod.run_id)
        metrics = run.data.metrics

        col1, col2, col3 = st.columns(3)
        col1.metric("Test AUC", metrics.get("test_auc", "N/A"))
        col2.metric("Test Recall", metrics.get("test_recall", "N/A"))
        col3.metric("Threshold", run.data.tags.get("threshold", "N/A"))
    else:
        st.info("No model in Production yet.")
except Exception as e:
    st.warning(f"Model registry not ready yet: {e}")

st.header("Drift Status")
try:
    drift_resp = requests.get(f"{PLATFORM_URL}/drift/latest", timeout=5)
    if drift_resp.status_code == 200:
        drift = drift_resp.json()
        severity = drift.get("severity", "low").upper()

        st.write(f"**Severity:** {severity}")
        st.write(f"Window size: {drift.get('window_size', 0)}")

        with st.expander("PSI values"):
            st.json(drift.get("psi_values", {}))

        with st.expander("Chi² values"):
            st.json(drift.get("chi2_values", {}))
    else:
        st.warning("Could not fetch drift report.")
except Exception as e:
    st.error(f"Drift endpoint error: {e}")

st.header("Trigger Drift Demo")

if st.button("Simulate High Drift"):
    payload = {
        "timestamp": "2026-05-07T12:00:00",
        "severity": "high",
        "drifted_features": {
            "numeric": ["age", "balance"],
            "categorical": ["job"]
        },
        "psi_values": {
            "age": 0.42,
            "balance": 0.31
        },
        "chi2_values": {
            "job": 18.5
        },
        "window_size": 200,
        "reference_run_id": "run_001"
    }

    try:
        resp = requests.post(
            f"{AGENT_URL}/webhooks/drift",
            json=payload,
            timeout=5
        )

        if resp.status_code == 200:
            st.success("High drift alert sent to agent.")
            st.json(resp.json())
        else:
            st.error(f"Failed to send drift alert: {resp.status_code}")
            st.text(resp.text)

    except Exception as e:
        st.error(f"Could not reach agent: {e}")

st.header("Agent Investigations / HIL Inbox")

try:
    approvals_resp = requests.get(f"{AGENT_URL}/approvals/pending", timeout=5)

    if approvals_resp.status_code == 200:
        approvals = approvals_resp.json().get("pending_approvals", [])

        if approvals:
            for approval in approvals:
                st.write(f"**Approval ID:** {approval['id']}")
                st.write(f"Action: {approval['action']}")
                st.write(f"Severity: {approval['severity']}")
                st.write(f"Status: {approval['status']}")

                if approval["status"] == "pending":
                    if st.button(f"Approve {approval['id']}"):
                        approve_resp = requests.post(
                            f"{AGENT_URL}/approvals/{approval['id']}/approve",
                            timeout=5
                        )

                        if approve_resp.status_code == 200:
                            st.success("Approved and queued successfully")
                            st.json(approve_resp.json())
                        else:
                            st.error("Failed to approve")
        else:
            st.info("No pending approvals.")
    else:
        st.warning("Could not fetch approvals from agent.")
except Exception as e:
    st.error(f"Agent approvals error: {e}")

st.header("Queue & DLQ")

try:
    queue_resp = requests.get(f"{AGENT_URL}/queue/status", timeout=5)

    if queue_resp.status_code == 200:
        queue_data = queue_resp.json()

        col1, col2 = st.columns(2)
        col1.metric("Queue Depth", queue_data.get("queue_depth", 0))
        col2.metric("DLQ Depth", queue_data.get("dlq_depth", 0))

        with st.expander("Queued Jobs"):
            st.json(queue_data.get("queued_jobs", []))

        with st.expander("Dead Letter Queue"):
            st.json(queue_data.get("dlq_jobs", []))
    else:
        st.warning("Could not fetch queue status.")
except Exception as e:
    st.error(f"Queue endpoint error: {e}")