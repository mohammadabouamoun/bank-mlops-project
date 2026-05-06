import streamlit as st
import requests
import mlflow
from mlflow.tracking import MlflowClient
from contracts.settings import get_settings

settings = get_settings()

st.set_page_config(page_title="Bank MLOps Dashboard", layout="wide")
st.title("Drift Triage Co‑Pilot – Dashboard")

# ------------------------------------------------------------------
# Platform Health
# ------------------------------------------------------------------
st.header("Platform Health")
try:
    resp = requests.get("http://platform:8000/health", timeout=5)
    if resp.status_code == 200:
        st.success("Platform is healthy")
    else:
        st.error(f"Platform returned {resp.status_code}")
except Exception as e:
    st.error(f"Cannot reach platform: {e}")

# ------------------------------------------------------------------
# Model Registry
# ------------------------------------------------------------------
st.header("Model Registry")
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
client = MlflowClient()
try:
    prod_versions = client.get_latest_versions("bank_marketing_classifier", stages=["Production"])
    if prod_versions:
        prod = prod_versions[0]
        st.write(f"**Production version:** {prod.version}")
        run = client.get_run(prod.run_id)
        metrics = run.data.metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Test AUC", f"{metrics.get('test_auc', 'N/A'):.4f}")
        col2.metric("Test Recall", f"{metrics.get('test_recall', 'N/A'):.4f}")
        col3.metric("Threshold", run.data.tags.get("threshold", "N/A"))
    else:
        st.info("No model in Production yet.")
except Exception as e:
    st.error(f"Failed to load registry: {e}")

# ------------------------------------------------------------------
# Drift Status (placeholder)
# ------------------------------------------------------------------
st.header("Drift Status")
st.info("Drift monitoring data will be displayed here once the agent is integrated.")

# ------------------------------------------------------------------
# Agent Investigations (placeholder – partner will implement)
# ------------------------------------------------------------------
st.header("Agent Investigations")
st.info("Investigation list coming soon – partner's agent module required.")

# ------------------------------------------------------------------
# Queue & HIL (placeholder)
# ------------------------------------------------------------------
st.header("Queue & HIL")
st.info("Queue depth and Human‑in‑the‑Loop approvals will appear here.")