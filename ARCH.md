# Architecture

The system consists of four main services, all started with `docker compose up`:

- **Platform** (FastAPI + MLflow + drift)  
  Trains a binary classifier, serves predictions, monitors drift (PSI, chi², output distribution), sends drift alerts to the agent via webhook, and handles programmatic promotion.

- **Agent** (LangGraph supervisor + Redis queue)  
  Receives drift alerts, conducts investigations using three sub‑agents (triage, action, comms), dispatches slow actions (replay, retrain, rollback) through a Redis‑backed queue with idempotency keys and dead‑letter queue, persists state with Postgres checkpoints, and pauses for human approval before promotion.

- **Postgres** – stores LangGraph checkpoints so the agent can resume after crashes.

- **Redis** – job queue with exponential backoff, retries, and dead‑letter queue.

A **Streamlit dashboard** shows the model registry, drift status, agent investigations, queue depth, and the human‑in‑the‑loop (HIL) approval inbox.

## Endpoints & contracts

| Direction          | Endpoint                     | Payload model       |
|--------------------|------------------------------|---------------------|
| Platform → Agent   | `POST /webhook`              | `DriftPayload`      |
| Agent → Platform   | `POST /promote`              | `PromoteRequest`    |
| Platform → Client  | `POST /predict`              | `PredictRequest`    |
| Dashboard reads from both services (registry, drift, investigations, queue) | | |

All API schemas are versioned in `contracts/v1.py`.
