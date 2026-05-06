## Prerequisites
- Docker & Docker Compose
- Python 3.12+ and `uv` (for local development)

## Quick start (all services with Docker)

1. Clone the repository.
2. Copy `.env.example` to `.env` and fill in:
   - `LLM_API_KEY` – your Groq API key (dummy value is fine for the platform alone)
   - `PROMOTION_API_KEY` – shared secret (same value for both platform and agent)
3. Start the platform:
   ```bash
   docker compose up --build platform
The platform will be available at http://localhost:8000.
4. (When the agent is ready) start the full stack:

bash
docker compose --profile agent up --build
Local development (without Docker)
Create a virtual environment:

bash
uv venv
source .venv/bin/activate
Install platform dependencies:

bash
uv pip install -r platform_service/requirements.txt
Train the model:

bash
python platform_service/train.py
Start the server:

bash
uvicorn platform_service.main:app --reload
Start the dashboard (optional):

bash
streamlit run dashboard/app.py --server.port 8501
Running tests
bash
pytest tests/ -v
CI
GitHub Actions runs pytest and ruff on every push to main or feat/* branches.

