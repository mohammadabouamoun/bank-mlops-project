FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (scikit-learn needs gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy and install Python dependencies using uv
COPY platform_service/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code and models
COPY contracts/ contracts/
COPY platform_service/ platform_service/
COPY models/ models/

# Rolling window persistence directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "platform_service.main:app", "--host", "0.0.0.0", "--port", "8000"]