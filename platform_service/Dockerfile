FROM python:3.12-slim

WORKDIR /app

# System dependencies for scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies (requirements.txt lives in platform_service/)
COPY platform_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code needed at runtime: contracts, platform_service, models
COPY contracts/ contracts/
COPY platform_service/ platform_service/
COPY models/ models/

# Rolling window persistence directory (will be mounted as volume)
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "platform_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
