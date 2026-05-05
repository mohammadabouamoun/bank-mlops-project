# contracts/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    # LLM
    llm_provider: str = "groq"
    llm_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"

    # DB
    postgres_user: str = "agent"
    postgres_password: str = "agentpass"
    postgres_db: str = "agentdb"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # Redis
    redis_password: str = "redispass"
    redis_host: str = "redis"
    redis_port: int = 6379

    # Promotion secret
    promotion_api_key: str

    # Agent webhook URL (where drift alerts go)
    agent_webhook_url: str = "http://127.0.0.1:8001/webhook"

    # Promotion checklist thresholds
    promotion_min_recall: float = 0.75
    promotion_min_auc: float = 0.70 

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

@lru_cache
def get_settings() -> Settings:
    return Settings()