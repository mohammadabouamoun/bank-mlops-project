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

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

@lru_cache
def get_settings() -> Settings:
    return Settings()