from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/isitlegit"

    # JWT Authentication
    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours

    # Gemini API
    gemini_api_key: str = ""
    use_mock_gemini: bool = False
    gemini_model: str = "gemini-3-pro-preview"
    gemini_max_retries: int = 3
    gemini_timeout_seconds: int = 30
    gemini_cache_ttl_seconds: int = 300

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
