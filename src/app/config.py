"""Configuration settings for the submission grader."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API settings
    app_name: str = "Submission Grader API"
    app_version: str = "1.0.0"
    debug: bool = False

    # LLM settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Rate limiting
    max_submission_length: int = 50000  # characters

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore unknown environment variables


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
