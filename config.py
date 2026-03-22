"""Application settings loaded from environment and optional `.env` file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    default_llm_provider: str = Field(
        default="ollama",
        validation_alias="DEFAULT_LLM_PROVIDER",
    )
    default_ollama_model: str = Field(
        default="llama3.2",
        validation_alias="DEFAULT_OLLAMA_MODEL",
    )
    default_gemini_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias="DEFAULT_GEMINI_MODEL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
