from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "Image Tagging API"
    default_provider: str = "openai"
    default_openai_model: str = "gpt-4.1-mini"
    default_anthropic_model: str = "claude-3-5-sonnet-latest"
    default_gemini_model: str = "gemini-2.0-flash"
    default_ollama_model: str = "llava:latest"

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    request_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    def default_model_for(self, provider: str) -> str:
        defaults = {
            "openai": self.default_openai_model,
            "anthropic": self.default_anthropic_model,
            "gemini": self.default_gemini_model,
            "ollama": self.default_ollama_model,
        }
        return defaults[provider]
