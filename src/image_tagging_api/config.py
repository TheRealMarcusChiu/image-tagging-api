from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEPRECATED_ANTHROPIC_MODEL_ALIASES = {
    "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
}


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "Image Tagging API"
    default_provider: str = "openai"
    default_openai_model: str = "gpt-4.1-mini"
    default_anthropic_model: str = "claude-sonnet-4-6"
    default_gemini_model: str = "gemini-2.0-flash"
    default_ollama_model: str = "llava:latest"

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_auth_token: str | None = Field(default=None, alias="ANTHROPIC_AUTH_TOKEN")
    claude_code_oauth_token: str | None = Field(default=None, alias="CLAUDE_CODE_OAUTH_TOKEN")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    request_timeout_seconds: float = 60.0
    validate_provider_credentials_on_startup: bool = True
    startup_check_timeout_seconds: float = 10.0
    anthropic_startup_check_model: str = Field(
        default="claude-sonnet-4-6", alias="ANTHROPIC_STARTUP_CHECK_MODEL"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    def default_model_for(self, provider: str) -> str:
        defaults = {
            "openai": self.default_openai_model,
            "anthropic": DEPRECATED_ANTHROPIC_MODEL_ALIASES.get(
                self.default_anthropic_model, self.default_anthropic_model
            ),
            "gemini": self.default_gemini_model,
            "ollama": self.default_ollama_model,
        }
        return defaults[provider]

    def anthropic_auth_headers(self) -> dict[str, str]:
        """Build Anthropic auth headers without exposing credential values."""
        bearer_token = self.anthropic_auth_token or self.claude_code_oauth_token
        if bearer_token:
            return {"Authorization": f"Bearer {bearer_token}"}
        if self.anthropic_api_key:
            return {"x-api-key": self.anthropic_api_key}
        return {}
