from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProviderName = Literal["openai", "anthropic", "gemini", "ollama"]
SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini", "ollama")


class ImagePayload(BaseModel):
    filename: str = Field(min_length=1)
    content_base64: str = Field(min_length=1)
    mime_type: str | None = None


class JsonTaggingRequest(BaseModel):
    provider: ProviderName = "openai"
    model: str | None = None
    candidate_tags: list[str] | None = None
    max_tags: int = Field(default=10, ge=1, le=100)
    include_explanations: bool = True
    images: list[ImagePayload] = Field(min_length=1, max_length=100)

    @field_validator("candidate_tags")
    @classmethod
    def normalize_candidate_tags(cls, tags: list[str] | None) -> list[str] | None:
        if tags is None:
            return None
        cleaned = [tag.strip() for tag in tags if tag.strip()]
        return cleaned or None


class ImageTagResult(BaseModel):
    filename: str
    tags: list[str]
    confidence: float | None = Field(default=None, ge=0, le=1)
    explanation: str | None = None
    raw_response: dict | None = None


class TaggingResponse(BaseModel):
    provider: ProviderName
    model: str
    results: list[ImageTagResult]


class HealthResponse(BaseModel):
    status: str = "ok"
    providers: tuple[str, ...]
    credential_checks: list[dict] = Field(default_factory=list)


class ModelTagOutput(BaseModel):
    """Validated shape requested from vision providers."""

    model_config = ConfigDict(extra="ignore")

    images: list[ImageTagResult]
