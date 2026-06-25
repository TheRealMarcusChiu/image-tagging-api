from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from image_tagging_api.models import ProviderName, TaggingResponse


@dataclass(frozen=True)
class ImageInput:
    filename: str
    content: bytes
    mime_type: str


@dataclass(frozen=True)
class ProviderRequest:
    provider: ProviderName
    model: str
    images: list[ImageInput]
    candidate_tags: list[str] | None
    max_tags: int
    include_explanations: bool


class ImageTagger(Protocol):
    async def tag_images(self, request: ProviderRequest) -> TaggingResponse: ...
