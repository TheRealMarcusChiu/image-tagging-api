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


@dataclass(frozen=True)
class MediaInput:
    filename: str
    content: bytes
    mime_type: str
    language: str | None = None


@dataclass(frozen=True)
class Transcript:
    filename: str
    text: str
    language: str | None = None
    duration: float | None = None


@dataclass(frozen=True)
class TranscriptTaggingRequest:
    provider: ProviderName
    model: str
    transcripts: list[Transcript]
    candidate_tags: list[str] | None
    max_tags: int
    include_explanations: bool


class TranscriptionClient(Protocol):
    async def transcribe(self, media: MediaInput) -> Transcript: ...


class TranscriptTagger(Protocol):
    async def tag_transcripts(self, request: TranscriptTaggingRequest) -> TaggingResponse: ...


class Tagger(ImageTagger, TranscriptTagger, Protocol):
    """A tagger that handles both image and transcript requests."""
