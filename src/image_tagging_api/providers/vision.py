from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import HTTPException, status
from pydantic import ValidationError

from image_tagging_api.config import Settings
from image_tagging_api.models import ImageTagResult, ModelTagOutput, ProviderName, TaggingResponse
from image_tagging_api.providers.base import (
    ImageInput,
    ImageTagger,
    ProviderRequest,
    TranscriptTaggingRequest,
)


class ProviderError(RuntimeError):
    pass


def build_prompt(request: ProviderRequest) -> str:
    tag_instructions = (
        "Choose only from this candidate tag list: " + ", ".join(request.candidate_tags)
        if request.candidate_tags
        else "Create concise, searchable tags for the visible content."
    )
    explanations = (
        "Include a short explanation for each image."
        if request.include_explanations
        else "Use null for explanation."
    )
    filenames = ", ".join(image.filename for image in request.images)
    return f"""
You are an image tagging service. Tag each input image by visible content only.
Images: {filenames}
{tag_instructions}
Return at most {request.max_tags} tags per image.
Write multi-word tags with spaces (e.g. "wedding vows"), not underscores or camelCase.
{explanations}
Return strictly valid JSON with this shape:
{{"images":[{{"filename":"example.jpg","tags":["tag1"],"confidence":0.0,"explanation":"..."}}]}}
""".strip()


def build_transcript_prompt(request: TranscriptTaggingRequest) -> str:
    tag_instructions = (
        "Choose only from this candidate tag list: " + ", ".join(request.candidate_tags)
        if request.candidate_tags
        else "Create concise, searchable tags that describe the spoken content."
    )
    explanations = (
        "Include a short explanation for each item."
        if request.include_explanations
        else "Use null for explanation."
    )
    sections = "\n\n".join(
        f'Transcript for "{transcript.filename}":\n{transcript.text or "(no speech detected)"}'
        for transcript in request.transcripts
    )
    return f"""
You are a media tagging service. Tag each item using only the spoken content in its transcript.
{tag_instructions}
Return at most {request.max_tags} tags per item.
Write multi-word tags with spaces (e.g. "wedding vows"), not underscores or camelCase.
{explanations}
Return strictly valid JSON with this shape:
{{"images":[{{"filename":"example.mp4","tags":["tag1"],"confidence":0.0,"explanation":"..."}}]}}

{sections}
""".strip()


def normalize_tag(tag: str) -> str:
    """Render a tag as a human-readable phrase: underscores become spaces and
    surrounding/duplicate whitespace is collapsed (e.g. ``wedding_vows`` ->
    ``wedding vows``)."""
    return " ".join(tag.replace("_", " ").split())


def image_data_url(image: ImageInput) -> str:
    return f"data:{image.mime_type};base64,{base64.b64encode(image.content).decode()}"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            excerpt = stripped.replace("\n", " ")[:200].strip()
            suffix = "..." if len(stripped.replace("\n", " ").strip()) > 200 else ""
            raise ProviderError(
                "Provider did not return JSON. "
                f"Response excerpt: {excerpt}{suffix}"
            ) from None
        return json.loads(stripped[start : end + 1])


def _assemble_tagging_response(
    text: str,
    *,
    provider: ProviderName,
    model: str,
    filenames: list[str],
    max_tags: int,
) -> TaggingResponse:
    try:
        output = ModelTagOutput.model_validate(extract_json_object(text))
    except (json.JSONDecodeError, ValidationError, ProviderError) as exc:
        raise ProviderError(f"Could not parse provider response: {exc}") from exc

    by_filename = {result.filename: result for result in output.images}
    results: list[ImageTagResult] = []
    for filename in filenames:
        result = by_filename.get(filename)
        if result is None:
            result = ImageTagResult(
                filename=filename,
                tags=[],
                confidence=None,
                explanation="Provider response did not include this filename.",
            )
        result.tags = [
            cleaned for tag in result.tags[:max_tags] if (cleaned := normalize_tag(tag))
        ]
        results.append(result)
    return TaggingResponse(provider=provider, model=model, results=results)


def parse_model_output(text: str, request: ProviderRequest) -> TaggingResponse:
    return _assemble_tagging_response(
        text,
        provider=request.provider,
        model=request.model,
        filenames=[image.filename for image in request.images],
        max_tags=request.max_tags,
    )


class VisionProvider(ABC):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    @abstractmethod
    async def complete(self, model: str, prompt: str, images: list[ImageInput]) -> str:
        """Send a prompt (optionally with images) to the model and return raw text."""
        raise NotImplementedError

    def require_key(self, key: str | None, env_name: str) -> str:
        if not key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{env_name} is required for this provider.",
            )
        return key


class OpenAIProvider(VisionProvider):
    async def complete(self, model: str, prompt: str, images: list[ImageInput]) -> str:
        key = self.require_key(self.settings.openai_api_key, "OPENAI_API_KEY")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicProvider(VisionProvider):
    async def complete(self, model: str, prompt: str, images: list[ImageInput]) -> str:
        auth_headers = self.settings.anthropic_auth_headers()
        if not auth_headers:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or "
                    "CLAUDE_CODE_OAUTH_TOKEN is required for this provider."
                ),
            )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.mime_type,
                        "data": base64.b64encode(image.content).decode(),
                    },
                }
            )
        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={**auth_headers, "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": 2048,
                "temperature": 0,
                "messages": [{"role": "user", "content": content}],
            },
        )
        response.raise_for_status()
        return "".join(part.get("text", "") for part in response.json().get("content", []))


class GeminiProvider(VisionProvider):
    async def complete(self, model: str, prompt: str, images: list[ImageInput]) -> str:
        key = self.require_key(self.settings.gemini_api_key, "GEMINI_API_KEY")
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for image in images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": image.mime_type,
                        "data": base64.b64encode(image.content).decode(),
                    }
                }
            )
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}"
        )
        response = await self.client.post(url, json={"contents": [{"parts": parts}]})
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


class OllamaProvider(VisionProvider):
    @staticmethod
    def _payload_excerpt(payload: Any) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return serialized[:500] + ("..." if len(serialized) > 500 else "")

    @staticmethod
    def _extract_text_payload(payload: dict[str, Any]) -> str:
        for field in ("response", "thinking"):
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return value
        raise ProviderError(
            "Ollama returned an empty response field. "
            f"Payload excerpt: {OllamaProvider._payload_excerpt(payload)}"
        )

    async def complete(self, model: str, prompt: str, images: list[ImageInput]) -> str:
        response = await self.client.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [base64.b64encode(image.content).decode() for image in images],
                "format": "json",
                "think": False,
                "options": {"temperature": 0},
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return self._extract_text_payload(payload)


class MultiProviderImageTagger(ImageTagger):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.providers: dict[ProviderName, VisionProvider] = {
            "openai": OpenAIProvider(settings, client),
            "anthropic": AnthropicProvider(settings, client),
            "gemini": GeminiProvider(settings, client),
            "ollama": OllamaProvider(settings, client),
        }

    @staticmethod
    def _describe_http_error(exc: httpx.HTTPError) -> str:
        message = str(exc).strip()
        return message or exc.__class__.__name__

    async def _dispatch(
        self, provider: ProviderName, work: Callable[[], Awaitable[TaggingResponse]]
    ) -> TaggingResponse:
        try:
            return await work()
        except HTTPException:
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                retry_after = exc.response.headers.get("retry-after")
                retry_hint = f"; retry after {retry_after} seconds" if retry_after else ""
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Provider {provider} rate limited the request{retry_hint}: "
                        f"{exc.response.text}"
                    ),
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Provider {provider} returned HTTP {exc.response.status_code}: "
                    f"{exc.response.text}"
                ),
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Provider {provider} request failed: "
                    f"{self._describe_http_error(exc)}"
                ),
            ) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    async def tag_images(self, request: ProviderRequest) -> TaggingResponse:
        async def work() -> TaggingResponse:
            text = await self.providers[request.provider].complete(
                request.model, build_prompt(request), request.images
            )
            return parse_model_output(text, request)

        return await self._dispatch(request.provider, work)

    async def tag_transcripts(self, request: TranscriptTaggingRequest) -> TaggingResponse:
        if not any(transcript.text.strip() for transcript in request.transcripts):
            # Nothing was transcribed (e.g. silent audio or VAD dropped everything);
            # skip the LLM call and return empty tags for each item.
            return TaggingResponse(
                provider=request.provider,
                model=request.model,
                results=[
                    ImageTagResult(
                        filename=transcript.filename,
                        tags=[],
                        confidence=None,
                        explanation="No speech detected.",
                    )
                    for transcript in request.transcripts
                ],
            )

        async def work() -> TaggingResponse:
            text = await self.providers[request.provider].complete(
                request.model, build_transcript_prompt(request), []
            )
            return _assemble_tagging_response(
                text,
                provider=request.provider,
                model=request.model,
                filenames=[transcript.filename for transcript in request.transcripts],
                max_tags=request.max_tags,
            )

        return await self._dispatch(request.provider, work)
