from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from typing import Any

import httpx
from fastapi import HTTPException, status
from pydantic import ValidationError

from image_tagging_api.config import Settings
from image_tagging_api.models import ImageTagResult, ModelTagOutput, ProviderName, TaggingResponse
from image_tagging_api.providers.base import ImageInput, ImageTagger, ProviderRequest


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
{explanations}
Return strictly valid JSON with this shape:
{{"images":[{{"filename":"example.jpg","tags":["tag1"],"confidence":0.0,"explanation":"..."}}]}}
""".strip()


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
            raise ProviderError("Provider did not return JSON") from None
        return json.loads(stripped[start : end + 1])


def parse_model_output(text: str, request: ProviderRequest) -> TaggingResponse:
    try:
        output = ModelTagOutput.model_validate(extract_json_object(text))
    except (json.JSONDecodeError, ValidationError, ProviderError) as exc:
        raise ProviderError(f"Could not parse provider response: {exc}") from exc

    by_filename = {result.filename: result for result in output.images}
    results: list[ImageTagResult] = []
    for image in request.images:
        result = by_filename.get(image.filename)
        if result is None:
            result = ImageTagResult(
                filename=image.filename,
                tags=[],
                confidence=None,
                explanation="Provider response did not include this filename.",
            )
        if len(result.tags) > request.max_tags:
            result.tags = result.tags[: request.max_tags]
        results.append(result)
    return TaggingResponse(provider=request.provider, model=request.model, results=results)


class VisionProvider(ABC):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    @abstractmethod
    async def tag_images(self, request: ProviderRequest) -> TaggingResponse:
        raise NotImplementedError

    def require_key(self, key: str | None, env_name: str) -> str:
        if not key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{env_name} is required for this provider.",
            )
        return key


class OpenAIProvider(VisionProvider):
    async def tag_images(self, request: ProviderRequest) -> TaggingResponse:
        key = self.require_key(self.settings.openai_api_key, "OPENAI_API_KEY")
        content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(request)}]
        for image in request.images:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": request.model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return parse_model_output(text, request)


class AnthropicProvider(VisionProvider):
    async def tag_images(self, request: ProviderRequest) -> TaggingResponse:
        auth_headers = self.settings.anthropic_auth_headers()
        if not auth_headers:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or "
                    "CLAUDE_CODE_OAUTH_TOKEN is required for this provider."
                ),
            )
        content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(request)}]
        for image in request.images:
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
                "model": request.model,
                "max_tokens": 2048,
                "temperature": 0,
                "messages": [{"role": "user", "content": content}],
            },
        )
        response.raise_for_status()
        text = "".join(part.get("text", "") for part in response.json().get("content", []))
        return parse_model_output(text, request)


class GeminiProvider(VisionProvider):
    async def tag_images(self, request: ProviderRequest) -> TaggingResponse:
        key = self.require_key(self.settings.gemini_api_key, "GEMINI_API_KEY")
        parts: list[dict[str, Any]] = [{"text": build_prompt(request)}]
        for image in request.images:
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
            f"{request.model}:generateContent?key={key}"
        )
        response = await self.client.post(url, json={"contents": [{"parts": parts}]})
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return parse_model_output(text, request)


class OllamaProvider(VisionProvider):
    async def tag_images(self, request: ProviderRequest) -> TaggingResponse:
        response = await self.client.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": request.model,
                "prompt": build_prompt(request),
                "images": [base64.b64encode(image.content).decode() for image in request.images],
                "format": "json",
                "stream": False,
            },
        )
        response.raise_for_status()
        return parse_model_output(response.json()["response"], request)


class MultiProviderImageTagger(ImageTagger):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.providers: dict[ProviderName, VisionProvider] = {
            "openai": OpenAIProvider(settings, client),
            "anthropic": AnthropicProvider(settings, client),
            "gemini": GeminiProvider(settings, client),
            "ollama": OllamaProvider(settings, client),
        }

    async def tag_images(self, request: ProviderRequest) -> TaggingResponse:
        try:
            return await self.providers[request.provider].tag_images(request)
        except HTTPException:
            raise
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Provider {request.provider} returned HTTP {exc.response.status_code}: "
                    f"{exc.response.text}"
                ),
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Provider {request.provider} request failed: {exc}",
            ) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
