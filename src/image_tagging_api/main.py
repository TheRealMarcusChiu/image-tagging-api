from __future__ import annotations

import base64
import json
import mimetypes
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status

from image_tagging_api.config import Settings
from image_tagging_api.models import (
    SUPPORTED_PROVIDERS,
    HealthResponse,
    JsonTaggingRequest,
    TaggingResponse,
)
from image_tagging_api.providers.base import ImageInput, ImageTagger, ProviderRequest
from image_tagging_api.providers.vision import MultiProviderImageTagger


def guess_mime_type(filename: str, provided: str | None = None) -> str:
    if provided and provided != "application/octet-stream":
        return provided
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def decode_image_payload(
    filename: str, content_base64: str, mime_type: str | None = None
) -> ImageInput:
    try:
        content = base64.b64decode(content_base64, validate=True)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 content for {filename}",
        ) from exc
    return ImageInput(
        filename=filename,
        content=content,
        mime_type=guess_mime_type(filename, mime_type),
    )


def parse_candidate_tags(candidate_tags: str | None) -> list[str] | None:
    if not candidate_tags:
        return None
    text = candidate_tags.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(",")]
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="candidate_tags must be a JSON array or comma-separated string.",
        )
    cleaned = [str(tag).strip() for tag in parsed if str(tag).strip()]
    return cleaned or None


def create_app(settings: Settings | None = None, tagger: ImageTagger | None = None) -> FastAPI:
    settings = settings or Settings()
    tagger = tagger or MultiProviderImageTagger(settings)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Tag one image or a batch of images using OpenAI, Anthropic, Gemini, or Ollama."
        ),
    )

    def get_settings() -> Settings:
        return settings

    def get_tagger() -> ImageTagger:
        return tagger

    settings_dependency = Depends(get_settings)
    tagger_dependency = Depends(get_tagger)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(providers=SUPPORTED_PROVIDERS)

    @app.post("/v1/tag/json", response_model=TaggingResponse)
    async def tag_images_json(
        payload: JsonTaggingRequest,
        app_settings: Settings = settings_dependency,
        image_tagger: ImageTagger = tagger_dependency,
    ) -> TaggingResponse:
        model = payload.model or app_settings.default_model_for(payload.provider)
        images = [
            decode_image_payload(image.filename, image.content_base64, image.mime_type)
            for image in payload.images
        ]
        request = ProviderRequest(
            provider=payload.provider,
            model=model,
            images=images,
            candidate_tags=payload.candidate_tags,
            max_tags=payload.max_tags,
            include_explanations=payload.include_explanations,
        )
        return await image_tagger.tag_images(request)

    @app.post("/v1/tag", response_model=TaggingResponse)
    async def tag_images_multipart(
        images: Annotated[list[UploadFile], File(description="One or more image files")],
        app_settings: Settings = settings_dependency,
        image_tagger: ImageTagger = tagger_dependency,
        provider: Annotated[str, Form()] = "openai",
        model: Annotated[str | None, Form()] = None,
        candidate_tags: Annotated[str | None, Form()] = None,
        max_tags: Annotated[int, Form(ge=1, le=100)] = 10,
        include_explanations: Annotated[bool, Form()] = True,
    ) -> TaggingResponse:
        if provider not in SUPPORTED_PROVIDERS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Unsupported provider '{provider}'. "
                    f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
                ),
            )
        if not images:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="At least one image is required.",
            )
        image_inputs = [
            ImageInput(
                filename=image.filename or "image",
                content=await image.read(),
                mime_type=guess_mime_type(image.filename or "image", image.content_type),
            )
            for image in images
        ]
        request = ProviderRequest(
            provider=provider,  # type: ignore[arg-type]
            model=model or app_settings.default_model_for(provider),  # type: ignore[arg-type]
            images=image_inputs,
            candidate_tags=parse_candidate_tags(candidate_tags),
            max_tags=max_tags,
            include_explanations=include_explanations,
        )
        return await image_tagger.tag_images(request)

    return app


app = create_app()


def run() -> None:
    uvicorn.run("image_tagging_api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
