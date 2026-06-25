from __future__ import annotations

import base64
import json
import mimetypes
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status

from image_tagging_api.config import Settings
from image_tagging_api.models import (
    SUPPORTED_PROVIDERS,
    HealthResponse,
    JsonAudioTaggingRequest,
    JsonTaggingRequest,
    TaggingResponse,
)
from image_tagging_api.providers.base import (
    ImageInput,
    MediaInput,
    ProviderRequest,
    Tagger,
    Transcript,
    TranscriptionClient,
    TranscriptTaggingRequest,
)
from image_tagging_api.providers.healthcheck import (
    HttpProviderStartupChecker,
    ProviderCredentialCheck,
    ProviderStartupChecker,
)
from image_tagging_api.providers.transcription import (
    SttTtsTranscriptionClient,
    TranscriptionError,
)
from image_tagging_api.providers.vision import MultiProviderImageTagger

# Upper bound on files per multipart request, mirroring the JSON model's
# ``media`` max_length. Each file fans out to one stt-tts transcription call.
MAX_MEDIA_FILES = 100


def guess_mime_type(filename: str, provided: str | None = None) -> str:
    if provided and provided != "application/octet-stream":
        return provided
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def decode_base64_content(filename: str, content_base64: str) -> bytes:
    try:
        return base64.b64decode(content_base64, validate=True)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 content for {filename}",
        ) from exc


def decode_image_payload(
    filename: str, content_base64: str, mime_type: str | None = None
) -> ImageInput:
    return ImageInput(
        filename=filename,
        content=decode_base64_content(filename, content_base64),
        mime_type=guess_mime_type(filename, mime_type),
    )


async def transcribe_media(
    transcriber: TranscriptionClient, media_inputs: list[MediaInput]
) -> list[Transcript]:
    transcripts: list[Transcript] = []
    for media in media_inputs:
        try:
            transcripts.append(await transcriber.transcribe(media))
        except TranscriptionError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
            ) from exc
    return transcripts


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


def create_app(
    settings: Settings | None = None,
    tagger: Tagger | None = None,
    startup_checker: ProviderStartupChecker | None = None,
    transcription_client: TranscriptionClient | None = None,
) -> FastAPI:
    settings = settings or Settings()
    tagger = tagger or MultiProviderImageTagger(settings)
    startup_checker = startup_checker or HttpProviderStartupChecker()
    transcription_client = transcription_client or SttTtsTranscriptionClient(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        checks = await startup_checker.check_configured_providers(settings)
        app.state.provider_credential_checks = checks
        try:
            yield
        finally:
            aclose = getattr(transcription_client, "aclose", None)
            if aclose is not None:
                await aclose()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Tag one image or a batch of images using OpenAI, Anthropic, Gemini, or Ollama."
        ),
        lifespan=lifespan,
    )

    def get_settings() -> Settings:
        return settings

    def get_tagger() -> Tagger:
        return tagger

    def get_transcription_client() -> TranscriptionClient:
        return transcription_client

    settings_dependency = Depends(get_settings)
    tagger_dependency = Depends(get_tagger)
    transcription_dependency = Depends(get_transcription_client)

    def credential_checks() -> list[ProviderCredentialCheck]:
        return getattr(app.state, "provider_credential_checks", [])

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            providers=SUPPORTED_PROVIDERS,
            credential_checks=[check.model_dump() for check in credential_checks()],
        )

    @app.post("/v1/tag/json", response_model=TaggingResponse)
    async def tag_images_json(
        payload: JsonTaggingRequest,
        app_settings: Settings = settings_dependency,
        image_tagger: Tagger = tagger_dependency,
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
        image_tagger: Tagger = tagger_dependency,
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

    @app.post("/v1/tag/audio", response_model=TaggingResponse)
    async def tag_audio_multipart(
        files: Annotated[list[UploadFile], File(description="One or more audio or video files")],
        app_settings: Settings = settings_dependency,
        image_tagger: Tagger = tagger_dependency,
        transcriber: TranscriptionClient = transcription_dependency,
        provider: Annotated[str, Form()] = "openai",
        model: Annotated[str | None, Form()] = None,
        candidate_tags: Annotated[str | None, Form()] = None,
        max_tags: Annotated[int, Form(ge=1, le=100)] = 10,
        include_explanations: Annotated[bool, Form()] = True,
        language: Annotated[str | None, Form()] = None,
    ) -> TaggingResponse:
        if provider not in SUPPORTED_PROVIDERS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Unsupported provider '{provider}'. "
                    f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
                ),
            )
        if not files:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="At least one audio or video file is required.",
            )
        if len(files) > MAX_MEDIA_FILES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"At most {MAX_MEDIA_FILES} files are allowed per request.",
            )
        media_inputs: list[MediaInput] = []
        used_names: set[str] = set()
        for index, media in enumerate(files):
            # Keep filenames unique so distinct uploads don't collapse into one
            # result when they share a name (or have none).
            name = media.filename or f"media-{index}"
            if name in used_names:
                name = f"{name}-{index}"
            used_names.add(name)
            media_inputs.append(
                MediaInput(
                    filename=name,
                    content=await media.read(),
                    mime_type=guess_mime_type(name, media.content_type),
                    language=language,
                )
            )
        transcripts = await transcribe_media(transcriber, media_inputs)
        request = TranscriptTaggingRequest(
            provider=provider,  # type: ignore[arg-type]
            model=model or app_settings.default_model_for(provider),  # type: ignore[arg-type]
            transcripts=transcripts,
            candidate_tags=parse_candidate_tags(candidate_tags),
            max_tags=max_tags,
            include_explanations=include_explanations,
        )
        return await image_tagger.tag_transcripts(request)

    @app.post("/v1/tag/audio/json", response_model=TaggingResponse)
    async def tag_audio_json(
        payload: JsonAudioTaggingRequest,
        app_settings: Settings = settings_dependency,
        image_tagger: Tagger = tagger_dependency,
        transcriber: TranscriptionClient = transcription_dependency,
    ) -> TaggingResponse:
        model = payload.model or app_settings.default_model_for(payload.provider)
        media_inputs = [
            MediaInput(
                filename=item.filename,
                content=decode_base64_content(item.filename, item.content_base64),
                mime_type=guess_mime_type(item.filename, item.mime_type),
                language=item.language or payload.language,
            )
            for item in payload.media
        ]
        transcripts = await transcribe_media(transcriber, media_inputs)
        request = TranscriptTaggingRequest(
            provider=payload.provider,
            model=model,
            transcripts=transcripts,
            candidate_tags=payload.candidate_tags,
            max_tags=payload.max_tags,
            include_explanations=payload.include_explanations,
        )
        return await image_tagger.tag_transcripts(request)

    return app


app = create_app()


def run() -> None:
    uvicorn.run("image_tagging_api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
