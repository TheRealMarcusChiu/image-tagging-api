from __future__ import annotations

import httpx

from image_tagging_api.config import Settings
from image_tagging_api.providers.base import MediaInput, Transcript


class TranscriptionError(RuntimeError):
    """Raised when the external stt-tts service fails to produce a transcript."""


class SttTtsTranscriptionClient:
    """Transcribes audio/video by calling a self-hosted stt-tts server.

    Posts each file to the server's OpenAI-compatible
    ``POST /v1/audio/transcriptions`` endpoint and returns the transcript.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        # Only clients we create ourselves are ours to close; an injected client
        # (e.g. a test transport) is owned by the caller.
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # httpx.verify accepts a CA bundle path or a bool. A self-signed
            # .lan certificate needs either a CA path or verify=False.
            verify: str | bool = self.settings.stt_tts_ca_cert or self.settings.stt_tts_verify_ssl
            self._client = httpx.AsyncClient(
                timeout=self.settings.stt_request_timeout_seconds,
                verify=verify,
            )
        return self._client

    async def transcribe(self, media: MediaInput) -> Transcript:
        url = f"{self.settings.stt_tts_base_url.rstrip('/')}/v1/audio/transcriptions"
        data: dict[str, str] = {
            "model": self.settings.stt_model,
            "response_format": "verbose_json",
        }
        language = media.language or self.settings.stt_language
        if language:
            data["language"] = language
        if self.settings.stt_vad_filter is not None:
            data["vad_filter"] = "true" if self.settings.stt_vad_filter else "false"

        files = {"file": (media.filename, media.content, media.mime_type)}
        try:
            response = await self._ensure_client().post(url, data=data, files=files)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TranscriptionError(
                f"Transcription service returned HTTP {exc.response.status_code} for "
                f"{media.filename}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise TranscriptionError(
                f"Transcription request failed for {media.filename}: {message}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise TranscriptionError(
                f"Transcription service returned a non-JSON response for {media.filename}."
            ) from exc

        if not isinstance(payload, dict):
            raise TranscriptionError(
                f"Transcription service returned an unexpected JSON shape for {media.filename}."
            )

        return Transcript(
            filename=media.filename,
            text=(payload.get("text") or "").strip(),
            language=payload.get("language"),
            duration=payload.get("duration"),
        )
