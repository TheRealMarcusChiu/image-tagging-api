from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from image_tagging_api.config import Settings
from image_tagging_api.main import create_app
from image_tagging_api.models import ImageTagResult, TaggingResponse
from image_tagging_api.providers.base import MediaInput, Transcript, TranscriptTaggingRequest
from image_tagging_api.providers.transcription import TranscriptionError

MEDIA_BYTES = b"fake-media-bytes"


class RecordingTranscriptionClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[MediaInput] = []
        self.fail = fail

    async def transcribe(self, media: MediaInput) -> Transcript:
        self.calls.append(media)
        if self.fail:
            raise TranscriptionError(f"stt-tts unavailable for {media.filename}")
        return Transcript(
            filename=media.filename,
            text=f"transcript of {media.filename}",
            language="en",
            duration=1.5,
        )


class RecordingTranscriptTagger:
    def __init__(self) -> None:
        self.calls: list[TranscriptTaggingRequest] = []

    async def tag_transcripts(self, request: TranscriptTaggingRequest) -> TaggingResponse:
        self.calls.append(request)
        return TaggingResponse(
            provider=request.provider,
            model=request.model,
            results=[
                ImageTagResult(
                    filename=transcript.filename,
                    tags=["meeting", "budget"],
                    confidence=0.8,
                    explanation="Discusses the budget.",
                )
                for transcript in request.transcripts
            ],
        )


def build_client(
    tagger: RecordingTranscriptTagger | None = None,
    transcriber: RecordingTranscriptionClient | None = None,
    settings: Settings | None = None,
) -> TestClient:
    app = create_app(
        settings=settings or Settings(),
        tagger=tagger or RecordingTranscriptTagger(),
        transcription_client=transcriber or RecordingTranscriptionClient(),
    )
    return TestClient(app)


def test_multipart_audio_transcribes_then_tags_and_forwards_choices():
    tagger = RecordingTranscriptTagger()
    transcriber = RecordingTranscriptionClient()
    client = build_client(tagger, transcriber)

    response = client.post(
        "/v1/tag/audio",
        data={
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "candidate_tags": '["meeting", "budget"]',
            "max_tags": "3",
            "include_explanations": "true",
            "language": "en",
        },
        files=[
            ("files", ("meeting.mp3", MEDIA_BYTES, "audio/mpeg")),
            ("files", ("call.mp4", MEDIA_BYTES, "video/mp4")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "anthropic"
    assert body["model"] == "claude-sonnet-4-6"
    assert [result["filename"] for result in body["results"]] == ["meeting.mp3", "call.mp4"]
    assert body["results"][0]["tags"] == ["meeting", "budget"]

    # Every media file was sent to the transcription service with the language hint.
    assert [media.filename for media in transcriber.calls] == ["meeting.mp3", "call.mp4"]
    assert all(media.language == "en" for media in transcriber.calls)

    forwarded = tagger.calls[0]
    assert forwarded.provider == "anthropic"
    assert forwarded.model == "claude-sonnet-4-6"
    assert forwarded.candidate_tags == ["meeting", "budget"]
    assert forwarded.max_tags == 3
    assert forwarded.transcripts[0].text == "transcript of meeting.mp3"


def test_json_audio_accepts_base64_media():
    tagger = RecordingTranscriptTagger()
    transcriber = RecordingTranscriptionClient()
    client = build_client(tagger, transcriber)

    response = client.post(
        "/v1/tag/audio/json",
        json={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "candidate_tags": ["call", "support"],
            "media": [
                {
                    "filename": "voicemail.m4a",
                    "content_base64": base64.b64encode(MEDIA_BYTES).decode(),
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["filename"] == "voicemail.m4a"
    assert transcriber.calls[0].filename == "voicemail.m4a"
    assert tagger.calls[0].candidate_tags == ["call", "support"]


def test_multipart_audio_rejects_unsupported_provider():
    client = build_client()

    response = client.post(
        "/v1/tag/audio",
        data={"provider": "bogus"},
        files=[("files", ("a.mp3", MEDIA_BYTES, "audio/mpeg"))],
    )

    assert response.status_code == 422


@pytest.mark.parametrize("provider", ["bogus", ""])
def test_json_audio_rejects_unsupported_provider(provider: str):
    client = build_client()

    response = client.post(
        "/v1/tag/audio/json",
        json={
            "provider": provider,
            "media": [
                {
                    "filename": "a.mp3",
                    "content_base64": base64.b64encode(MEDIA_BYTES).decode(),
                }
            ],
        },
    )

    assert response.status_code == 422


def test_json_audio_language_precedence_per_item_over_payload():
    transcriber = RecordingTranscriptionClient()
    client = build_client(transcriber=transcriber)

    response = client.post(
        "/v1/tag/audio/json",
        json={
            "provider": "openai",
            "language": "es",
            "media": [
                {
                    "filename": "with-item-language.m4a",
                    "content_base64": base64.b64encode(MEDIA_BYTES).decode(),
                    "language": "fr",
                },
                {
                    "filename": "falls-back-to-payload.m4a",
                    "content_base64": base64.b64encode(MEDIA_BYTES).decode(),
                },
            ],
        },
    )

    assert response.status_code == 200
    # Per-item language wins; otherwise the payload-level language is used.
    assert [media.language for media in transcriber.calls] == ["fr", "es"]


def test_multipart_audio_disambiguates_duplicate_filenames():
    transcriber = RecordingTranscriptionClient()
    client = build_client(transcriber=transcriber)

    response = client.post(
        "/v1/tag/audio",
        data={"provider": "openai"},
        files=[
            ("files", ("dupe.mp3", MEDIA_BYTES, "audio/mpeg")),
            ("files", ("dupe.mp3", MEDIA_BYTES, "audio/mpeg")),
        ],
    )

    assert response.status_code == 200
    # Both uploads survive as distinct results instead of collapsing into one.
    filenames = [result["filename"] for result in response.json()["results"]]
    assert len(filenames) == 2
    assert len(set(filenames)) == 2


def test_json_audio_requires_at_least_one_file():
    client = build_client()

    response = client.post("/v1/tag/audio/json", json={"provider": "openai", "media": []})

    assert response.status_code == 422


def test_json_audio_rejects_invalid_base64():
    client = build_client()

    response = client.post(
        "/v1/tag/audio/json",
        json={
            "provider": "openai",
            "media": [{"filename": "a.mp3", "content_base64": "not-valid-base64!!"}],
        },
    )

    assert response.status_code == 422


def test_transcription_failure_is_returned_as_502():
    client = build_client(transcriber=RecordingTranscriptionClient(fail=True))

    response = client.post(
        "/v1/tag/audio",
        data={"provider": "openai"},
        files=[("files", ("a.mp3", MEDIA_BYTES, "audio/mpeg"))],
    )

    assert response.status_code == 502
    assert "stt-tts unavailable" in response.json()["detail"]
