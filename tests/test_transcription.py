from __future__ import annotations

import json

import httpx
import pytest

from image_tagging_api.config import Settings
from image_tagging_api.providers.base import MediaInput, Transcript, TranscriptTaggingRequest
from image_tagging_api.providers.transcription import (
    SttTtsTranscriptionClient,
    TranscriptionError,
)
from image_tagging_api.providers.vision import MultiProviderImageTagger, normalize_tag


@pytest.mark.asyncio
async def test_transcription_client_posts_file_and_parses_transcript():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content"] = request.content
        return httpx.Response(
            200,
            json={
                "task": "transcribe",
                "language": "en",
                "duration": 12.3,
                "text": "  Hello world.  ",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(stt_tts_base_url="https://stt-tts.lan", stt_model="large-v3-turbo")
    transcriber = SttTtsTranscriptionClient(settings, client=client)

    transcript = await transcriber.transcribe(
        MediaInput(filename="meeting.mp3", content=b"audio-bytes", mime_type="audio/mpeg")
    )

    assert transcript == Transcript(
        filename="meeting.mp3", text="Hello world.", language="en", duration=12.3
    )
    assert captured["url"] == "https://stt-tts.lan/v1/audio/transcriptions"
    assert b"verbose_json" in captured["content"]
    assert b"large-v3-turbo" in captured["content"]
    assert b"meeting.mp3" in captured["content"]
    assert b"audio-bytes" in captured["content"]
    await client.aclose()


@pytest.mark.asyncio
async def test_transcription_client_maps_http_error_to_transcription_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transcriber = SttTtsTranscriptionClient(Settings(), client=client)

    with pytest.raises(TranscriptionError) as exc_info:
        await transcriber.transcribe(
            MediaInput(filename="a.mp3", content=b"x", mime_type="audio/mpeg")
        )

    assert "HTTP 503" in str(exc_info.value)
    await client.aclose()


def test_normalize_tag_converts_underscores_to_spaces():
    assert normalize_tag("wedding_vows") == "wedding vows"
    assert normalize_tag("yes_i_do") == "yes i do"
    assert normalize_tag("  multiple__under_scores ") == "multiple under scores"
    assert normalize_tag("already spaced") == "already spaced"


@pytest.mark.asyncio
async def test_tag_transcripts_renders_tags_with_spaces():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "images": [
                                        {
                                            "filename": "vows.mp4",
                                            "tags": ["wedding_vows", "yes_i_do"],
                                            "confidence": 0.9,
                                            "explanation": "A wedding ceremony.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tagger = MultiProviderImageTagger(Settings(openai_api_key="test-key"), client=client)

    response = await tagger.tag_transcripts(
        TranscriptTaggingRequest(
            provider="openai",
            model="gpt-4.1-mini",
            transcripts=[
                Transcript(filename="vows.mp4", text="I do.", language="en", duration=3.0)
            ],
            candidate_tags=None,
            max_tags=5,
            include_explanations=True,
        )
    )

    assert response.results[0].tags == ["wedding vows", "yes i do"]
    await client.aclose()


@pytest.mark.asyncio
async def test_transcription_client_rejects_non_object_json():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transcriber = SttTtsTranscriptionClient(Settings(), client=client)

    with pytest.raises(TranscriptionError) as exc_info:
        await transcriber.transcribe(
            MediaInput(filename="a.mp3", content=b"x", mime_type="audio/mpeg")
        )

    assert "unexpected JSON shape" in str(exc_info.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_tag_transcripts_skips_llm_when_all_transcripts_empty():
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tagger = MultiProviderImageTagger(Settings(openai_api_key="test-key"), client=client)

    response = await tagger.tag_transcripts(
        TranscriptTaggingRequest(
            provider="openai",
            model="gpt-4.1-mini",
            transcripts=[
                Transcript(filename="silent.mp4", text="   ", language=None, duration=0.0)
            ],
            candidate_tags=None,
            max_tags=5,
            include_explanations=True,
        )
    )

    assert called is False
    assert response.results[0].filename == "silent.mp4"
    assert response.results[0].tags == []
    assert response.results[0].explanation == "No speech detected."
    await client.aclose()


@pytest.mark.asyncio
async def test_tag_transcripts_sends_text_only_prompt_and_parses_output():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "images": [
                                        {
                                            "filename": "talk.mp4",
                                            "tags": ["ai", "ethics", "extra"],
                                            "confidence": 0.7,
                                            "explanation": "About AI ethics.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tagger = MultiProviderImageTagger(Settings(openai_api_key="test-key"), client=client)

    response = await tagger.tag_transcripts(
        TranscriptTaggingRequest(
            provider="openai",
            model="gpt-4.1-mini",
            transcripts=[
                Transcript(
                    filename="talk.mp4",
                    text="We discussed AI ethics at length.",
                    language="en",
                    duration=5.0,
                )
            ],
            candidate_tags=None,
            max_tags=2,
            include_explanations=True,
        )
    )

    assert response.results[0].filename == "talk.mp4"
    # Tags truncated to max_tags.
    assert response.results[0].tags == ["ai", "ethics"]

    # The LLM request is text-only (no image blocks) and embeds the transcript.
    content = captured["body"]["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "We discussed AI ethics at length." in content[0]["text"]
    await client.aclose()
