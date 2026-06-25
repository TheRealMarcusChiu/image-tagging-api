from __future__ import annotations

import json

import httpx
import pytest
from fastapi import HTTPException

from image_tagging_api.config import Settings
from image_tagging_api.models import ImageTagResult, TaggingResponse
from image_tagging_api.providers.base import ImageInput, ProviderRequest
from image_tagging_api.providers.vision import MultiProviderImageTagger, parse_model_output


@pytest.fixture
def provider_request() -> ProviderRequest:
    return ProviderRequest(
        provider="openai",
        model="gpt-4.1-mini",
        images=[ImageInput(filename="cat.png", content=b"image-bytes", mime_type="image/png")],
        candidate_tags=["cat", "dog"],
        max_tags=3,
        include_explanations=True,
    )


def test_parse_model_output_maps_results_to_input_order(provider_request: ProviderRequest):
    output = json.dumps(
        {
            "images": [
                {
                    "filename": "cat.png",
                    "tags": ["cat", "indoor", "pet", "extra"],
                    "confidence": 0.88,
                    "explanation": "A cat is visible.",
                }
            ]
        }
    )

    response = parse_model_output(output, provider_request)

    assert response == TaggingResponse(
        provider="openai",
        model="gpt-4.1-mini",
        results=[
            ImageTagResult(
                filename="cat.png",
                tags=["cat", "indoor", "pet"],
                confidence=0.88,
                explanation="A cat is visible.",
            )
        ],
    )


def test_parse_model_output_includes_response_excerpt_when_provider_returns_no_json(
    provider_request: ProviderRequest,
):
    with pytest.raises(Exception) as exc_info:
        parse_model_output(
            "I see a blue shirt hanging on a hanger with short sleeves.",
            provider_request,
        )

    assert str(exc_info.value) == (
        "Could not parse provider response: Provider did not return JSON. "
        "Response excerpt: I see a blue shirt hanging on a hanger with short sleeves."
    )


@pytest.mark.asyncio
async def test_openai_provider_uses_selected_model_and_image_payload(
    provider_request: ProviderRequest,
):
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
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
                                            "filename": "cat.png",
                                            "tags": ["cat"],
                                            "confidence": 0.9,
                                            "explanation": "A cat.",
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

    response = await tagger.tag_images(provider_request)

    assert response.results[0].tags == ["cat"]
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "gpt-4.1-mini"
    assert captured["body"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_provider_uses_claude_code_oauth_token_as_bearer_auth():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "images": [
                                    {
                                        "filename": "cat.png",
                                        "tags": ["cat"],
                                        "confidence": 0.9,
                                        "explanation": "A cat.",
                                    }
                                ]
                            }
                        ),
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tagger = MultiProviderImageTagger(
        Settings(claude_code_oauth_token="oauth-token", anthropic_api_key=None),
        client=client,
    )

    response = await tagger.tag_images(
        ProviderRequest(
            provider="anthropic",
            model="claude-sonnet-4-6",
            images=[ImageInput(filename="cat.png", content=b"image-bytes", mime_type="image/png")],
            candidate_tags=None,
            max_tags=3,
            include_explanations=True,
        )
    )

    assert response.results[0].tags == ["cat"]
    assert captured["headers"]["authorization"] == "Bearer oauth-token"
    assert "x-api-key" not in captured["headers"]
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-sonnet-4-6"
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_provider_disables_thinking_and_requests_json_output():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "images": [
                            {
                                "filename": "cat.png",
                                "tags": ["cat"],
                                "confidence": 0.9,
                                "explanation": "A cat.",
                            }
                        ]
                    }
                )
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tagger = MultiProviderImageTagger(Settings(), client=client)

    response = await tagger.tag_images(
        ProviderRequest(
            provider="ollama",
            model="qwen3-vl:8b",
            images=[ImageInput(filename="cat.png", content=b"image-bytes", mime_type="image/png")],
            candidate_tags=None,
            max_tags=3,
            include_explanations=True,
        )
    )

    assert response.results[0].tags == ["cat"]
    assert captured["body"]["model"] == "qwen3-vl:8b"
    assert captured["body"]["format"] == "json"
    assert captured["body"]["stream"] is False
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["temperature"] == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_provider_falls_back_to_thinking_field_when_response_empty():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3-vl:8b",
                "response": "",
                "thinking": json.dumps(
                    {
                        "images": [
                            {
                                "filename": "cat.png",
                                "tags": ["cat"],
                                "confidence": 0.9,
                                "explanation": "A cat.",
                            }
                        ]
                    }
                ),
                "done": True,
                "done_reason": "stop",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tagger = MultiProviderImageTagger(Settings(), client=client)

    response = await tagger.tag_images(
        ProviderRequest(
            provider="ollama",
            model="qwen3-vl:8b",
            images=[
                ImageInput(
                    filename="cat.png",
                    content=b"image-bytes",
                    mime_type="image/png",
                )
            ],
            candidate_tags=None,
            max_tags=3,
            include_explanations=True,
        )
    )

    assert response.results[0].tags == ["cat"]
    assert response.results[0].confidence == 0.9
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_provider_reports_empty_response_with_raw_payload_details():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3-vl:8b",
                "response": "",
                "done": True,
                "done_reason": "stop",
                "total_duration": 123,
                "eval_count": 456,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tagger = MultiProviderImageTagger(Settings(), client=client)

    with pytest.raises(HTTPException) as exc_info:
        await tagger.tag_images(
            ProviderRequest(
                provider="ollama",
                model="qwen3-vl:8b",
                images=[
                    ImageInput(
                        filename="cat.png",
                        content=b"image-bytes",
                        mime_type="image/png",
                    )
                ],
                candidate_tags=None,
                max_tags=3,
                include_explanations=True,
            )
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == (
        "Ollama returned an empty response field. "
        'Payload excerpt: {"model":"qwen3-vl:8b","response":"","done":true,'
        '"done_reason":"stop","total_duration":123,"eval_count":456}'
    )
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_rate_limit_is_returned_as_client_429():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "Error"},
                "request_id": "req_test",
            },
            headers={"retry-after": "30"},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tagger = MultiProviderImageTagger(Settings(openai_api_key="test-key"), client=client)

    with pytest.raises(HTTPException) as exc_info:
        await tagger.tag_images(
            ProviderRequest(
                provider="openai",
                model="gpt-4.1-mini",
                images=[
                    ImageInput(
                        filename="cat.png",
                        content=b"image-bytes",
                        mime_type="image/png",
                    )
                ],
                candidate_tags=None,
                max_tags=3,
                include_explanations=True,
            )
        )

    assert exc_info.value.status_code == 429
    assert "rate limited" in exc_info.value.detail
    assert "retry after 30 seconds" in exc_info.value.detail
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_request_failure_includes_exception_type_and_message():
    transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("", request=request))
    )
    client = httpx.AsyncClient(transport=transport)
    tagger = MultiProviderImageTagger(Settings(), client=client)

    with pytest.raises(HTTPException) as exc_info:
        await tagger.tag_images(
            ProviderRequest(
                provider="ollama",
                model="llava:latest",
                images=[
                    ImageInput(
                        filename="cat.png",
                        content=b"image-bytes",
                        mime_type="image/png",
                    )
                ],
                candidate_tags=None,
                max_tags=3,
                include_explanations=True,
            )
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == (
        "Provider ollama request failed: ConnectError"
    )
    await client.aclose()
