from __future__ import annotations

import json

import httpx
import pytest

from image_tagging_api.config import Settings
from image_tagging_api.providers.healthcheck import HttpProviderStartupChecker


@pytest.mark.asyncio
async def test_http_startup_checker_tests_anthropic_key_with_messages_endpoint():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    checker = HttpProviderStartupChecker(client=client)

    checks = await checker.check_configured_providers(
        Settings(
            anthropic_api_key="sk-ant-test",
            openai_api_key=None,
            gemini_api_key=None,
        )
    )

    assert len(checks) == 1
    assert checks[0].provider == "anthropic"
    assert checks[0].ok is True
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-3-5-sonnet-latest"
    await client.aclose()


@pytest.mark.asyncio
async def test_http_startup_checker_reports_invalid_anthropic_key():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "type": "error",
                "error": {"type": "authentication_error", "message": "invalid x-api-key"},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    checker = HttpProviderStartupChecker(client=client)

    checks = await checker.check_configured_providers(Settings(anthropic_api_key="bad-key"))

    assert checks[0].provider == "anthropic"
    assert checks[0].ok is False
    assert "HTTP 401" in checks[0].message
    assert "invalid x-api-key" in checks[0].message
    await client.aclose()
