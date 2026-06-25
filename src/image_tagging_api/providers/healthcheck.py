from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import BaseModel

from image_tagging_api.config import Settings
from image_tagging_api.models import ProviderName

logger = logging.getLogger(__name__)


class ProviderCredentialCheck(BaseModel):
    provider: ProviderName
    configured: bool
    ok: bool
    message: str


@dataclass(frozen=True)
class ProviderCredential:
    provider: ProviderName
    api_key: str | None


class ProviderStartupChecker(Protocol):
    async def check_configured_providers(
        self, settings: Settings
    ) -> list[ProviderCredentialCheck]: ...


class HttpProviderStartupChecker:
    """Performs lightweight provider auth checks for configured API keys at startup."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client

    async def check_configured_providers(self, settings: Settings) -> list[ProviderCredentialCheck]:
        if not settings.validate_provider_credentials_on_startup:
            return []

        checks: list[ProviderCredentialCheck] = []
        async with self._client(settings) as client:
            if settings.anthropic_auth_headers():
                checks.append(await self._check_anthropic(client, settings))
            if settings.openai_api_key:
                checks.append(await self._check_openai(client, settings))
            if settings.gemini_api_key:
                checks.append(await self._check_gemini(client, settings))

        for check in checks:
            log = logger.info if check.ok else logger.error
            log("Provider startup credential check: %s - %s", check.provider, check.message)

        return checks

    def _client(self, settings: Settings) -> httpx.AsyncClient:
        if self.client is not None:
            return _BorrowedAsyncClient(self.client)
        return httpx.AsyncClient(timeout=settings.startup_check_timeout_seconds)

    async def _check_anthropic(
        self, client: httpx.AsyncClient, settings: Settings
    ) -> ProviderCredentialCheck:
        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    **settings.anthropic_auth_headers(),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.default_anthropic_model,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "Reply with ok."}],
                },
            )
            return self._result_from_response("anthropic", response)
        except httpx.HTTPError as exc:
            return ProviderCredentialCheck(
                provider="anthropic",
                configured=True,
                ok=False,
                message=f"Anthropic startup check request failed: {exc}",
            )

    async def _check_openai(
        self, client: httpx.AsyncClient, settings: Settings
    ) -> ProviderCredentialCheck:
        try:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
            return self._result_from_response("openai", response)
        except httpx.HTTPError as exc:
            return ProviderCredentialCheck(
                provider="openai",
                configured=True,
                ok=False,
                message=f"OpenAI startup check request failed: {exc}",
            )

    async def _check_gemini(
        self, client: httpx.AsyncClient, settings: Settings
    ) -> ProviderCredentialCheck:
        try:
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": settings.gemini_api_key},
            )
            return self._result_from_response("gemini", response)
        except httpx.HTTPError as exc:
            return ProviderCredentialCheck(
                provider="gemini",
                configured=True,
                ok=False,
                message=f"Gemini startup check request failed: {exc}",
            )

    def _result_from_response(
        self, provider: ProviderName, response: httpx.Response
    ) -> ProviderCredentialCheck:
        if 200 <= response.status_code < 300:
            return ProviderCredentialCheck(
                provider=provider,
                configured=True,
                ok=True,
                message=f"{provider} credential accepted.",
            )
        return ProviderCredentialCheck(
            provider=provider,
            configured=True,
            ok=False,
            message=(
                f"{provider} startup check returned HTTP {response.status_code}: "
                f"{response.text}"
            ),
        )


class _BorrowedAsyncClient(httpx.AsyncClient):
    """Context manager wrapper that does not close an injected test/shared client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._borrowed_client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._borrowed_client

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        return None
