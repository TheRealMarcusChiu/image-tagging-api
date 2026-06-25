from __future__ import annotations

from fastapi.testclient import TestClient

from image_tagging_api.config import Settings
from image_tagging_api.main import create_app
from image_tagging_api.providers.healthcheck import ProviderCredentialCheck, ProviderStartupChecker


class RecordingStartupChecker(ProviderStartupChecker):
    def __init__(self) -> None:
        self.called_with: Settings | None = None

    async def check_configured_providers(self, settings: Settings) -> list[ProviderCredentialCheck]:
        self.called_with = settings
        return [
            ProviderCredentialCheck(
                provider="anthropic",
                configured=True,
                ok=True,
                message="Anthropic API key accepted.",
            )
        ]


def test_startup_checks_configured_provider_api_keys():
    checker = RecordingStartupChecker()
    settings = Settings(anthropic_api_key="sk-ant-test")
    app = create_app(settings=settings, startup_checker=checker)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert checker.called_with is settings
    assert app.state.provider_credential_checks == [
        ProviderCredentialCheck(
            provider="anthropic",
            configured=True,
            ok=True,
            message="Anthropic API key accepted.",
        )
    ]


def test_health_includes_startup_credential_check_results():
    checker = RecordingStartupChecker()
    app = create_app(settings=Settings(anthropic_api_key="sk-ant-test"), startup_checker=checker)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.json()["credential_checks"] == [
        {
            "provider": "anthropic",
            "configured": True,
            "ok": True,
            "message": "Anthropic API key accepted.",
        }
    ]
