import base64

import pytest
from fastapi.testclient import TestClient

from image_tagging_api.config import Settings
from image_tagging_api.main import create_app
from image_tagging_api.models import ImageTagResult, TaggingResponse
from image_tagging_api.providers.base import ImageTagger, ProviderRequest

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class RecordingTagger(ImageTagger):
    def __init__(self):
        self.calls: list[ProviderRequest] = []

    async def tag_images(self, request: ProviderRequest) -> TaggingResponse:
        self.calls.append(request)
        return TaggingResponse(
            provider=request.provider,
            model=request.model,
            results=[
                ImageTagResult(
                    filename=image.filename,
                    tags=["cat", "indoor"],
                    confidence=0.91,
                    explanation="A cat appears indoors.",
                )
                for image in request.images
            ],
        )


def build_client(tagger: RecordingTagger, settings: Settings | None = None) -> TestClient:
    app = create_app(settings=settings or Settings(), tagger=tagger)
    return TestClient(app)


def test_health_reports_supported_providers():
    tagger = RecordingTagger()
    client = build_client(tagger)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["providers"]) == {"openai", "anthropic", "gemini", "ollama"}


def test_multipart_endpoint_tags_batch_and_forwards_model_choice():
    tagger = RecordingTagger()
    client = build_client(tagger)

    response = client.post(
        "/v1/tag",
        data={
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "candidate_tags": '["cat", "dog", "indoor"]',
            "max_tags": "2",
            "include_explanations": "true",
        },
        files=[
            ("images", ("first.png", PNG_BYTES, "image/png")),
            ("images", ("second.png", PNG_BYTES, "image/png")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "anthropic"
    assert body["model"] == "claude-sonnet-4-6"
    assert [result["filename"] for result in body["results"]] == ["first.png", "second.png"]
    assert body["results"][0]["tags"] == ["cat", "indoor"]

    forwarded = tagger.calls[0]
    assert forwarded.provider == "anthropic"
    assert forwarded.model == "claude-sonnet-4-6"
    assert forwarded.candidate_tags == ["cat", "dog", "indoor"]
    assert forwarded.max_tags == 2
    assert forwarded.include_explanations is True


def test_multipart_endpoint_maps_stale_default_anthropic_model_when_model_omitted():
    tagger = RecordingTagger()
    client = build_client(
        tagger,
        Settings(default_anthropic_model="claude-3-5-sonnet-latest"),
    )

    response = client.post(
        "/v1/tag",
        data={"provider": "anthropic"},
        files=[("images", ("shirt.png", PNG_BYTES, "image/png"))],
    )

    assert response.status_code == 200
    assert response.json()["model"] == "claude-sonnet-4-6"
    assert tagger.calls[0].model == "claude-sonnet-4-6"


def test_json_endpoint_accepts_base64_images():
    tagger = RecordingTagger()
    client = build_client(tagger)

    response = client.post(
        "/v1/tag/json",
        json={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "candidate_tags": ["cat", "outdoor"],
            "images": [
                {
                    "filename": "pixel.png",
                    "content_base64": base64.b64encode(PNG_BYTES).decode(),
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["filename"] == "pixel.png"
    assert tagger.calls[0].images[0].mime_type == "image/png"


@pytest.mark.parametrize("provider", ["bogus", ""])
def test_rejects_unsupported_provider(provider: str):
    tagger = RecordingTagger()
    client = build_client(tagger)

    response = client.post(
        "/v1/tag/json",
        json={
            "provider": provider,
            "model": "anything",
            "images": [
                {
                    "filename": "pixel.png",
                    "content_base64": base64.b64encode(PNG_BYTES).decode(),
                }
            ],
        },
    )

    assert response.status_code == 422


def test_requires_at_least_one_image():
    tagger = RecordingTagger()
    client = build_client(tagger)

    response = client.post("/v1/tag/json", json={"provider": "openai", "images": []})

    assert response.status_code == 422
