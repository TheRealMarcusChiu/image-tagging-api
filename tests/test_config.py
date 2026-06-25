from image_tagging_api.config import Settings


def test_default_model_for_maps_deprecated_anthropic_latest_alias():
    settings = Settings(default_anthropic_model="claude-3-5-sonnet-latest")

    assert settings.default_model_for("anthropic") == "claude-sonnet-4-6"
