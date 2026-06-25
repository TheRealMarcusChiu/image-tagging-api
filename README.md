# Image Tagging API

A FastAPI server that accepts one image or a batch of images and returns tags based on visible image content. The caller can choose the provider and model per HTTP request.

Supported providers:

- OpenAI vision chat models
- Anthropic Claude vision models
- Google Gemini vision models
- Local Ollama vision models

## Features

- `POST /v1/tag` multipart upload for one or more files
- `POST /v1/tag/json` JSON API with base64-encoded images
- Per-request `provider` and `model` selection
- Optional candidate tag list so the model chooses from your taxonomy
- Optional explanations and confidence scores
- Batch responses preserve input filenames
- Provider errors are returned as HTTP 502; validation errors as HTTP 422

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn image_tagging_api.main:app --host 0.0.0.0 --port 8000
```

Open docs at http://localhost:8000/docs.

## Environment variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
```

Only set the keys for providers you intend to use. Ollama needs a local vision-capable model, for example:

```bash
ollama pull llava:latest
```

## Multipart API

```bash
curl -X POST http://localhost:8000/v1/tag \
  -F provider=openai \
  -F model=gpt-4.1-mini \
  -F 'candidate_tags=["cat","dog","indoor","outdoor","food","document"]' \
  -F max_tags=5 \
  -F include_explanations=true \
  -F images=@/path/to/photo1.jpg \
  -F images=@/path/to/photo2.png
```

Response:

```json
{
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "results": [
    {
      "filename": "photo1.jpg",
      "tags": ["cat", "indoor"],
      "confidence": 0.91,
      "explanation": "A cat appears indoors.",
      "raw_response": null
    }
  ]
}
```

## JSON API

```bash
BASE64_IMAGE=$(base64 -w0 /path/to/photo.jpg)

curl -X POST http://localhost:8000/v1/tag/json \
  -H 'Content-Type: application/json' \
  -d "{
    \"provider\": \"anthropic\",
    \"model\": \"claude-3-5-sonnet-latest\",
    \"candidate_tags\": [\"cat\", \"dog\", \"indoor\"],
    \"max_tags\": 5,
    \"include_explanations\": true,
    \"images\": [
      {
        \"filename\": \"photo.jpg\",
        \"mime_type\": \"image/jpeg\",
        \"content_base64\": \"$BASE64_IMAGE\"
      }
    ]
  }"
```

## Provider/model examples

- OpenAI: `provider=openai`, `model=gpt-4.1-mini` or another vision-capable chat model
- Anthropic: `provider=anthropic`, `model=claude-3-5-sonnet-latest`
- Gemini: `provider=gemini`, `model=gemini-2.0-flash`
- Ollama: `provider=ollama`, `model=llava:latest`

If `model` is omitted, provider-specific defaults from `Settings` are used.

## Docker

```bash
docker build -t image-tagging-api .
docker run --rm -p 8000:8000 --env-file .env image-tagging-api
```

## Development

```bash
source .venv/bin/activate
pytest -q
ruff check .
```
