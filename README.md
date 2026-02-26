# Japanese to English Translator API

FastAPI-based REST API for translating Japanese text to English using the Helsinki-NLP/opus-mt-ja-en ONNX model.

## Setup

```bash
# Install dependencies
uv sync
```

## Running

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/translate` | POST | Translate single text |
| `/translate/batch` | POST | Translate multiple texts (max 32) |

## Examples

### Translate single text

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "私は日本語を話します。"}'
```

Response:
```json
{
  "input": "私は日本語を話します。",
  "translation": "I speak Japanese."
}
```

### Batch translation

```bash
curl -X POST http://localhost:8000/translate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["私は日本語を話します。", "東京は日本の首都です。"]}'
```

## Documentation

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).
