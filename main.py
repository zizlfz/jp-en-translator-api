from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import uvicorn

app = FastAPI(
    title="Japanese to English Translator API",
    description="Translates Japanese text to English using Helsinki-NLP/opus-mt-ja-en ONNX model",
    version="1.0.0",
)

MODEL_ID = "Helsinki-NLP/opus-mt-ja-en"
REVISION = "e1b0895a1cb46d229c140658331bd34bd3e0bfee"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ID, subfolder="onnx", revision=REVISION)
print("Model loaded!")


class TranslateRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "私は日本語を話します。"}]
        }
    }


class TranslateResponse(BaseModel):
    input: str
    translation: str


class BatchTranslateRequest(BaseModel):
    texts: list[str]

    model_config = {
        "json_schema_extra": {
            "examples": [{"texts": ["私は日本語を話します。", "東京は日本の首都です。"]}]
        }
    }


class BatchTranslateResponse(BaseModel):
    results: list[TranslateResponse]


@app.get("/")
def root():
    return {"message": "Japanese to English Translator API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return TranslateResponse(input=request.text, translation=translation)


@app.post("/translate/batch", response_model=BatchTranslateResponse)
def translate_batch(request: BatchTranslateRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    if len(request.texts) > 32:
        raise HTTPException(status_code=400, detail="Maximum batch size is 32")

    inputs = tokenizer(request.texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results = [
        TranslateResponse(input=text, translation=translation)
        for text, translation in zip(request.texts, translations)
    ]
    return BatchTranslateResponse(results=results)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
