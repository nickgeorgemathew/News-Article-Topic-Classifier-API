"""
FastAPI REST API for text classification inference.

Run with:
    uvicorn app.fastapi_app:app --reload

Endpoints:
    POST /predict        — classify a single text
    POST /predict/batch  — classify a list of texts
    GET  /health         — health check
    GET  /info           — model & config info
"""

import os
import sys
import yaml
import joblib
import time
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessor import TextPreprocessor

# -----------------------------------------------------------------------
# Config & globals
# -----------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")

AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Lazy-loaded resources
_model = None
_preprocessor = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run the training pipeline first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor(config_path=CONFIG_PATH)
    return _preprocessor


def resolve_class_names():
    if CONFIG["dataset"]["name"] == "ag_news":
        return list(AG_NEWS_LABELS.values())
    return None


# -----------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, example="Tesla earnings beat forecasts.")

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class PredictionResult(BaseModel):
    text_preview: str
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    preprocessing_ms: float
    inference_ms: float

class BatchPredictionResponse(BaseModel):
    count: int
    results: List[PredictionResult]
    total_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class InfoResponse(BaseModel):
    dataset: str
    class_names: Optional[List[str]]
    model_path: str
    config: dict


# -----------------------------------------------------------------------
# App
# -----------------------------------------------------------------------

app = FastAPI(
    title="Text Classification API",
    description="Classify news text using a TF-IDF + ML pipeline.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    model_loaded = os.path.exists(MODEL_PATH)
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.get("/info", response_model=InfoResponse, tags=["System"])
def info():
    return InfoResponse(
        dataset=CONFIG["dataset"]["name"],
        class_names=resolve_class_names(),
        model_path=MODEL_PATH,
        config={k: v for k, v in CONFIG.items() if k != "paths"}
    )


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict(request: PredictRequest):
    """Classify a single text string."""
    model = get_model()
    preprocessor = get_preprocessor()
    class_names = resolve_class_names()

    t0 = time.perf_counter()
    processed = preprocessor.preprocess(request.text)
    t1 = time.perf_counter()
    pred = model.predict([processed])[0]
    proba = model.predict_proba([processed])[0]
    t2 = time.perf_counter()

    label = class_names[pred] if class_names else str(pred)

    return PredictionResult(
        text_preview=request.text[:120],
        predicted_class=int(pred),
        predicted_label=label,
        confidence=round(float(proba.max()), 6),
        probabilities={
            (class_names[i] if class_names else str(i)): round(float(p), 6)
            for i, p in enumerate(proba)
        },
        preprocessing_ms=round((t1 - t0) * 1000, 2),
        inference_ms=round((t2 - t1) * 1000, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: PredictBatchRequest):
    """Classify a batch of texts (max 100)."""
    model = get_model()
    preprocessor = get_preprocessor()
    class_names = resolve_class_names()

    t_start = time.perf_counter()

    results = []
    for text in request.texts:
        t0 = time.perf_counter()
        processed = preprocessor.preprocess(text)
        t1 = time.perf_counter()
        pred = model.predict([processed])[0]
        proba = model.predict_proba([processed])[0]
        t2 = time.perf_counter()

        label = class_names[pred] if class_names else str(pred)
        results.append(PredictionResult(
            text_preview=text[:120],
            predicted_class=int(pred),
            predicted_label=label,
            confidence=round(float(proba.max()), 6),
            probabilities={
                (class_names[i] if class_names else str(i)): round(float(p), 6)
                for i, p in enumerate(proba)
            },
            preprocessing_ms=round((t1 - t0) * 1000, 2),
            inference_ms=round((t2 - t1) * 1000, 2),
        ))

    return BatchPredictionResponse(
        count=len(results),
        results=results,
        total_ms=round((time.perf_counter() - t_start) * 1000, 2),
    )
