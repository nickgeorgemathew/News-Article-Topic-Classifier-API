import os
from datetime import datetime,timezone,time
from fastapi import FastAPI,HTTPException
from pathlib import Path
from pydantic import BaseModel
from typing import List
from data_and_model.predict_utils import ModelWrapper
from logs.db import sessionlocal
from logs.model import Predictionlog

MODEL_DIR = os.environ.get("MODEL_DIR", "../models/distilbert-agnews-v1")
db_path = os.environ.get("DB_PATH", "./logs/logs.db")


class PredictRequest(BaseModel):
    text: str
    top_k: int = 3

class PredictResponse(BaseModel):
    predictions: List[dict]
    model_version: str



app=FastAPI(title="News Topic Classifier(DistilBert)")


@app.get('/')
def root():
    return("News Topic Classifier(DistilBert)")
@app.post('/prediction',response_model=PredictResponse)
def prediciton(req:PredictRequest):
    db=sessionlocal()
    start=time.perf_counter()
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400,detail="text is empty")
    preds,prob=ModelWrapper.predict(text=req.text,top_k=req.top_k)
    top=preds[0]
    log=Predictionlog(        
    timestamp=datetime.now(timezone.utc()),
    Text=req.text[:4000],
    top_prediction=top["label"],
    top_score=top["score"],
    all_scores=preds,
    model_version=ModelWrapper.model_version,
    latency_ms=((time.perf_counter()-start)*1000)

    )
    db.add(log)
    db.commit()
    return {"predictions": preds, "model_version": ModelWrapper.model_version}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": ModelWrapper.model_version}

@app.get("/metrics")
def get_metrics():
    
    with Session(engine) as session:
        statement = select(User)
        results = session.exec(statement)
        return results.all()