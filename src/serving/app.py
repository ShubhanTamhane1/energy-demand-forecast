"""FastAPI serving layer for the ISO-NE demand forecast model.

Exposes:
  - GET  /ping         -- SageMaker health check contract (must return 200)
  - POST /invocations   -- SageMaker inference contract (request/response shape
                            is ours to define for a bring-your-own-container
                            endpoint; mirrors /predict)
  - POST /predict        -- same contract, friendlier route name for local/
                            generic use
  - GET  /health          -- plain liveness check for local dev / non-SageMaker use
"""

from pathlib import Path
from typing import List

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = Path(__file__).parent / "model"

app = FastAPI(title="ISO-NE Demand Forecast API")
model = mlflow.pyfunc.load_model(str(MODEL_DIR))


class DemandFeatures(BaseModel):
    temperature_2m: float
    precipitation: float
    rain: float
    snowfall: float
    hour: int
    day_of_week: int
    month: int
    is_weekend: bool
    is_holiday: bool
    hour_sin: float
    hour_cos: float
    day_of_week_sin: float
    day_of_week_cos: float
    did_rain: bool
    did_snow: bool
    heating_degree_hours: float
    cooling_degree_hours: float
    lag_24h: float
    lag_48h: float
    lag_168h: float
    rolling_24h_mean: float
    rolling_168h_mean: float


class PredictRequest(BaseModel):
    instances: List[DemandFeatures]


class PredictResponse(BaseModel):
    predictions: List[float]


# The logged model's signature requires int32 for these columns; pandas
# defaults python ints to int64, which mlflow's schema enforcement rejects.
INT32_COLS = ["hour", "day_of_week", "month"]


def _predict(request: PredictRequest) -> PredictResponse:
    df = pd.DataFrame([instance.model_dump() for instance in request.instances])
    df[INT32_COLS] = df[INT32_COLS].astype("int32")
    preds = model.predict(df)
    return PredictResponse(predictions=[float(p) for p in preds])


@app.get("/ping")
def ping():
    # SageMaker polls this to decide whether the container is healthy.
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/invocations", response_model=PredictResponse)
def invocations(request: PredictRequest):
    return _predict(request)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return _predict(request)
