from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DIR
from src.inference import latest_alert_for_unit, load_artifacts


app = FastAPI(title="Signal Anomaly Detection and Decision Support API")


class SensorReading(BaseModel):
    unit_id: int = Field(..., examples=[1])
    cycle: int = Field(..., examples=[215])
    setting_1: float = 0.0
    setting_2: float = 0.0
    setting_3: float = 0.0
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sample-alert")
def sample_alert() -> dict[str, Any]:
    sample_path = PROCESSED_DIR / "test.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Run training pipeline first to create processed data.")
    df = pd.read_csv(sample_path)
    unit_id = int(df["unit_id"].iloc[0])
    return latest_alert_for_unit(df[df["unit_id"] == unit_id], load_artifacts()).__dict__


@app.post("/predict")
def predict(readings: list[SensorReading]) -> dict[str, Any]:
    if not readings:
        raise HTTPException(status_code=400, detail="Provide at least one reading for a single unit.")
    df = pd.DataFrame([reading.model_dump() for reading in readings])
    alert = latest_alert_for_unit(df, load_artifacts())
    return alert.__dict__
