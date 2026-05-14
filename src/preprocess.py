from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ANOMALY_RUL_THRESHOLD, SENSOR_COLUMNS, WARNING_RUL_THRESHOLD


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_anomaly"] = (out["rul"] <= ANOMALY_RUL_THRESHOLD).astype(int)
    conditions = [
        out["rul"] <= ANOMALY_RUL_THRESHOLD,
        out["rul"] <= WARNING_RUL_THRESHOLD,
    ]
    choices = ["critical_failure_risk", "degradation_warning"]
    out["event_type"] = np.select(conditions, choices, default="normal_operation")
    return out


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def sensor_columns_present(df: pd.DataFrame) -> list[str]:
    return [column for column in SENSOR_COLUMNS if column in df.columns]
