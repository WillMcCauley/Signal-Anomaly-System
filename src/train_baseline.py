from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from .config import MODEL_DIR, RANDOM_STATE
from .features import get_feature_columns
from .preprocess import build_preprocessor


def train_isolation_forest(train_df: pd.DataFrame, contamination: float = 0.12) -> tuple[Pipeline, list[str]]:
    feature_columns = get_feature_columns(train_df)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                IsolationForest(
                    n_estimators=250,
                    contamination=contamination,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(train_df[feature_columns])
    return pipeline, feature_columns


def anomaly_scores(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    raw = -model.named_steps["model"].score_samples(
        model.named_steps["preprocess"].transform(df[feature_columns])
    )
    low, high = np.percentile(raw, [2, 98])
    return np.clip((raw - low) / max(high - low, 1e-6), 0, 1)


def save_baseline(model: Pipeline, feature_columns: list[str]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns}, MODEL_DIR / "baseline_isolation_forest.joblib")
