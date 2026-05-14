from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR, PROCESSED_DIR
from .decision_support import Alert, explain_row
from .features import build_time_series_features
from .preprocess import add_targets
from .train_baseline import anomaly_scores


def load_artifacts() -> dict:
    return {
        "baseline": joblib.load(MODEL_DIR / "baseline_isolation_forest.joblib"),
        "advanced": joblib.load(MODEL_DIR / "advanced_event_classifier.joblib"),
        "context": joblib.load(MODEL_DIR / "decision_support_context.joblib"),
    }


def score_frame(raw_df: pd.DataFrame, artifacts: dict | None = None) -> pd.DataFrame:
    artifacts = artifacts or load_artifacts()
    df = build_time_series_features(add_targets(raw_df.copy()) if "rul" in raw_df else raw_df.copy())

    baseline = artifacts["baseline"]
    advanced = artifacts["advanced"]
    scores = anomaly_scores(baseline["model"], df, baseline["feature_columns"])
    probabilities = advanced["model"].predict_proba(df[advanced["feature_columns"]])
    event_predictions = advanced["model"].predict(df[advanced["feature_columns"]])
    class_order = list(advanced["model"].named_steps["model"].classes_)
    failure_indices = [idx for idx, name in enumerate(class_order) if name != "normal_operation"]

    out = df.copy()
    out["anomaly_score"] = scores
    out["predicted_event_type"] = event_predictions
    out["probability_failure"] = probabilities[:, failure_indices].sum(axis=1) if failure_indices else 0.0
    return out


def latest_alert_for_unit(unit_df: pd.DataFrame, artifacts: dict | None = None) -> Alert:
    artifacts = artifacts or load_artifacts()
    scored = score_frame(unit_df, artifacts)
    latest = scored.sort_values("cycle").iloc[-1]
    return explain_row(
        latest,
        artifacts["context"]["baseline_medians"],
        artifacts["context"]["baseline_stds"],
        float(latest["anomaly_score"]),
        float(latest["probability_failure"]),
        str(latest["predicted_event_type"]),
    )


def score_saved_test_sample() -> dict:
    sample_path = PROCESSED_DIR / "test.csv"
    if not sample_path.exists():
        raise FileNotFoundError("Run python -m src.evaluate --synthetic first to create processed test data.")
    df = pd.read_csv(sample_path)
    unit_id = int(df["unit_id"].iloc[0])
    alert = latest_alert_for_unit(df[df["unit_id"] == unit_id])
    return alert.__dict__


if __name__ == "__main__":
    print(json.dumps(score_saved_test_sample(), indent=2))
