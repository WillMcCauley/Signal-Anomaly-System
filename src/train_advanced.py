from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from .config import MODEL_DIR, RANDOM_STATE
from .features import get_feature_columns
from .preprocess import build_preprocessor


def train_event_classifier(train_df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    feature_columns = get_feature_columns(train_df)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.07,
                    max_iter=220,
                    l2_regularization=0.08,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(train_df[feature_columns], train_df["event_type"])
    return pipeline, feature_columns


def train_signal_importance_model(train_df: pd.DataFrame, feature_columns: list[str]) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train_df[feature_columns])
    model.fit(x_train, train_df["is_anomaly"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"preprocessor": preprocessor, "model": model, "feature_columns": feature_columns},
        MODEL_DIR / "signal_importance_random_forest.joblib",
    )
    return model


def save_advanced(model: Pipeline, feature_columns: list[str]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns}, MODEL_DIR / "advanced_event_classifier.joblib")
