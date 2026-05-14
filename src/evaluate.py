from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import METRICS_DIR, MODEL_DIR, PROCESSED_DIR, ensure_directories
from .data_loader import load_dataset
from .decision_support import explain_row
from .features import build_time_series_features, get_feature_columns
from .preprocess import add_targets
from .train_advanced import save_advanced, train_event_classifier, train_signal_importance_model
from .train_baseline import anomaly_scores, save_baseline, train_isolation_forest
from .visualize import plot_confusion_matrix, plot_feature_importance, plot_signal_with_scores


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate_pipeline(prefer_nasa: bool = True) -> dict:
    ensure_directories()
    bundle = load_dataset(prefer_nasa=prefer_nasa)
    train = build_time_series_features(add_targets(bundle.train))
    validation = build_time_series_features(add_targets(bundle.validation))
    test = build_time_series_features(add_targets(bundle.test))

    train.to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    validation.to_csv(PROCESSED_DIR / "validation_features.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test_features.csv", index=False)

    baseline, baseline_features = train_isolation_forest(train)
    validation_scores = anomaly_scores(baseline, validation, baseline_features)
    threshold, validation_f1 = tune_threshold(validation["is_anomaly"].to_numpy(), validation_scores)
    test_scores = anomaly_scores(baseline, test, baseline_features)
    baseline_predictions = (test_scores >= threshold).astype(int)

    save_baseline(baseline, baseline_features)
    advanced, advanced_features = train_event_classifier(train)
    save_advanced(advanced, advanced_features)
    importance_model = train_signal_importance_model(train, advanced_features)

    event_predictions = advanced.predict(test[advanced_features])
    probabilities = advanced.predict_proba(test[advanced_features])
    class_order = list(advanced.named_steps["model"].classes_)
    failure_indices = [idx for idx, name in enumerate(class_order) if name != "normal_operation"]
    probability_failure = probabilities[:, failure_indices].sum(axis=1) if failure_indices else np.zeros(len(test))

    labels = ["normal_operation", "degradation_warning", "critical_failure_risk"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        test["is_anomaly"], baseline_predictions, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(test["is_anomaly"], test_scores)
    except ValueError:
        roc_auc = float("nan")
    pr_auc = average_precision_score(test["is_anomaly"], test_scores)

    feature_columns = get_feature_columns(train)
    baseline_medians = train[feature_columns].median().to_dict()
    baseline_stds = train[feature_columns].std().replace(0, 1).to_dict()

    alert_rows = test.assign(anomaly_score=test_scores, probability_failure=probability_failure)
    alert_rows = alert_rows.sort_values(["probability_failure", "anomaly_score"], ascending=False).head(8)
    predictions_by_index = pd.Series(event_predictions, index=test.index)
    alerts = []
    for index, row in alert_rows.iterrows():
        alerts.append(
            explain_row(
                row,
                baseline_medians,
                baseline_stds,
                anomaly_score=float(row["anomaly_score"]),
                probability_failure=float(row["probability_failure"]),
                predicted_event_type=str(predictions_by_index.loc[index]),
            ).__dict__
        )

    test_with_scores = test.copy()
    test_with_scores["anomaly_score"] = test_scores
    test_with_scores["predicted_event_type"] = event_predictions
    test_with_scores["probability_failure"] = probability_failure
    test_with_scores.to_csv(PROCESSED_DIR / "test_scored.csv", index=False)

    importances = pd.DataFrame(
        {"feature": advanced_features, "importance": importance_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(METRICS_DIR / "feature_importance.csv", index=False)

    figure_paths = {
        "signal_alert_plot": plot_signal_with_scores(test_with_scores),
        "confusion_matrix": plot_confusion_matrix(test["event_type"], event_predictions, labels),
        "feature_importance": plot_feature_importance(importances),
    }

    metrics = {
        "dataset_source": bundle.source,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "baseline_isolation_forest": {
            "threshold": threshold,
            "validation_f1_at_threshold": validation_f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc,
            "false_positive_rate": float(
                confusion_matrix(test["is_anomaly"], baseline_predictions, labels=[0, 1])[0, 1]
                / max((test["is_anomaly"] == 0).sum(), 1)
            ),
        },
        "advanced_event_classifier": {
            "macro_f1": f1_score(test["event_type"], event_predictions, average="macro"),
            "classification_report": classification_report(
                test["event_type"], event_predictions, labels=labels, output_dict=True, zero_division=0
            ),
        },
        "top_alert_examples": alerts,
        "figures": figure_paths,
    }

    with (METRICS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    joblib.dump(
        {
            "baseline_medians": baseline_medians,
            "baseline_stds": baseline_stds,
            "threshold": threshold,
            "class_order": class_order,
        },
        MODEL_DIR / "decision_support_context.joblib",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate signal anomaly detection system.")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic demo data instead of NASA raw files.")
    args = parser.parse_args()
    metrics = evaluate_pipeline(prefer_nasa=not args.synthetic)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
