from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Alert:
    unit_id: int
    cycle: int
    anomaly_score: float
    probability_failure: float
    predicted_event_type: str
    severity: str
    recommended_action: str
    explanation: str
    contributing_signals: list[str]


def severity_from_score(anomaly_score: float, probability_failure: float) -> str:
    risk = max(anomaly_score, probability_failure)
    if risk >= 0.85:
        return "critical"
    if risk >= 0.65:
        return "high"
    if risk >= 0.4:
        return "medium"
    return "low"


def action_from_severity(severity: str) -> str:
    return {
        "critical": "escalate and inspect immediately",
        "high": "inspect during next maintenance window",
        "medium": "monitor closely and compare against recent trend",
        "low": "continue routine monitoring",
    }[severity]


def explain_row(
    row: pd.Series,
    baseline_medians: dict[str, float],
    baseline_stds: dict[str, float],
    anomaly_score: float,
    probability_failure: float,
    predicted_event_type: str,
    top_n: int = 4,
) -> Alert:
    z_scores: list[tuple[str, float]] = []
    for column, median in baseline_medians.items():
        if column not in row:
            continue
        std = max(float(baseline_stds.get(column, 1.0)), 1e-6)
        z_scores.append((column, abs(float(row[column]) - median) / std))

    top = sorted(z_scores, key=lambda item: item[1], reverse=True)[:top_n]
    contributing = [name for name, _ in top]
    severity = severity_from_score(anomaly_score, probability_failure)
    action = action_from_severity(severity)
    signal_text = ", ".join(contributing) if contributing else "no dominant signal"
    explanation = (
        f"{predicted_event_type.replace('_', ' ')} detected with {severity} severity. "
        f"Primary contributors: {signal_text}. "
        f"Combined model risk={max(anomaly_score, probability_failure):.2f}; recommended action: {action}."
    )
    return Alert(
        unit_id=int(row.get("unit_id", -1)),
        cycle=int(row.get("cycle", -1)),
        anomaly_score=float(np.clip(anomaly_score, 0, 1)),
        probability_failure=float(np.clip(probability_failure, 0, 1)),
        predicted_event_type=predicted_event_type,
        severity=severity,
        recommended_action=action,
        explanation=explanation,
        contributing_signals=contributing,
    )
