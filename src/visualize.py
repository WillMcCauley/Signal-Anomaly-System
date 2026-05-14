from __future__ import annotations

from html import escape

import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import FIGURE_DIR


def _scale(values: pd.Series, low: float, high: float, invert: bool = True) -> list[float]:
    min_v = float(values.min())
    max_v = float(values.max())
    span = max(max_v - min_v, 1e-9)
    scaled = low + (values - min_v) / span * (high - low)
    if invert:
        return list(high - (scaled - low))
    return list(scaled)


def _polyline(xs: list[float], ys: list[float], color: str, width: int = 2) -> str:
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}" />'


def plot_signal_with_scores(df: pd.DataFrame, unit_id: int | None = None) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if unit_id is None:
        unit_id = int(df["unit_id"].iloc[0])
    unit = df[df["unit_id"] == unit_id].sort_values("cycle").copy()

    width, height = 1100, 620
    left, right = 70, 1040
    top, mid, bottom = 60, 300, 560
    xs = _scale(unit["cycle"], left, right, invert=False)
    sensor_2 = _scale(unit["sensor_2"], top, mid - 30)
    sensor_11 = _scale(unit["sensor_11"], top, mid - 30)
    scores = [bottom - float(score) * 190 for score in unit["anomaly_score"]]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8" />',
        f'<text x="{left}" y="34" font-family="Arial" font-size="22" font-weight="700">Unit {unit_id} Signal Trajectory and Alert Score</text>',
        f'<line x1="{left}" y1="{mid}" x2="{right}" y2="{mid}" stroke="#bbb" />',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#bbb" />',
        _polyline(xs, sensor_2, "#247ba0", 3),
        _polyline(xs, sensor_11, "#70a288", 3),
        _polyline(xs, scores, "#c1292e", 3),
        f'<text x="{left}" y="{mid + 35}" font-family="Arial" font-size="15" fill="#247ba0">sensor_2</text>',
        f'<text x="{left + 95}" y="{mid + 35}" font-family="Arial" font-size="15" fill="#70a288">sensor_11</text>',
        f'<text x="{left + 205}" y="{mid + 35}" font-family="Arial" font-size="15" fill="#c1292e">anomaly score</text>',
        f'<text x="{left}" y="{bottom + 38}" font-family="Arial" font-size="14">cycle</text>',
    ]
    for _, row in unit[unit["is_anomaly"] == 1].iloc[:: max(len(unit[unit["is_anomaly"] == 1]) // 12, 1)].iterrows():
        x = xs[unit.index.get_loc(row.name)]
        svg.append(f'<rect x="{x:.1f}" y="{bottom - 190}" width="3" height="190" fill="#f4a261" opacity="0.28" />')
    svg.append("</svg>")

    path = FIGURE_DIR / f"unit_{unit_id}_signal_alerts.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return str(path)


def plot_confusion_matrix(y_true, y_pred, labels: list[str]) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    cell = 135
    left, top = 270, 90
    width = left + cell * len(labels) + 80
    height = top + cell * len(labels) + 100
    max_value = max(int(matrix.max()), 1)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8" />',
        '<text x="60" y="42" font-family="Arial" font-size="22" font-weight="700">Event Classification Confusion Matrix</text>',
    ]
    for row_idx, label in enumerate(labels):
        y = top + row_idx * cell
        svg.append(f'<text x="35" y="{y + 74}" font-family="Arial" font-size="14">{escape(label)}</text>')
        for col_idx, pred_label in enumerate(labels):
            x = left + col_idx * cell
            value = int(matrix[row_idx, col_idx])
            intensity = 245 - int(150 * value / max_value)
            fill = f"rgb({intensity},{intensity + 5},245)"
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff" />')
            svg.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 8}" text-anchor="middle" font-family="Arial" font-size="26" font-weight="700">{value}</text>')
    for col_idx, label in enumerate(labels):
        x = left + col_idx * cell + cell / 2
        svg.append(f'<text x="{x}" y="{top - 18}" text-anchor="middle" font-family="Arial" font-size="13">{escape(label)}</text>')
    svg.append("</svg>")
    path = FIGURE_DIR / "event_confusion_matrix.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return str(path)


def plot_feature_importance(importances: pd.DataFrame) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    top = importances.head(15).copy()
    width, height = 1050, 620
    left, bar_left, right = 320, 340, 990
    top_y, row_h = 70, 34
    max_importance = max(float(top["importance"].max()), 1e-9)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8" />',
        '<text x="45" y="38" font-family="Arial" font-size="22" font-weight="700">Top Signal Features Contributing to Anomaly Risk</text>',
    ]
    for idx, row in enumerate(top.itertuples(index=False)):
        y = top_y + idx * row_h
        bar_width = float(row.importance) / max_importance * (right - bar_left)
        svg.append(f'<text x="{left}" y="{y + 20}" text-anchor="end" font-family="Arial" font-size="13">{escape(str(row.feature))}</text>')
        svg.append(f'<rect x="{bar_left}" y="{y}" width="{bar_width:.1f}" height="22" rx="2" fill="#247ba0" />')
        svg.append(f'<text x="{bar_left + bar_width + 8}" y="{y + 17}" font-family="Arial" font-size="12">{float(row.importance):.3f}</text>')
    svg.append("</svg>")
    path = FIGURE_DIR / "feature_importance.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return str(path)
