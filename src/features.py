from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SENSOR_COLUMNS, WINDOW_SIZE
from .preprocess import sensor_columns_present


def _spectral_energy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return 0.0
    centered = values - np.nanmean(values)
    spectrum = np.fft.rfft(centered)
    return float(np.sum(np.abs(spectrum[1:]) ** 2) / len(values))


def build_time_series_features(df: pd.DataFrame, window: int = WINDOW_SIZE) -> pd.DataFrame:
    out = df.sort_values(["unit_id", "cycle"]).copy()
    sensors = sensor_columns_present(out)
    grouped = out.groupby("unit_id", group_keys=False)

    for sensor in sensors:
        out[f"{sensor}_delta"] = grouped[sensor].diff().fillna(0)
        out[f"{sensor}_roll_mean"] = grouped[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=3).mean()
        )
        out[f"{sensor}_roll_std"] = grouped[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=3).std()
        )
        out[f"{sensor}_trend"] = grouped[sensor].transform(
            lambda x: x.diff().rolling(window=window, min_periods=3).mean()
        )

    spectral_sensors = sensors[:5]
    for sensor in spectral_sensors:
        out[f"{sensor}_spectral_energy"] = grouped[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=8).apply(_spectral_energy, raw=True)
        )

    feature_columns = get_feature_columns(out)
    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"unit_id", "cycle", "rul", "is_anomaly", "event_type", "source_dataset"}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if column not in excluded]
