from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import NASA_COLUMNS, PROCESSED_DIR, RAW_DIR, RANDOM_STATE, SENSOR_COLUMNS, ensure_directories


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    source: str


def nasa_file_available(raw_dir: Path = RAW_DIR, subset: str = "FD001") -> bool:
    return (raw_dir / f"train_{subset}.txt").exists()


def load_nasa_turbofan(raw_dir: Path = RAW_DIR, subset: str = "FD001") -> pd.DataFrame:
    """Load a NASA C-MAPSS turbofan train split from raw text files."""
    path = raw_dir / f"train_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download NASA C-MAPSS and place train_{subset}.txt in data/raw."
        )

    df = pd.read_csv(path, sep=r"\s+", header=None, names=NASA_COLUMNS)
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = max_cycle - df["cycle"]
    df["source_dataset"] = f"NASA C-MAPSS {subset}"
    return df


def generate_synthetic_turbofan(
    units: int = 90,
    min_cycles: int = 130,
    max_cycles: int = 260,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a deterministic turbofan-like dataset for demos and tests.

    The synthetic data imitates degrading multivariate sensor streams: some
    channels drift upward, others lose amplitude, and late-life noise increases.
    """
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | str]] = []
    for unit_id in range(1, units + 1):
        life = int(rng.integers(min_cycles, max_cycles + 1))
        operating_shift = rng.normal(0, 0.35, size=3)
        sensor_offsets = rng.normal(0, 0.6, size=len(SENSOR_COLUMNS))
        degradation_profile = rng.uniform(0.5, 1.8, size=len(SENSOR_COLUMNS))

        for cycle in range(1, life + 1):
            progress = cycle / life
            rul = life - cycle
            periodic = np.sin(cycle / 7.0) + 0.5 * np.cos(cycle / 17.0)
            late_life_noise = 0.08 + 0.35 * max(progress - 0.65, 0)

            row: dict[str, float | int | str] = {
                "unit_id": unit_id,
                "cycle": cycle,
                "setting_1": operating_shift[0] + rng.normal(0, 0.02),
                "setting_2": operating_shift[1] + rng.normal(0, 0.02),
                "setting_3": operating_shift[2] + rng.normal(0, 0.02),
                "rul": rul,
                "source_dataset": "synthetic turbofan demo",
            }

            for idx, name in enumerate(SENSOR_COLUMNS):
                base = 10 + idx * 0.35 + sensor_offsets[idx]
                slope = degradation_profile[idx] * progress
                nonlinear = 2.5 * max(progress - 0.72, 0) ** 2
                noise = rng.normal(0, late_life_noise)
                if idx % 4 == 0:
                    value = base + 2.0 * slope + nonlinear + 0.18 * periodic + noise
                elif idx % 4 == 1:
                    value = base - 1.6 * slope - nonlinear + 0.12 * periodic + noise
                elif idx % 4 == 2:
                    value = base + 0.8 * slope + 0.45 * periodic + noise
                else:
                    value = base + 0.3 * slope + noise
                row[name] = value
            rows.append(row)

    return pd.DataFrame(rows)


def split_by_unit(
    df: pd.DataFrame,
    train_frac: float = 0.65,
    validation_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    units = np.array(sorted(df["unit_id"].unique()))
    rng.shuffle(units)

    train_end = int(len(units) * train_frac)
    validation_end = train_end + int(len(units) * validation_frac)
    train_units = set(units[:train_end])
    validation_units = set(units[train_end:validation_end])

    train = df[df["unit_id"].isin(train_units)].copy()
    validation = df[df["unit_id"].isin(validation_units)].copy()
    test = df[~df["unit_id"].isin(train_units | validation_units)].copy()
    source = str(df["source_dataset"].iloc[0]) if "source_dataset" in df else "unknown"
    return DatasetBundle(train=train, validation=validation, test=test, source=source)


def load_dataset(prefer_nasa: bool = True) -> DatasetBundle:
    ensure_directories()
    if prefer_nasa and nasa_file_available():
        df = load_nasa_turbofan()
    else:
        df = generate_synthetic_turbofan()

    bundle = split_by_unit(df)
    bundle.train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    bundle.validation.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    bundle.test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    return bundle


if __name__ == "__main__":
    dataset = load_dataset()
    print(
        f"Loaded {dataset.source}: train={dataset.train.shape}, "
        f"validation={dataset.validation.shape}, test={dataset.test.shape}"
    )
