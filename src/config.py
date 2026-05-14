from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURE_DIR = OUTPUTS_DIR / "figures"

RANDOM_STATE = 42
WINDOW_SIZE = 30
ANOMALY_RUL_THRESHOLD = 30
WARNING_RUL_THRESHOLD = 60

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
INDEX_COLUMNS = ["unit_id", "cycle"]
NASA_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, METRICS_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

