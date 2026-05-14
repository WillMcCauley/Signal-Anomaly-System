# Signal Anomaly Detection and Decision Support System

An end-to-end AI system for noisy multivariate sensor streams. The project frames turbofan engine monitoring as an operational signal problem: ingest sensor time series, engineer rolling and spectral features, detect anomalies, classify likely degradation states, and translate model output into clear alert recommendations.

This is designed as a portfolio-grade project for the narrative:

```text
mission-critical signal environments -> AI for noisy data -> robust decision support
```

## Why It Matters

Real monitoring systems do not just need an anomaly score. Operators need to know what changed, how severe the risk is, and what to do next. This project emphasizes that bridge from model output to decision support, which makes it relevant to sonar, industrial monitoring, aerospace systems, predictive maintenance, and other high-noise technical environments.

## Dataset

Primary target dataset: NASA C-MAPSS turbofan engine degradation data.

The pipeline automatically uses NASA `train_FD001.txt` if it exists in `data/raw/`. If that file is not present, it uses a deterministic synthetic turbofan-style dataset so the full system remains runnable for demos, tests, and portfolio screenshots.

Download helper:

```bash
cd signal_anomaly_detection
python scripts/download_nasa_cmapps.py
```

Then train with the real raw file:

```bash
python -m src.evaluate
```

Run the reproducible demo data path:

```bash
python -m src.evaluate --synthetic
```

## Architecture

```text
data/raw -> data_loader -> preprocessing -> feature engineering
                                      -> baseline anomaly detector
                                      -> advanced event classifier
                                      -> decision support layer
                                      -> API / dashboard / figures
```

Core modules:

- `src/data_loader.py`: NASA C-MAPSS loader plus synthetic turbofan generator.
- `src/preprocess.py`: target construction, missing-value handling, normalization.
- `src/features.py`: rolling mean, rolling standard deviation, deltas, trend, and spectral energy features.
- `src/train_baseline.py`: Isolation Forest anomaly detector.
- `src/train_advanced.py`: gradient-boosted event classifier and random forest signal-importance model.
- `src/decision_support.py`: severity labels, likely issue type, contributing signals, and recommended actions.
- `src/evaluate.py`: full training, threshold tuning, metrics, saved artifacts, and plots.
- `src/inference.py`: reusable scoring functions for API/dashboard workflows.
- `app/main.py`: FastAPI service.
- `app/dashboard.py`: Streamlit decision-support dashboard.

## Modeling Approach

The project compares two model layers:

1. **Baseline anomaly model**
   - Isolation Forest
   - Uses engineered time-series features
   - Threshold tuned on validation F1
   - Evaluated with precision, recall, F1, ROC-AUC, PR-AUC, and false positive rate

2. **Advanced event model**
   - Histogram gradient boosting classifier
   - Predicts:
     - `normal_operation`
     - `degradation_warning`
     - `critical_failure_risk`
   - Uses remaining useful life windows to create operational labels
   - Evaluated with macro F1 and class-level precision/recall

## Decision Support Layer

Each alert includes:

- anomaly score
- failure probability
- predicted event type
- severity: `low`, `medium`, `high`, or `critical`
- contributing signals based on deviation from healthy baseline behavior
- recommended action:
  - continue routine monitoring
  - monitor closely and compare against recent trend
  - inspect during next maintenance window
  - escalate and inspect immediately

Example output:

```json
{
  "severity": "critical",
  "recommended_action": "escalate and inspect immediately",
  "predicted_event_type": "critical_failure_risk",
  "contributing_signals": ["sensor_2_roll_mean", "sensor_11_trend", "sensor_4_delta"]
}
```

## Evaluation Outputs

Running `python -m src.evaluate --synthetic` creates:

- `outputs/metrics/metrics.json`
- `outputs/metrics/feature_importance.csv`
- `outputs/figures/unit_<id>_signal_alerts.svg`
- `outputs/figures/event_confusion_matrix.svg`
- `outputs/figures/feature_importance.svg`
- trained models in `outputs/models/`
- processed data in `data/processed/`

The evaluation includes:

- validation threshold tuning
- anomaly precision/recall/F1
- ROC-AUC and PR-AUC
- false positive rate
- event classification macro F1
- confusion matrix
- top alert examples with explanations

## API Demo

Train first:

```bash
python -m src.evaluate --synthetic
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/sample-alert`
- `http://127.0.0.1:8000/docs`

## Dashboard Demo

```bash
streamlit run app/dashboard.py
```

The dashboard shows a selected unit's signal trajectory, anomaly score, current severity, recommended action, and explanation.

## Suggested Resume Bullets

- Designed and developed an AI-based anomaly detection and decision-support system for noisy multivariate time-series data, combining feature engineering, anomaly scoring, and alert interpretation for operational monitoring.
- Trained and evaluated baseline Isolation Forest and advanced gradient-boosted classification models on turbofan-style sensor data, comparing precision, recall, F1, ROC-AUC, PR-AUC, and false positive rate.
- Built an interpretable alert workflow with severity scoring, signal-level explanations, and recommended next actions, translating raw model output into operationally useful decisions.

## Future Improvements

- Add a PyTorch LSTM or temporal convolutional autoencoder for sequence reconstruction.
- Add SHAP explanations for classifier-level interpretability.
- Package Docker deployment for API and dashboard.
- Add streaming inference with stateful rolling windows.
- Extend the data layer to audio/spectral sonar-like signals for a stronger signal-processing bridge.
