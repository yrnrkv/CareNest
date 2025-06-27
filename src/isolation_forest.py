
---

## src/isolation_forest.py

```python
#!/usr/bin/env python3
# src/isolation_forest.py

"""
Isolation Forest Anomaly Detection for Incubator Data
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import joblib

# ─── Config ────────────────────────────────────────────────────────────────────

DATA_PATH       = Path(__file__).parent.parent / "data" / "incubator_readings.csv"
MODEL_PATH      = Path(__file__).parent.parent / "models" / "isolation_forest.joblib"
PLOT_PATH       = Path(__file__).parent.parent / "outputs" / "anomaly_plot.png"
WINDOW_SIZE     = 60    # seconds
CONTAMINATION   = 0.05  # expected anomaly rate
SEED            = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Functions ────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    logging.info("Loading data from %s", path)
    df = pd.read_csv(path, parse_dates=["Timestamp"], index_col="Timestamp")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling stats, diffs, and normalization."""
    logging.info("Engineering features")
    df = df.copy()
    sensors = [f"sensor{i}_temp" for i in range(1,5)]

    # Rolling mean & std
    for s in sensors:
        df[f"{s}_mean"] = df[s].rolling(WINDOW_SIZE).mean()
        df[f"{s}_std"]  = df[s].rolling(WINDOW_SIZE).std()

    df.bfill(inplace=True)

    # Differencing
    for s in sensors:
        df[f"{s}_diff"] = df[s].diff().fillna(0)

    # Normalization
    for s in sensors:
        df[f"{s}_norm"] = (df[s] - df[s].mean()) / df[s].std()

    return df.dropna()


def train_if(df: pd.DataFrame) -> IsolationForest:
    """Train and save Isolation Forest."""
    features = [c for c in df.columns if any(k in c for k in ["_temp","_mean","_std","_diff","_norm"])]
    X = df[features]

    logging.info("Training Isolation Forest (contamination=%.2f)", CONTAMINATION)
    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X)
    joblib.dump(model, MODEL_PATH)
    logging.info("Model saved to %s", MODEL_PATH)
    return model


def detect_anomalies(df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    """Add anomaly column (1=anomaly)."""
    logging.info("Detecting anomalies")
    features = [c for c in df.columns if any(k in c for k in ["_temp","_mean","_std","_diff","_norm"])]
    df["anomaly"] = model.predict(df[features])
    df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})
    logging.info("Found %d anomalies", int(df["anomaly"].sum()))
    return df


def plot_anomalies(df: pd.DataFrame], save_path: Path):
    """Plot temperature series with anomaly markers."""
    logging.info("Plotting anomalies to %s", save_path)
    plt.figure(figsize=(14,8))
    for i in range(1,5):
        plt.plot(df.index, df[f"sensor{i}_temp"], label=f"Sensor {i}")
    anomalies = df[df["anomaly"]==1]
    plt.scatter(anomalies.index, anomalies["sensor1_temp"], color="red", marker="x", label="Anomaly")
    plt.legend()
    plt.title("Sensor Temperatures & Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_data(DATA_PATH)
    df_feat = engineer_features(df)
    model = train_if(df_feat)
    df_ann = detect_anomalies(df_feat, model)

    PLOT_PATH.parent.mkdir(exist_ok=True)
    plot_anomalies(df_ann, PLOT_PATH)


if __name__ == "__main__":
    main()
