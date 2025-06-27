#!/usr/bin/env python3
# care_nest.py
"""
CareNest – Incubator Monitoring System
- Synthetic data generation
- Feature engineering
- Model training & evaluation
- Real-time prediction & alerting
- Dashboard data prep
- Temperature-range analysis
"""

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib

# ─── Configuration ─────────────────────────────────────────────────────────────

DATA_CYCLES         = 5
CSV_OUTPUT_PATH     = Path("incubator_sensor_data.csv")
MODEL_OUTPUT_PATH   = Path("incubator_model.pkl")
CM_PLOT_PATH        = Path("confusion_matrix.png")
FI_PLOT_PATH        = Path("feature_importance.png")
SAMPLE_SIZE         = 200_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ─── Synthetic Data Generation ──────────────────────────────────────────────────

def generate_synthetic_data(cycles: int = DATA_CYCLES) -> pd.DataFrame:
    """Simulate sensor readings across temperature cycles with random failure modes."""
    logging.info("Starting data generation (%d cycles)", cycles)
    np.random.seed(42)
    records = []

    temp_range = np.arange(23.0, 38.5, 0.5)
    for cycle in range(cycles):
        for idx, target in enumerate(temp_range):
            for minute in range(60):
                for second in range(60):
                    # ambient conditions
                    amb_t = np.random.normal(22, 2)
                    amb_h = np.random.normal(50, 8)
                    # determine failure mode
                    p_fail = 0.15 if (target < 28 or target > 36) else 0.05
                    mode = (
                        np.random.choice(
                            ["heater_issue","sensor_drift","air_circulation"],
                            p=[0.6,0.2,0.2]
                        )
                        if np.random.rand() < p_fail
                        else "none"
                    )

                    base = {
                        "cycle": cycle,
                        "hour": cycle * len(temp_range) + idx,
                        "minute": minute,
                        "second": second,
                        "set_temp": target,
                        "ambient_temp": amb_t,
                        "ambient_humidity": amb_h,
                        "failure_mode": mode
                    }

                    # per-sensor readings
                    for sid in range(1,5):
                        bias = (sid - 2.5) * 0.15
                        adj = min(1.0, (minute*60+second)/300)
                        if mode == "none":
                            temp = target*adj + amb_t*0.1*(1-adj) + bias + np.random.normal(0,0.1)
                            hum  = 80 + np.random.normal(0,1)
                        else:
                            # simplified faulty scenarios
                            delta = np.random.uniform(1.5,2.5) if mode=="heater_issue" else 0
                            temp = (target - delta)*adj + amb_t*0.1*(1-adj) + bias + np.random.normal(0,0.2)
                            hum  = 80 + np.random.normal(0,1.5)
                        base[f"sensor{sid}_temp"]     = temp
                        base[f"sensor{sid}_humidity"] = hum

                    records.append(base)

        logging.info("Completed cycle %d, total rows: %d", cycle+1, len(records))

    df = pd.DataFrame(records)
    logging.info("Generated %d rows (expected ~%d)", len(df),
                 cycles*len(temp_range)*60*60)
    return df

# ─── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based, statistical, difference, and one-hot features."""
    logging.info("Engineering features")
    df = df.copy()
    df["time_of_day"]  = df["hour"] % 24
    df["time_in_cycle"] = df["minute"]/60 + df["second"]/3600
    df["is_night"]     = df["time_of_day"].between(22,23) | df["time_of_day"].between(0,6)

    # sensor columns
    temp_cols = [c for c in df if c.startswith("sensor") and c.endswith("_temp")]
    hum_cols  = [c for c in df if c.startswith("sensor") and c.endswith("_humidity")]

    # basic stats
    df["mean_temp"] = df[temp_cols].mean(axis=1)
    df["std_temp"]  = df[temp_cols].std(axis=1)
    df["mean_hum"]  = df[hum_cols].mean(axis=1)
    df["std_hum"]   = df[hum_cols].std(axis=1)

    # diffs to set and ambient
    for c in temp_cols:
        df[f"{c}_diff_set"] = df[c] - df["set_temp"]
        df[f"{c}_diff_amb"] = df[c] - df["ambient_temp"]
    for c in hum_cols:
        df[f"{c}_diff_amb"] = df[c] - df["ambient_humidity"]

    # temperature range bucket
    ranges = [22.5, 28.0, 32.0, 36.0, 38.5]
    labels = ["low","med_low","med_high","high"]
    df["temp_range"] = pd.cut(df["set_temp"], bins=ranges, labels=labels)
    df = pd.get_dummies(df, columns=["temp_range"], prefix="tr")

    return df

# ─── Model Training & Evaluation ────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    """Sample, split, train RF model, and save outputs (metrics & figures)."""
    logging.info("Preparing data for training")
    df_feat = engineer_features(df)

    # sample to speed up
    if len(df_feat) > SAMPLE_SIZE:
        df_sample = (
            df_feat
            .groupby(["failure_mode"])
            .apply(lambda g: g.sample(n=min(len(g), SAMPLE_SIZE//4), random_state=42))
            .reset_index(drop=True)
        )
        logging.info("Sampled down to %d rows", len(df_sample))
    else:
        df_sample = df_feat

    X = df_sample.drop(columns=["failure_mode","cycle","hour","minute","second"])
    y = df_sample["failure_mode"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logging.info("Training RandomForestClassifier")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=15,
        min_samples_split=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # metrics
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    logging.info("Test Accuracy: %.4f", acc)
    logging.info("\n%s", classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.savefig(CM_PLOT_PATH)
    plt.close()

    # feature importance
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(6,8))
    sns.barplot(data=fi, x="importance", y="feature")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(FI_PLOT_PATH)
    plt.close()

    # save model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logging.info("Model and plots saved")

    return model

# ─── Real-Time Prediction (single-reading) ──────────────────────────────────────

def predict_maintenance(sensor_readings: dict, model: RandomForestClassifier) -> dict:
    """
    Given one reading dict, engineer features and predict.
    Returns: {prediction, confidence, recommendation, probabilities}
    """
    # minimal feature-engineering inline (for brevity)...
    df_r = pd.DataFrame([sensor_readings])
    df_feat = engineer_features(df_r)
    # ensure same columns as training
    missing = set(model.feature_names_in_) - set(df_feat.columns)
    for c in missing:
        df_feat[c] = 0
    df_feat = df_feat[model.feature_names_in_]

    pred  = model.predict(df_feat)[0]
    probs = model.predict_proba(df_feat)[0]
    top  = dict(zip(model.classes_, probs))
    conf = probs.max()

    recs = {
        "heater_issue":   "Check heating element & calibration.",
        "sensor_drift":   "Re-calibrate drifting sensors.",
        "air_circulation":"Inspect fans & vents.",
        "none":           "All systems nominal."
    }
    return {
        "prediction": pred,
        "confidence": conf,
        "recommendation": recs.get(pred, ""),
        "probabilities": top
    }

# ─── Main Entrypoint ────────────────────────────────────────────────────────────

def main():
    df = generate_synthetic_data()
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    logging.info("Data saved to %s", CSV_OUTPUT_PATH)

    model = train_and_evaluate(df)

    # example real-time
    example = {
        "cycle":0, "hour":0, "minute":0, "second":0,
        "set_temp":37.0, "ambient_temp":25.0, "ambient_humidity":45.0,
        **{f"sensor{i}_temp":37.0 for i in range(1,5)},
        **{f"sensor{i}_humidity":79.0 for i in range(1,5)}
    }
    res = predict_maintenance(example, model)
    logging.info("Example prediction: %s", res)

if __name__ == "__main__":
    main()
