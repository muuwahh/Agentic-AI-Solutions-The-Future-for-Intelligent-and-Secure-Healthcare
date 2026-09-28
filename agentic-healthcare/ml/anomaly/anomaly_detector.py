"""Fit and run an Isolation Forest on wearable-style synthetic observations."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ml.config import DATA_PATH, MODEL_DIR, RESULTS_DIR, SEED, WEARABLE_COLUMNS
from ml.training.preprocessing import split_indices


DETECTOR_PATH = MODEL_DIR / "wearable_isolation_forest.joblib"


def train_detector():
    import joblib

    frame = pd.read_csv(DATA_PATH)
    splits = split_indices(frame["demo_class"].astype(int).to_numpy())
    train_frame = frame.iloc[splits["train"]][WEARABLE_COLUMNS]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(train_frame)
    scaler = StandardScaler().fit(imputed)
    detector = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=SEED,
    ).fit(scaler.transform(imputed))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"imputer": imputer, "scaler": scaler, "model": detector}, DETECTOR_PATH)
    sample = frame.iloc[0][WEARABLE_COLUMNS].to_dict()
    return run_anomaly(sample)


def run_anomaly(observation, output_path=None):
    import joblib

    if not DETECTOR_PATH.exists():
        raise FileNotFoundError("Train the ML pipeline first; Isolation Forest artifact is missing.")
    artifacts = joblib.load(DETECTOR_PATH)
    values = pd.DataFrame([observation]).reindex(columns=WEARABLE_COLUMNS)
    processed = artifacts["scaler"].transform(artifacts["imputer"].transform(values))
    prediction = int(artifacts["model"].predict(processed)[0])
    score = float(-artifacts["model"].decision_function(processed)[0])
    result = {
        "status": "anomalous" if prediction == -1 else "normal",
        "anomaly_score": round(score, 6),
        "input": {key: None if pd.isna(value) else float(value) for key, value in observation.items()},
        "model": "Isolation Forest",
        "features": WEARABLE_COLUMNS,
        "disclaimer": "Anomaly detection flags unusual input patterns only; an anomaly does not imply illness.",
    }
    destination = output_path or RESULTS_DIR / "latest_anomaly.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON object with heart_rate, spo2, temperature, and motion")
    parser.add_argument("--fit", action="store_true", help="Fit and save the Isolation Forest")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()
    if args.fit:
        train_detector()
    elif args.input:
        run_anomaly(json.loads(args.input), args.output)
    else:
        parser.error("Provide --input or --fit.")