"""Generate labelled synthetic demonstration data; no patient data is used."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.config import DATA_PATH, FEATURE_COLUMNS, SEED


def generate_dataset(rows=2400, seed=SEED):
    rng = np.random.default_rng(seed)
    features = {
        "age": rng.uniform(18, 90, rows),
        "heart_rate": rng.normal(76, 13, rows),
        "spo2": rng.normal(97, 1.5, rows),
        "temperature": rng.normal(36.8, 0.5, rows),
        "systolic_bp": rng.normal(122, 18, rows),
        "diastolic_bp": rng.normal(79, 12, rows),
        "lab_marker_a": rng.normal(0, 1, rows),
        "lab_marker_b": rng.normal(0, 1, rows),
        "history_indicator_a": rng.integers(0, 2, rows),
        "history_indicator_b": rng.integers(0, 2, rows),
        "motion": rng.normal(0.5, 0.2, rows),
        "sleep_hours": rng.normal(7, 1.2, rows),
    }
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)

    # Arbitrary synthetic benchmark labels, not medical risk rules or outcomes.
    scaled = (frame - frame.mean()) / frame.std()
    signal = (
        0.8 * scaled["lab_marker_a"]
        - 0.65 * scaled["sleep_hours"]
        + 0.55 * scaled["motion"]
        + 0.35 * scaled["history_indicator_a"]
        + rng.normal(0, 0.65, rows)
    )
    frame["demo_class"] = pd.qcut(signal, q=3, labels=False).astype(int)

    for column in FEATURE_COLUMNS:
        missing = rng.random(rows) < 0.025
        frame.loc[missing, column] = np.nan
    frame["data_notice"] = "SYNTHETIC DEMONSTRATION DATA"
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=str(DATA_PATH))
    args = parser.parse_args()
    output = Path(args.output)
    data = generate_dataset(args.rows, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f"Wrote {len(data)} synthetic demonstration rows to {output}")