"""Local software orchestration demo; it makes no clinical decisions."""

import argparse
import json

from ml.anomaly.anomaly_detector import run_anomaly
from ml.config import RESULTS_DIR
from ml.inference.predict import run_prediction


def run_workflow(patient_input, wearable_input):
    processed_patient = {key: value for key, value in patient_input.items()}
    prediction = run_prediction(processed_patient)
    anomaly = run_anomaly(wearable_input)
    result = {
        "workflow": "local research demonstration",
        "steps": [
            "Autonomous Data Processing",
            "Adaptive Decision Support",
            "Predictive Analytics",
        ],
        "model_selection": "FNN classification and Isolation Forest anomaly detection",
        "patient_prediction": prediction,
        "wearable_anomaly": anomaly,
        "medical_imaging": "Not implemented",
        "disclaimer": "Software workflow demonstration only; not autonomous clinical decision-making.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest_workflow.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", required=True, help="JSON object of patient feature values")
    parser.add_argument("--wearable", required=True, help="JSON object of wearable feature values")
    args = parser.parse_args()
    result = run_workflow(json.loads(args.patient), json.loads(args.wearable))
    print(json.dumps(result, indent=2))