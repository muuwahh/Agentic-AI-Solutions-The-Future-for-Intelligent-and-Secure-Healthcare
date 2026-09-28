"""Run prediction and baseline-perturbation feature attribution with saved artifacts."""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from ml.config import (
    CLASS_NAMES,
    CONFIG_PATH,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    MODEL_VERSION,
    PREPROCESSOR_PATH,
    RESULTS_DIR,
)
from ml.models.healthcare_fnn import HealthcareFNN
from ml.training.preprocessing import transform_frame


def load_pipeline():
    import joblib

    with CONFIG_PATH.open() as file:
        config = json.load(file)
    artifacts = joblib.load(PREPROCESSOR_PATH)
    model = HealthcareFNN(
        config["input_dim"], config["hidden_dims"], config["num_classes"], config["dropout"]
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, artifacts, config


def run_prediction(patient_input=None, output_path=None):
    if patient_input is None:
        patient_input = pd.read_csv(DATA_PATH).iloc[0].drop(labels=["demo_class"]).to_dict()
    model, artifacts, config = load_pipeline()
    input_features = {name: patient_input.get(name) for name in FEATURE_COLUMNS}
    frame = pd.DataFrame([input_features])
    model_input = torch.tensor(transform_frame(frame, artifacts), dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.softmax(model(model_input), dim=1)[0].numpy()
    predicted_index = int(probabilities.argmax())

    reference = pd.DataFrame([dict(zip(artifacts["feature_names"], artifacts["reference_values"]))])
    baseline_input = torch.tensor(transform_frame(reference, artifacts), dtype=torch.float32)
    with torch.no_grad():
        baseline_probability = float(torch.softmax(model(baseline_input), dim=1)[0, predicted_index])
    attributions = []
    for feature in artifacts["selected_features"]:
        perturbed = frame.copy()
        perturbed[feature] = artifacts["reference_values"][artifacts["feature_names"].index(feature)]
        inputs = torch.tensor(transform_frame(perturbed, artifacts), dtype=torch.float32)
        with torch.no_grad():
            probability = float(torch.softmax(model(inputs), dim=1)[0, predicted_index])
        attributions.append({
            "feature": feature,
            "probability_change": round(float(probabilities[predicted_index]) - probability, 6),
        })
    attributions.sort(key=lambda item: abs(item["probability_change"]), reverse=True)
    result = {
        "prediction": CLASS_NAMES[predicted_index],
        "probability": round(float(probabilities[predicted_index]), 6),
        "class_probabilities": {
            name: round(float(value), 6) for name, value in zip(CLASS_NAMES, probabilities)
        },
        "model_version": config.get("model_version", MODEL_VERSION),
        "features": {name: None if pd.isna(value) else float(value) for name, value in input_features.items()},
        "selected_features": artifacts["selected_features"],
        "important_features": attributions[:5],
        "explanation_method": "single-feature replacement with training medians; descriptive, not causal",
        "disclaimer": "Research prototype only. Synthetic benchmark classes are not diagnoses or clinical risk estimates.",
    }
    destination = output_path or RESULTS_DIR / "latest_prediction.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON object containing configured feature values")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()
    run_prediction(json.loads(args.input) if args.input else None, args.output)