# Agentic Healthcare Research Prototype

A local research and education prototype that trains a PyTorch feedforward classifier and a scikit-learn Isolation Forest. It has no server, API, database, or clinical function. It is not a medical device; predictions, anomaly flags, and explanations are not diagnoses or clinical advice.

## Data

`ml/data/synthetic_healthcare.csv` is **SYNTHETIC DEMONSTRATION DATA**, not real patients and not MIMIC-III, PAMAP2, or TCIA. `demo_class` is an arbitrary synthetic benchmark label generated to exercise multiclass training; it is not a health outcome, medical threshold, or clinical risk category. Replace this CSV with an appropriately governed dataset using the same configured numerical feature columns and integer target classes `0`, `1`, and `2`. Edit `ml/config.py` and the target handling for a different schema. Missing numerical values are imputed; the imputer, standard scaler, and RFE selector are fitted only on the training split.

## Run

From this directory, install the Python dependencies and train the model:

```bash
python -m pip install -r requirements.txt
python -m ml.data.generate_synthetic_data
python -m ml.training.train
```

Training prints per-epoch train/validation loss and accuracy, plus actual held-out metrics. It saves the FNN state, fitted preprocessing objects, model configuration, and training history under `ml/saved_models/`; plots and JSON results are written under `ml/results/`. Training also fits the wearable anomaly detector and writes example outputs using rows from the synthetic dataset.

Run inference without retraining, or score a supplied JSON object:

```bash
python -m ml.inference.predict
python -m ml.inference.predict --input '{"age": 55, "heart_rate": 72, "spo2": 97, "temperature": 36.8, "systolic_bp": 120, "diastolic_bp": 78, "lab_marker_a": 0.2, "lab_marker_b": -0.1, "history_indicator_a": 0, "history_indicator_b": 1, "motion": 0.4, "sleep_hours": 7}'
python -m ml.anomaly.anomaly_detector --input '{"heart_rate": 72, "spo2": 97, "temperature": 36.8, "motion": 0.4}'
```

The feature-importance output uses single-feature replacement with training medians. It describes changes to model probabilities, not causes or medical reasoning. Isolation Forest anomaly scores indicate unusualness relative to its training data only.

## Frontend

The React + Vite dashboard reads the JSON and image files emitted by the Python pipeline. Train the Python models before starting the frontend; the frontend does not generate or simulate predictions. Start it with `npm install` and `npm run dev` from `frontend/`.

## Imaging

Medical imaging module: Not implemented. See `ml/imaging/README.md` for future integration notes.