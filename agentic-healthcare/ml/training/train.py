"""Train the FNN, select features, and write all dashboard-ready results."""

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CONFIG_PATH,
    DATA_PATH,
    DROPOUT,
    EPOCHS,
    FEATURE_COLUMNS,
    HIDDEN_DIMS,
    HISTORY_PATH,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_VERSION,
    PREPROCESSOR_PATH,
    RESULTS_DIR,
    SEED,
)
from ml.data.generate_synthetic_data import generate_dataset
from ml.models.healthcare_fnn import HealthcareFNN
from ml.training.evaluate import save_evaluation_artifacts
from ml.training.preprocessing import fit_transform_splits, split_indices


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(features, labels, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(np.asarray(labels), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_probabilities(model, features, batch_size=BATCH_SIZE):
    loader = DataLoader(torch.tensor(features, dtype=torch.float32), batch_size=batch_size)
    model.eval()
    batches = []
    with torch.no_grad():
        for inputs in loader:
            batches.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def load_test_split(artifacts, seed=SEED):
    frame = pd.read_csv(DATA_PATH)
    labels = frame["demo_class"].astype(int).to_numpy()
    splits = split_indices(labels, seed)
    from ml.training.preprocessing import transform_frame

    return frame, transform_frame(frame.iloc[splits["test"]][FEATURE_COLUMNS], artifacts), labels[splits["test"]]


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED):
    set_seeds(seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        generate_dataset().to_csv(DATA_PATH, index=False)
    frame = pd.read_csv(DATA_PATH)
    if "demo_class" not in frame:
        raise ValueError("Dataset must contain the target column 'demo_class'.")
    absent = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if absent:
        raise ValueError(f"Dataset is missing configured feature columns: {absent}")
    labels = frame["demo_class"].astype(int).to_numpy()
    if set(np.unique(labels)) != set(range(len(CLASS_NAMES))):
        raise ValueError("demo_class must contain the integer classes 0, 1, and 2.")

    splits = split_indices(labels, seed)
    artifacts, transformed = fit_transform_splits(frame[FEATURE_COLUMNS], labels, splits)
    y_splits = {key: labels[index] for key, index in splits.items()}
    train_loader = make_loader(transformed["train"], y_splits["train"], batch_size, True)
    validation_loader = make_loader(transformed["validation"], y_splits["validation"], batch_size, False)
    model = HealthcareFNN(
        input_dim=transformed["train"].shape[1],
        hidden_dims=HIDDEN_DIMS,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_truth, train_predictions = [], []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(targets)
            train_truth.extend(targets.numpy())
            train_predictions.extend(logits.argmax(dim=1).detach().numpy())

        model.eval()
        validation_loss_sum = 0.0
        validation_truth, validation_predictions = [], []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                logits = model(inputs)
                validation_loss_sum += criterion(logits, targets).item() * len(targets)
                validation_truth.extend(targets.numpy())
                validation_predictions.extend(logits.argmax(dim=1).numpy())

        train_loss = train_loss_sum / len(train_loader.dataset)
        validation_loss = validation_loss_sum / len(validation_loader.dataset)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(accuracy_score(train_truth, train_predictions))
        history["validation_accuracy"].append(accuracy_score(validation_truth, validation_predictions))
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | train loss {train_loss:.4f} | "
            f"validation loss {validation_loss:.4f} | train accuracy {history['train_accuracy'][-1]:.3f} | "
            f"validation accuracy {history['validation_accuracy'][-1]:.3f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    import joblib

    joblib.dump(artifacts, PREPROCESSOR_PATH)
    with HISTORY_PATH.open("w") as file:
        json.dump(history, file, indent=2)
    model_config = {
        "model_version": MODEL_VERSION,
        "input_dim": len(artifacts["selected_features"]),
        "hidden_dims": HIDDEN_DIMS,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "dropout": DROPOUT,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
        "selected_features": artifacts["selected_features"],
        "history": history,
    }
    CONFIG_PATH.write_text(json.dumps(model_config, indent=2) + "\n")

    _, test_features, test_labels = load_test_split(artifacts, seed)
    test_probabilities = predict_probabilities(model, test_features, batch_size)
    report = save_evaluation_artifacts(test_labels, test_probabilities, history)
    print("\nHeld-out test metrics (generated by the trained model):")
    for key in ("accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"):
        print(f"  {key}: {report[key]}")
    print(f"  test rows: {len(test_labels)} | selected features: {', '.join(artifacts['selected_features'])}")

    from ml.anomaly.anomaly_detector import train_detector
    from ml.inference.predict import run_prediction

    train_detector()
    run_prediction()
    return model, artifacts, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.learning_rate, args.seed)