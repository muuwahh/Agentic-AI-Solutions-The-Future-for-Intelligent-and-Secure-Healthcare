"""Metrics and plots for held-out model predictions."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.config import CLASS_NAMES, MODEL_VERSION, RESULTS_DIR


def calculate_metrics(y_true, probabilities):
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, predictions, labels=np.arange(len(CLASS_NAMES)))
    true_positive = np.diag(matrix)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    true_negative = matrix.sum() - (true_positive + false_positive + false_negative)
    specificity = np.divide(
        true_negative,
        true_negative + false_positive,
        out=np.zeros_like(true_negative, dtype=float),
        where=(true_negative + false_positive) != 0,
    )
    try:
        roc_auc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = None
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "specificity": float(np.mean(specificity)),
        "f1_score": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
        "support": int(len(y_true)),
    }


def save_evaluation_artifacts(y_true, probabilities, history, dataset_name="Synthetic Demonstration Data"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = calculate_metrics(y_true, probabilities)
    metrics_path = RESULTS_DIR / "model_results.json"
    report = {
        "model_version": MODEL_VERSION,
        "dataset": dataset_name,
        "data_notice": "SYNTHETIC DEMONSTRATION DATA; classes are arbitrary benchmark labels, not medical outcomes.",
        **metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, history["train_loss"], label="Train")
    axis.plot(epochs, history["validation_loss"], label="Validation")
    axis.set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, history["train_accuracy"], label="Train")
    axis.plot(epochs, history["validation_accuracy"], label="Validation")
    axis.set(title="Training and validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(title="Held-out confusion matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    y_array = np.asarray(y_true)
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_truth = (y_array == class_index).astype(int)
        if binary_truth.min() == binary_truth.max():
            continue
        false_positive, true_positive, _ = roc_curve(binary_truth, probabilities[:, class_index])
        axis.plot(false_positive, true_positive, label=class_name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#707070", linewidth=1)
    axis.set(title="One-vs-rest ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close(figure)
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    from ml.inference.predict import load_pipeline
    from ml.training.train import load_test_split, predict_probabilities

    model, artifacts, config = load_pipeline()
    _, test_features, test_labels = load_test_split(artifacts, config["seed"])
    probabilities = predict_probabilities(model, test_features, config["batch_size"])
    result = save_evaluation_artifacts(test_labels, probabilities, config["history"])
    print(json.dumps({key: value for key, value in result.items() if key in {
        "accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"
    }}, indent=2))