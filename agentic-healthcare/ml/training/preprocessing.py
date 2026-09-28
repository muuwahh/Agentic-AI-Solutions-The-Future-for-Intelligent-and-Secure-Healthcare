"""Train-only preprocessing and deterministic stratified data splits."""

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.config import (
    FEATURE_COLUMNS,
    SEED,
    SELECTED_FEATURE_COUNT,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def split_indices(labels, seed=SEED):
    indices = np.arange(len(labels))
    train_validation, test = train_test_split(
        indices, test_size=TEST_SIZE, random_state=seed, stratify=labels
    )
    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        random_state=seed,
        stratify=np.asarray(labels)[train_validation],
    )
    return {"train": train, "validation": validation, "test": test}


def fit_preprocessor(train_frame, selected_feature_count=SELECTED_FEATURE_COUNT):
    feature_names = list(train_frame.columns)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_frame)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    selector = RFE(
        estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        n_features_to_select=min(selected_feature_count, len(feature_names)),
        step=1,
    )
    selector.fit(train_scaled, np.asarray(train_frame.attrs["labels"]))
    selected_names = [
        name for name, keep in zip(feature_names, selector.support_) if keep
    ]
    artifacts = {
        "feature_names": feature_names,
        "selected_features": selected_names,
        "imputer": imputer,
        "scaler": scaler,
        "selector": selector,
        "reference_values": imputer.statistics_.tolist(),
    }
    return artifacts


def transform_frame(frame, artifacts):
    aligned = frame.reindex(columns=artifacts["feature_names"])
    imputed = artifacts["imputer"].transform(aligned)
    scaled = artifacts["scaler"].transform(imputed)
    return artifacts["selector"].transform(scaled).astype(np.float32)


def fit_transform_splits(frame, labels, split_map):
    train_frame = frame.iloc[split_map["train"]].copy()
    train_frame.attrs["labels"] = np.asarray(labels)[split_map["train"]]
    artifacts = fit_preprocessor(train_frame)
    transformed = {
        name: transform_frame(frame.iloc[indices], artifacts)
        for name, indices in split_map.items()
    }
    return artifacts, transformed


def expected_feature_columns():
    return list(FEATURE_COLUMNS)