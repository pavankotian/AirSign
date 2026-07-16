# =============================================================================
# perception/gesture/trainer.py
# AirSign — Gesture Classifier Trainer
# -----------------------------------------------------------------------------
# Loads all collected landmark CSVs from data/landmarks/, trains a
# scikit-learn Pipeline (StandardScaler → MLPClassifier), evaluates it on a
# held-out split, and saves the trained model + label encoder to
# perception/gesture/models/.
#
# Design contract:
#   - Feature column order is derived exclusively from FEATURE_KEYS
#     (perception/gesture/feature_extractor.py). This module never
#     hardcodes a column layout — if FEATURE_KEYS changes, the CSV reader
#     and the trained model's input contract change together automatically.
#   - CSV files must contain exactly len(FEATURE_KEYS) + 1 columns per row:
#     the feature values in FEATURE_KEYS order, followed by the gesture
#     label string. No header row (matches data/collector.py output format).
#   - Deterministic: random_state is fixed throughout for reproducible
#     train/test splits and MLP weight initialisation.
#   - The saved label encoder is a plain sorted list of class name strings,
#     not a sklearn LabelEncoder object — this keeps classifier.py's load
#     path simple (no sklearn preprocessing class version compatibility
#     concerns across joblib save/load boundaries).
#
# Usage::
#
#   python perception/gesture/trainer.py
#
# Expected output on a full 6-class, 200-samples-per-class dataset:
#   ~93-98% test accuracy, training completes in under 5 minutes on CPU.
# =============================================================================

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from perception.gesture.feature_extractor import FEATURE_KEYS
from shared.gesture_labels import CLASSIFIABLE_GESTURES

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_DIR: str = os.path.dirname(__file__)

DEFAULT_DATA_DIR: str = os.path.join(_MODULE_DIR, "..", "..", "data", "landmarks")
DEFAULT_MODEL_DIR: str = os.path.join(_MODULE_DIR, "models")

_MODEL_FILENAME: str = "gesture_mlp.joblib"
_ENCODER_FILENAME: str = "label_encoder.joblib"

_NUM_FEATURES: int = len(FEATURE_KEYS)

# Fixed seed for deterministic train/test split and MLP weight init.
_RANDOM_STATE: int = 42

# MLP architecture — two hidden layers, sized for a 22-feature input space
# and up to 6 output classes. Small enough to train in minutes on CPU.
_HIDDEN_LAYER_SIZES: Tuple[int, ...] = (128, 64)
_MAX_ITER: int = 300
_VALIDATION_FRACTION: float = 0.1
_TEST_SPLIT_FRACTION: float = 0.2


# =============================================================================
# Dataset loading
# =============================================================================

def load_dataset(data_dir: str = DEFAULT_DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all per-gesture CSV files from ``data_dir`` into feature/label arrays.

    Scans ``data_dir`` for files named ``<GESTURE_LABEL>.csv``. Only files
    whose stem (filename without extension) is a member of
    ``CLASSIFIABLE_GESTURES`` are loaded; other files are skipped with a
    warning. Each CSV row must contain exactly ``len(FEATURE_KEYS)`` feature
    columns (in ``FEATURE_KEYS`` order) followed by the label string, with
    no header row — matching the format written by ``data/collector.py``.

    Args:
        data_dir: Directory containing per-gesture CSV files. Defaults to
                  ``data/landmarks`` relative to this module.

    Returns:
        Tuple of ``(X, y)`` where:
            X: numpy array of shape ``(N, len(FEATURE_KEYS))``, dtype float32.
            y: numpy array of shape ``(N,)``, dtype ``<U..`` (string), each
               entry a gesture label from ``CLASSIFIABLE_GESTURES``.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
        ValueError: If no valid CSV files are found, or if any row does not
                    contain exactly ``len(FEATURE_KEYS) + 1`` columns.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir!r}. "
            f"Run data/collector.py first to generate training samples."
        )

    expected_columns = _NUM_FEATURES + 1  # features + label

    features_rows: List[List[float]] = []
    label_rows: List[str] = []
    per_class_counts: dict = {}

    csv_files = sorted(Path(data_dir).glob("*.csv"))

    for csv_path in csv_files:
        gesture_label = csv_path.stem
        if gesture_label not in CLASSIFIABLE_GESTURES:
            print(
                f"Skipping {csv_path.name}: '{gesture_label}' is not a "
                f"recognised gesture label in {CLASSIFIABLE_GESTURES}."
            )
            continue

        count_for_this_file = 0
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) != expected_columns:
                    raise ValueError(
                        f"{csv_path.name} row {row_idx} has {len(row)} columns, "
                        f"expected {expected_columns} "
                        f"({_NUM_FEATURES} features + 1 label). "
                        f"Check that FEATURE_KEYS has not changed since this "
                        f"file was collected."
                    )
                *feature_values, label_value = row
                features_rows.append([float(v) for v in feature_values])
                label_rows.append(label_value)
                count_for_this_file += 1

        per_class_counts[gesture_label] = count_for_this_file

    if not features_rows:
        raise ValueError(
            f"No valid training samples found in {data_dir!r}. "
            f"Run data/collector.py for each gesture in "
            f"{CLASSIFIABLE_GESTURES} before training."
        )

    for label, count in sorted(per_class_counts.items()):
        print(f"{label}: {count} samples")

    X = np.array(features_rows, dtype=np.float32)
    y = np.array(label_rows, dtype=str)

    return X, y


# =============================================================================
# Training and evaluation
# =============================================================================

def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Train a StandardScaler + MLPClassifier pipeline and report test metrics.

    Performs a stratified train/test split so class proportions are
    preserved in both partitions, fits the pipeline on the training split,
    and prints accuracy plus a full per-class precision/recall/F1 report
    evaluated on the held-out test split.

    Args:
        X: Feature array of shape ``(N, len(FEATURE_KEYS))``, as returned
           by ``load_dataset()``.
        y: Label array of shape ``(N,)``, as returned by ``load_dataset()``.

    Returns:
        The fitted ``sklearn.pipeline.Pipeline`` (StandardScaler → MLP),
        trained on the training split only. Ready to be passed to
        ``save_model()``.

    Raises:
        ValueError: If ``X`` and ``y`` have mismatched lengths, or if any
                    class has fewer than 2 samples (required for a
                    stratified split).
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y length mismatch: len(X)={len(X)}, len(y)={len(y)}."
        )

    unique, counts = np.unique(y, return_counts=True)
    insufficient = [cls for cls, cnt in zip(unique, counts) if cnt < 2]
    if insufficient:
        raise ValueError(
            f"Classes with fewer than 2 samples cannot be stratified-split: "
            f"{insufficient}. Collect more samples for these gestures."
        )

    # ── Integer-encode labels before fitting ─────────────────────────────
    # NOTE: scikit-learn's MLPClassifier with early_stopping=True calls an
    # internal scoring routine that applies np.isnan() to predicted labels.
    # When y is a string array, this raises TypeError (numpy cannot apply
    # isnan to string dtype). Encoding y as integers avoids this entirely
    # and is functionally identical to fitting on strings directly — the
    # class ordering (sorted(set(y))) is preserved via `classes_sorted`,
    # which is also what save_model() uses to build the label encoder file.
    # Predictions are decoded back to strings immediately after predict().
    classes_sorted: List[str] = sorted(set(y.tolist()))
    label_to_index = {label: idx for idx, label in enumerate(classes_sorted)}
    y_encoded = np.array([label_to_index[label] for label in y], dtype=np.int64)

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_encoded,
        test_size=_TEST_SPLIT_FRACTION,
        stratify=y_encoded,
        random_state=_RANDOM_STATE,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=_HIDDEN_LAYER_SIZES,
            activation="relu",
            max_iter=_MAX_ITER,
            early_stopping=True,
            validation_fraction=_VALIDATION_FRACTION,
            random_state=_RANDOM_STATE,
        )),
    ])

    pipeline.fit(X_train, y_train_enc)

    y_pred_enc = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test_enc, y_pred_enc)

    # Decode back to string labels for human-readable reporting.
    y_test_labels = [classes_sorted[i] for i in y_test_enc]
    y_pred_labels = [classes_sorted[i] for i in y_pred_enc]

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    return pipeline


# =============================================================================
# Model persistence
# =============================================================================

def save_model(
    pipeline: Pipeline,
    label_classes: List[str],
    output_dir: str = DEFAULT_MODEL_DIR,
) -> Tuple[str, str]:
    """
    Save the trained pipeline and its label class list to disk.

    Two files are written:
    - ``gesture_mlp.joblib``   — the fitted sklearn Pipeline.
    - ``label_encoder.joblib`` — a plain sorted list of class name strings,
      used by ``classifier.MLPClassifier`` to map ``predict_proba()``
      column indices back to gesture label strings.

    Args:
        pipeline:      Fitted pipeline from ``train_and_evaluate()``.
        label_classes: List of gesture label strings, in the same order
                       as the pipeline's internal class ordering. Callers
                       should pass ``sorted(set(y))`` to match sklearn's
                       default alphabetical class ordering.
        output_dir:    Directory to write both files into. Created if it
                       does not already exist.

    Returns:
        Tuple of ``(model_path, encoder_path)`` — absolute paths to the
        two saved files.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path   = os.path.join(output_dir, _MODEL_FILENAME)
    encoder_path = os.path.join(output_dir, _ENCODER_FILENAME)

    joblib.dump(pipeline, model_path)
    joblib.dump(list(label_classes), encoder_path)

    print(f"\nModel saved to:   {model_path}")
    print(f"Label encoder saved to: {encoder_path}")

    return model_path, encoder_path


# =============================================================================
# CLI entry point
# =============================================================================

def run_training(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_MODEL_DIR,
) -> None:
    """
    End-to-end training pipeline: load data, train, evaluate, save.

    Args:
        data_dir:   Directory containing per-gesture CSV files.
        output_dir: Directory to save the trained model and label encoder.
    """
    X, y = load_dataset(data_dir)
    pipeline = train_and_evaluate(X, y)
    save_model(pipeline, sorted(set(y)), output_dir)


if __name__ == "__main__":
    run_training()